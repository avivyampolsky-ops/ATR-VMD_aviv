#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

struct BBox {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

class KalmanIoUTrackerCpp {
public:
    struct Entity {
        int id = 0;
        BBox bbox;
        cv::KalmanFilter kf;
        int age = 1;
        int lost = 0;
        bool moving = false;
        float ema_area = 0.0f;

        py::array bbox_array() const {
            py::array_t<int32_t> out(4);
            auto buf = out.mutable_unchecked<1>();
            buf(0) = static_cast<int32_t>(std::round(bbox.x));
            buf(1) = static_cast<int32_t>(std::round(bbox.y));
            buf(2) = static_cast<int32_t>(std::round(bbox.w));
            buf(3) = static_cast<int32_t>(std::round(bbox.h));
            return out;
        }
    };

    KalmanIoUTrackerCpp(
        float iou_thresh = 0.05f,
        int max_lost = 8,
        float min_move = 2.0f,
        float ema_alpha = 0.3f,
        float dist_gate_scale = 2.0f,
        float easy_iou_thresh = 0.6f
    )
        : iou_thresh_(iou_thresh),
          max_lost_(max_lost),
          min_move_(min_move),
          ema_alpha_(ema_alpha),
          dist_gate_scale_(dist_gate_scale),
          easy_iou_thresh_(easy_iou_thresh) {}

    void apply_shift(float dx, float dy) {
        // Compensate tracker state for registration jumps so IoU matching stays stable.
        if (dx == 0.0f && dy == 0.0f) {
            return;
        }
        for (auto &kv : tracks_) {
            Entity &entity = kv.second;
            entity.bbox.x += dx;
            entity.bbox.y += dy;
            entity.kf.statePost.at<float>(0, 0) += dx;
            entity.kf.statePost.at<float>(1, 0) += dy;
            entity.kf.statePre.at<float>(0, 0) += dx;
            entity.kf.statePre.at<float>(1, 0) += dy;
        }
    }

    void reset() {
        tracks_.clear();
        current_id_ = 0;
    }

    const std::map<int, Entity> &get_tracks() const {
        return tracks_;
    }

    std::map<int, Entity> update(const py::array &observations) {
        std::vector<BBox> obs = parse_observations(observations);
        const int num_obs = static_cast<int>(obs.size());
        std::vector<bool> used_mask(num_obs, false);
        std::map<int, Entity> updated_tracks;

        // Predict
        for (auto &kv : tracks_) {
            kv.second.kf.predict();
        }

        std::vector<std::pair<int, Entity>> track_items;
        track_items.reserve(tracks_.size());
        for (const auto &kv : tracks_) {
            track_items.emplace_back(kv.first, kv.second);
        }

        std::vector<std::vector<float>> iou_matrix = build_iou_matrix(track_items, obs);
        std::vector<int> unmatched_tracks;

        if (!track_items.empty() && num_obs > 0) {
            for (size_t t_idx = 0; t_idx < track_items.size(); ++t_idx) {
                float best_iou = -1.0f;
                int best_idx = -1;
                for (int o = 0; o < num_obs; ++o) {
                    if (used_mask[o]) {
                        continue;
                    }
                    float val = iou_matrix[t_idx][o];
                    if (val > best_iou) {
                        best_iou = val;
                        best_idx = o;
                    }
                }

                if (best_iou >= easy_iou_thresh_) {
                    Entity entity = track_items[t_idx].second;
                    apply_match(entity, obs[best_idx]);
                    updated_tracks[track_items[t_idx].first] = entity;
                    used_mask[best_idx] = true;
                } else {
                    unmatched_tracks.push_back(static_cast<int>(t_idx));
                }
            }
        } else {
            for (size_t t_idx = 0; t_idx < track_items.size(); ++t_idx) {
                unmatched_tracks.push_back(static_cast<int>(t_idx));
            }
        }

        // Remaining greedy matches
        for (int t_idx : unmatched_tracks) {
            int track_id = track_items[t_idx].first;
            Entity entity = track_items[t_idx].second;
            float best_iou = -1.0f;
            int best_idx = -1;
            if (num_obs > 0 && !iou_matrix.empty()) {
                for (int o = 0; o < num_obs; ++o) {
                    if (used_mask[o]) {
                        continue;
                    }
                    float val = iou_matrix[t_idx][o];
                    if (val > best_iou) {
                        best_iou = val;
                        best_idx = o;
                    }
                }
            }

            if (best_iou >= iou_thresh_) {
                apply_match(entity, obs[best_idx]);
                updated_tracks[track_id] = entity;
                used_mask[best_idx] = true;
            } else {
                entity.lost += 1;
                if (entity.lost <= max_lost_) {
                    const cv::Mat &pred = entity.kf.statePre;
                    float cx = pred.at<float>(0, 0);
                    float cy = pred.at<float>(1, 0);
                    float w = entity.bbox.w;
                    float h = entity.bbox.h;
                    entity.bbox = {cx - w * 0.5f, cy - h * 0.5f, w, h};
                    updated_tracks[track_id] = entity;
                }
            }
        }

        // New tracks
        for (int i = 0; i < num_obs; ++i) {
            if (!used_mask[i]) {
                Entity entity;
                entity.id = current_id_;
                entity.bbox = obs[i];
                entity.kf = create_kalman();
                entity.age = 1;
                entity.lost = 0;
                entity.moving = false;
                entity.ema_area = obs[i].w * obs[i].h;
                init_kalman_state(entity.kf, obs[i]);
                updated_tracks[current_id_] = entity;
                current_id_ += 1;
            }
        }

        tracks_ = updated_tracks;
        return tracks_;
    }

private:
    static void init_kalman_state(cv::KalmanFilter &kf, const BBox &bbox) {
        float cx = bbox.x + bbox.w * 0.5f;
        float cy = bbox.y + bbox.h * 0.5f;
        kf.statePost = (cv::Mat_<float>(4, 1) << cx, cy, 0.0f, 0.0f);
    }

    static cv::KalmanFilter create_kalman() {
        cv::KalmanFilter kf(4, 2, 0, CV_32F);

        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0
        );

        kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.03f;
        kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F);
        kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);

        return kf;
    }

    void apply_match(Entity &entity, const BBox &bbox) {
        float cx = bbox.x + bbox.w * 0.5f;
        float cy = bbox.y + bbox.h * 0.5f;
        cv::Mat measurement = (cv::Mat_<float>(2, 1) << cx, cy);
        entity.kf.correct(measurement);

        float px = entity.bbox.x + entity.bbox.w * 0.5f;
        float py = entity.bbox.y + entity.bbox.h * 0.5f;
        float dist = std::hypot(cx - px, cy - py);

        float ema_area = (ema_alpha_ * (bbox.w * bbox.h)) +
                         ((1.0f - ema_alpha_) * entity.ema_area);

        entity.bbox = bbox;
        entity.age += 1;
        entity.lost = 0;
        entity.moving = dist >= min_move_;
        entity.ema_area = ema_area;
    }

    static float iou(const BBox &a, const BBox &b) {
        float xA = std::max(a.x, b.x);
        float yA = std::max(a.y, b.y);
        float xB = std::min(a.x + a.w, b.x + b.w);
        float yB = std::min(a.y + a.h, b.y + b.h);
        float interW = std::max(0.0f, xB - xA);
        float interH = std::max(0.0f, yB - yA);
        float inter = interW * interH;
        float union_area = a.w * a.h + b.w * b.h - inter;
        if (union_area <= 0.0f) {
            return 0.0f;
        }
        return inter / union_area;
    }

    std::vector<std::vector<float>> build_iou_matrix(
        const std::vector<std::pair<int, Entity>> &track_items,
        const std::vector<BBox> &obs
    ) const {
        const int num_tracks = static_cast<int>(track_items.size());
        const int num_obs = static_cast<int>(obs.size());
        std::vector<std::vector<float>> iou_matrix(
            num_tracks,
            std::vector<float>(num_obs, -1.0f)
        );
        if (num_tracks == 0 || num_obs == 0) {
            return iou_matrix;
        }

        std::vector<float> obs_area(num_obs);
        std::vector<float> obs_cx(num_obs);
        std::vector<float> obs_cy(num_obs);
        for (int i = 0; i < num_obs; ++i) {
            obs_area[i] = obs[i].w * obs[i].h;
            obs_cx[i] = obs[i].x + obs[i].w * 0.5f;
            obs_cy[i] = obs[i].y + obs[i].h * 0.5f;
        }

        for (int t_idx = 0; t_idx < num_tracks; ++t_idx) {
            const Entity &entity = track_items[t_idx].second;
            float ema_area = entity.ema_area;
            const cv::Mat &pred = entity.kf.statePre;
            float pred_cx = pred.at<float>(0, 0);
            float pred_cy = pred.at<float>(1, 0);
            float dist_gate = dist_gate_scale_ * std::max(entity.bbox.w, entity.bbox.h);

            for (int i = 0; i < num_obs; ++i) {
                float area = obs_area[i];
                bool area_ok = (area >= 0.5f * ema_area) && (area <= 2.0f * ema_area);
                float dist = std::hypot(obs_cx[i] - pred_cx, obs_cy[i] - pred_cy);
                bool dist_ok = dist <= dist_gate;
                if (!area_ok || !dist_ok) {
                    continue;
                }
                iou_matrix[t_idx][i] = iou(entity.bbox, obs[i]);
            }
        }

        return iou_matrix;
    }

    static std::vector<BBox> parse_observations(const py::array &observations) {
        // Parse Nx4 detections into a simple bbox list for matching.
        auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(observations);
        if (!arr) {
            throw std::runtime_error("Expected a contiguous numpy array.");
        }
        py::buffer_info info = arr.request();
        if (info.ndim != 2 || info.shape[1] != 4) {
            throw std::runtime_error("Expected Nx4 observations array.");
        }
        const int count = static_cast<int>(info.shape[0]);
        const auto *ptr = static_cast<const float *>(info.ptr);
        std::vector<BBox> out;
        out.reserve(count);
        for (int i = 0; i < count; ++i) {
            float x = ptr[i * 4 + 0];
            float y = ptr[i * 4 + 1];
            float w = ptr[i * 4 + 2];
            float h = ptr[i * 4 + 3];
            out.push_back({x, y, w, h});
        }
        return out;
    }

    float iou_thresh_ = 0.05f;
    int max_lost_ = 8;
    float min_move_ = 2.0f;
    float ema_alpha_ = 0.3f;
    float dist_gate_scale_ = 2.0f;
    float easy_iou_thresh_ = 0.6f;

    int current_id_ = 0;
    std::map<int, Entity> tracks_;
};

PYBIND11_MODULE(_kalman_cpp, m) {
    m.doc() = "Kalman IoU tracker (C++/pybind11).";

    py::class_<KalmanIoUTrackerCpp::Entity>(m, "KalmanEntity")
        .def_property_readonly("id", [](const KalmanIoUTrackerCpp::Entity &e) { return e.id; })
        .def_property_readonly("bbox", &KalmanIoUTrackerCpp::Entity::bbox_array)
        .def_property_readonly("age", [](const KalmanIoUTrackerCpp::Entity &e) { return e.age; })
        .def_property_readonly("lost", [](const KalmanIoUTrackerCpp::Entity &e) { return e.lost; })
        .def_property_readonly("moving", [](const KalmanIoUTrackerCpp::Entity &e) { return e.moving; })
        .def_property_readonly("ema_area", [](const KalmanIoUTrackerCpp::Entity &e) { return e.ema_area; });

    py::class_<KalmanIoUTrackerCpp>(m, "KalmanIoUTrackerCpp")
        .def(py::init<float, int, float, float, float, float>(),
             py::arg("iou_thresh") = 0.05f,
             py::arg("max_lost") = 8,
             py::arg("min_move") = 2.0f,
             py::arg("ema_alpha") = 0.3f,
             py::arg("dist_gate_scale") = 2.0f,
             py::arg("easy_iou_thresh") = 0.6f)
        .def("apply_shift", &KalmanIoUTrackerCpp::apply_shift)
        .def("reset", &KalmanIoUTrackerCpp::reset)
        .def("update", &KalmanIoUTrackerCpp::update)
        .def("get_tracks", &KalmanIoUTrackerCpp::get_tracks, py::return_value_policy::reference_internal);
}
