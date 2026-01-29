#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class BackgroundSubtractorMOG2GPUCpp {
public:
    BackgroundSubtractorMOG2GPUCpp(
        const py::tuple &frame_shape,
        bool input_is_gray,
        bool enable_timing = true,
        float detect_scale = 1.0f,
        float learning_rate = 0.05f,
        int reference_window_frames = 1,
        float mog2_var_threshold = 20.0f
    )
        : input_is_gray_(input_is_gray),
          detect_scale_(detect_scale > 0.0f ? detect_scale : 1.0f),
          scale_inv_(1.0f / detect_scale_),
          learning_rate_(learning_rate),
          timing_enabled_(enable_timing) {
        // The detector runs most of the pipeline on GPU; CPU is only used for contours.
        if (frame_shape.size() < 2) {
            throw std::runtime_error("frame_shape must have at least 2 elements.");
        }
        int h = frame_shape[0].cast<int>();
        int w = frame_shape[1].cast<int>();
        full_h_ = h;
        full_w_ = w;
        if (detect_scale_ != 1.0f) {
            scaled_h_ = std::max(1, static_cast<int>(std::round(h * detect_scale_)));
            scaled_w_ = std::max(1, static_cast<int>(std::round(w * detect_scale_)));
        } else {
            scaled_h_ = h;
            scaled_w_ = w;
        }
        // Preallocate GPU/CPU buffers to avoid per-frame allocations.
        zeros_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        zeros_gpu_.setTo(cv::Scalar(0));

        mog2_history_ = reference_window_frames > 0 ? reference_window_frames : 1;
        mog2_var_threshold_ = mog2_var_threshold;
        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,   // history length (frames)
            mog2_var_threshold_,    // variance threshold for foreground
            false  // disable shadow detection for speed
        );

        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        frame_small_gpu_.create(scaled_h_, scaled_w_, type);
        blur_small_gpu_.create(scaled_h_, scaled_w_, type);
        blur_full_gpu_.create(full_h_, full_w_, type);
        blur_cpu_.create(full_h_, full_w_, type);
        // Mask is downloaded at low-res for CPU contours.
        mask_cpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        gray_small_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        diff_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        fg_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        prev_gray_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);

        gaussian_filter_ = cv::cuda::createGaussianFilter(
            type,
            type,
            cv::Size(3, 3),
            0.0
        );
        gaussian_filter_type_ = type;

        if (timing_enabled_) {
            timing_ = {
                {"calls", 0.0},
                {"blur", 0.0},
                {"gray", 0.0},
                {"diff", 0.0},
                {"bgsub", 0.0},
                {"contours", 0.0},
                {"upload", 0.0},
                {"download", 0.0},
                {"total", 0.0},
            };
        }
    }

    static bool is_available() {
        try {
            return cv::cuda::getCudaEnabledDeviceCount() > 0;
        } catch (const cv::Exception &) {
            return false;
        }
    }

    py::array detect(const py::array &frame) {
        if (!timing_enabled_) {
            return detect_fast(frame);
        }
        return detect_timed(frame);
    }

    void on_reanchor() {
        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,
            mog2_var_threshold_,
            false
        );
        prev_valid_ = false;
    }

    py::dict timing_summary(bool reset = false) {
        py::dict out;
        if (!timing_enabled_) {
            out["enabled"] = false;
            return out;
        }
        const double calls = timing_["calls"];
        out["enabled"] = true;
        out["calls"] = static_cast<int>(calls);
        if (calls > 0.0) {
            out["total_ms_avg"] = (timing_["total"] / calls) * 1000.0;
            out["blur_ms_avg"] = (timing_["blur"] / calls) * 1000.0;
            out["gray_ms_avg"] = (timing_["gray"] / calls) * 1000.0;
            out["diff_ms_avg"] = (timing_["diff"] / calls) * 1000.0;
            out["bgsub_ms_avg"] = (timing_["bgsub"] / calls) * 1000.0;
            out["contours_ms_avg"] = (timing_["contours"] / calls) * 1000.0;
            out["upload_ms_avg"] = (timing_["upload"] / calls) * 1000.0;
            out["download_ms_avg"] = (timing_["download"] / calls) * 1000.0;
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    void reset_timing() {
        if (!timing_enabled_) {
            return;
        }
        for (auto &kv : timing_) {
            kv.second = 0.0;
        }
    }

    py::array get_last_mask() {
        if (mask_cpu_.empty()) {
            return py::array();
        }
        cv::Mat mask_out;
        if (detect_scale_ == 1.0f) {
            mask_out = mask_cpu_;
        } else {
            cv::resize(mask_cpu_, mask_out, cv::Size(full_w_, full_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }
        return mat_to_array(mask_out);
    }

    py::array get_last_cc_mask() {
        if (cc_mask_cpu_.empty()) {
            return py::array();
        }
        return mat_to_array(cc_mask_cpu_);
    }

    py::array get_last_bb_mask() {
        if (bb_mask_cpu_.empty()) {
            return py::array();
        }
        return mat_to_array(bb_mask_cpu_);
    }

private:
    py::array detect_fast(const py::array &frame) {
        cv::Mat frame_cpu = ensure_mat(frame);
        upload_and_scale(frame_cpu);
        gaussian_blur();
        calc_diff();
        backsub_->apply(blur_small_gpu_, fg_gpu_, learning_rate_);
        fg_gpu_.copyTo(mask_gpu_);
        return finalize_detections();
    }

    py::array detect_timed(const py::array &frame) {
        const auto start = now();
        cv::Mat frame_cpu = ensure_mat(frame);
        upload_and_scale(frame_cpu);
        gaussian_blur();
        const auto t_blur = now();
        timing_["blur"] += elapsed_seconds(start, t_blur);

        calc_diff();
        const auto t_diff = now();
        timing_["gray"] += elapsed_seconds(t_blur, t_diff);
        timing_["diff"] += elapsed_seconds(t_blur, t_diff);

        backsub_->apply(blur_small_gpu_, fg_gpu_, learning_rate_);
        const auto t_fg = now();
        timing_["bgsub"] += elapsed_seconds(t_diff, t_fg);

        fg_gpu_.copyTo(mask_gpu_);

        py::array detections = finalize_detections();
        const auto t_contours = now();
        timing_["contours"] += elapsed_seconds(t_fg, t_contours);
        timing_["total"] += elapsed_seconds(start, t_contours);
        timing_["calls"] += 1.0;
        return detections;
    }

    cv::Mat ensure_mat(const py::array &array) const {
        py::array arr = py::array::ensure(array, py::array::c_style | py::array::forcecast);
        if (!arr) {
            throw std::runtime_error("Expected a contiguous numpy array.");
        }
        py::buffer_info info = arr.request();
        if (info.ndim != 2 && info.ndim != 3) {
            throw std::runtime_error("Expected a 2D or 3D numpy array.");
        }
        if (info.format != py::format_descriptor<uint8_t>::format()) {
            throw std::runtime_error("Expected uint8 numpy array.");
        }
        int height = static_cast<int>(info.shape[0]);
        int width = static_cast<int>(info.shape[1]);
        if (info.ndim == 2) {
            cv::Mat mat(height, width, CV_8UC1);
            std::memcpy(mat.data, info.ptr, mat.total() * mat.elemSize());
            return mat;
        }
        int channels = static_cast<int>(info.shape[2]);
        if (channels != 3) {
            throw std::runtime_error("Expected 3-channel BGR numpy array.");
        }
        cv::Mat mat(height, width, CV_8UC3);
        std::memcpy(mat.data, info.ptr, mat.total() * mat.elemSize());
        return mat;
    }

    py::array mat_to_array(const cv::Mat &mat) const {
        cv::Mat contiguous = mat;
        if (!mat.isContinuous()) {
            contiguous = mat.clone();
        }
        const int channels = contiguous.channels();
        std::vector<ssize_t> shape;
        if (channels == 1) {
            shape = {contiguous.rows, contiguous.cols};
        } else {
            shape = {contiguous.rows, contiguous.cols, channels};
        }
        py::array out(py::dtype::of<uint8_t>(), shape);
        std::memcpy(out.mutable_data(), contiguous.data, contiguous.total() * contiguous.elemSize());
        return out;
    }

    void upload_and_scale(const cv::Mat &frame_cpu) {
        // Upload to GPU and downscale for detector if requested.
        const auto t_upload_start = timing_enabled_ ? now() : std::chrono::steady_clock::time_point();
        frame_gpu_.upload(frame_cpu);
        if (timing_enabled_) {
            timing_["upload"] += elapsed_seconds(t_upload_start, now());
        }
        if (detect_scale_ == 1.0f) {
            return;
        }
        cv::cuda::resize(
            frame_gpu_,
            frame_small_gpu_,
            cv::Size(scaled_w_, scaled_h_),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    void gaussian_blur() {
        // Apply prebuilt Gaussian filter on the scaled frame.
        const cv::cuda::GpuMat &src = detect_scale_ == 1.0f ? frame_gpu_ : frame_small_gpu_;
        gaussian_filter_->apply(src, blur_small_gpu_);
    }

    void upscale_blur_and_move_to_cpu() {
        // Optional upsample for visualization and download to CPU.
        if (detect_scale_ == 1.0f) {
            blur_full_gpu_ = blur_small_gpu_;
            const auto t_dl_start = timing_enabled_ ? now() : std::chrono::steady_clock::time_point();
            blur_full_gpu_.download(blur_cpu_);
            if (timing_enabled_) {
                timing_["download"] += elapsed_seconds(t_dl_start, now());
            }
            return;
        }
        cv::cuda::resize(
            blur_small_gpu_,
            blur_full_gpu_,
            cv::Size(full_w_, full_h_),
            0.0,
            0.0,
            cv::INTER_LINEAR
        );
        const auto t_dl_start = timing_enabled_ ? now() : std::chrono::steady_clock::time_point();
        blur_full_gpu_.download(blur_cpu_);
        if (timing_enabled_) {
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        }
    }

    void to_gray() {
        // Convert blur output to grayscale for diff/background subtraction.
        if (input_is_gray_) {
            gray_small_gpu_ = blur_small_gpu_;
            return;
        }
        cv::cuda::cvtColor(blur_small_gpu_, gray_small_gpu_, cv::COLOR_BGR2GRAY);
    }

    void calc_diff() {
        // Frame differencing with threshold on the scaled grayscale image.
        to_gray();
        const cv::cuda::GpuMat &gray_small = gray_small_gpu_;
        if (prev_valid_) {
            cv::cuda::absdiff(prev_gray_gpu_, gray_small, diff_gpu_);
            cv::cuda::threshold(diff_gpu_, diff_gpu_, 25.0, 255.0, cv::THRESH_BINARY);
        } else {
            prev_valid_ = true;
            gray_small.copyTo(prev_gray_gpu_);
            diff_gpu_ = zeros_gpu_;
            return;
        }
        gray_small.copyTo(prev_gray_gpu_);
    }

    // CPU contour extraction from the downloaded low-res mask.
    py::array finalize_detections() {
        const auto t_dl_start = timing_enabled_ ? now() : std::chrono::steady_clock::time_point();
        mask_gpu_.download(mask_cpu_);
        if (timing_enabled_) {
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        }

        if (close_iterations_ > 0 || open_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            if (close_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_CLOSE,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    close_iterations_
                );
            }
            if (open_iterations_ > 0) {
                cv::morphologyEx(
                    mask_cpu_,
                    mask_cpu_,
                    cv::MORPH_OPEN,
                    morph_kernel_,
                    cv::Point(-1, -1),
                    open_iterations_
                );
            }
        }
        if (dilate_iterations_ > 0) {
            if (morph_kernel_.empty()) {
                morph_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            }
            cv::dilate(mask_cpu_, mask_cpu_, morph_kernel_, cv::Point(-1, -1), dilate_iterations_);
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask_cpu_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cc_mask_cpu_ = cv::Mat::zeros(mask_cpu_.size(), CV_8UC3);
        if (!contours.empty()) {
            for (size_t i = 0; i < contours.size(); ++i) {
                cv::Scalar color(
                    static_cast<double>((37 * i) % 255),
                    static_cast<double>((17 * i) % 255),
                    static_cast<double>((97 * i) % 255)
                );
                cv::drawContours(cc_mask_cpu_, contours, static_cast<int>(i), color, cv::FILLED);
            }
        }
        if (detect_scale_ != 1.0f) {
            cv::resize(cc_mask_cpu_, cc_mask_cpu_, cv::Size(full_w_, full_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }

        std::vector<cv::Rect> detections;
        detections.reserve(contours.size());
        for (const auto &c : contours) {
            double area = cv::contourArea(c);
            if (!(min_contour_area_ < area && area < max_contour_area_)) {
                continue;
            }
            cv::Rect bbox = cv::boundingRect(c);
            if (bbox.width <= 0 || bbox.height <= 0) {
                continue;
            }
            double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
            if (!(0.2 <= aspect && aspect <= 5.0)) {
                continue;
            }
            if (detect_scale_ != 1.0f) {
                int x = static_cast<int>(std::round(bbox.x * scale_inv_));
                int y = static_cast<int>(std::round(bbox.y * scale_inv_));
                int w = static_cast<int>(std::round(bbox.width * scale_inv_));
                int h = static_cast<int>(std::round(bbox.height * scale_inv_));
                bbox = cv::Rect(x, y, w, h);
            }
            detections.push_back(bbox);
        }

        if (cc_mask_cpu_.channels() == 3) {
            bb_mask_cpu_ = cc_mask_cpu_.clone();
        } else {
            bb_mask_cpu_ = cv::Mat::zeros(cv::Size(full_w_, full_h_), CV_8UC3);
        }
        for (const auto &bbox : detections) {
            cv::rectangle(bb_mask_cpu_, bbox, cv::Scalar(0, 0, 255), 1);
        }

        py::array detections_arr(
            py::dtype::of<int32_t>(),
            {static_cast<py::ssize_t>(detections.size()), static_cast<py::ssize_t>(4)}
        );
        auto buf = detections_arr.mutable_unchecked<int32_t, 2>();
        for (ssize_t i = 0; i < static_cast<ssize_t>(detections.size()); ++i) {
            const cv::Rect &r = detections[i];
            buf(i, 0) = r.x;
            buf(i, 1) = r.y;
            buf(i, 2) = r.width;
            buf(i, 3) = r.height;
        }

        return detections_arr;
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                                  const std::chrono::steady_clock::time_point &end) {
        return std::chrono::duration<double>(end - start).count();
    }

    bool input_is_gray_ = true;
    float detect_scale_ = 1.0f;
    float scale_inv_ = 1.0f;
    float learning_rate_ = 0.05f;
    int full_h_ = 0;
    int full_w_ = 0;
    int scaled_h_ = 0;
    int scaled_w_ = 0;

    bool timing_enabled_ = true;
    std::map<std::string, double> timing_;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> backsub_;
    cv::Ptr<cv::cuda::Filter> gaussian_filter_;
    int gaussian_filter_type_ = -1;

    cv::cuda::GpuMat frame_gpu_;
    cv::cuda::GpuMat frame_small_gpu_;
    cv::cuda::GpuMat blur_small_gpu_;
    cv::cuda::GpuMat blur_full_gpu_;
    cv::cuda::GpuMat gray_small_gpu_;
    cv::cuda::GpuMat diff_gpu_;
    cv::cuda::GpuMat fg_gpu_;
    cv::cuda::GpuMat mask_gpu_;
    cv::cuda::GpuMat zeros_gpu_;
    cv::cuda::GpuMat prev_gray_gpu_;
    cv::Mat blur_cpu_;
    cv::Mat mask_cpu_;
    cv::Mat cc_mask_cpu_;
    cv::Mat bb_mask_cpu_;
    bool prev_valid_ = false;

    int mog2_history_ = 1;
    float mog2_var_threshold_ = 20.0f;
    cv::Mat morph_kernel_;
    int close_iterations_ = 1;
    int open_iterations_ = 1;
    int dilate_iterations_ = 2;
    double min_contour_area_ = 50.0;
    double max_contour_area_ = 1500.0;
};

PYBIND11_MODULE(_detector_cpp, m) {
    m.doc() = "GPU background subtraction detector (C++/pybind11).";

    py::class_<BackgroundSubtractorMOG2GPUCpp>(m, "BackgroundSubtractorMOG2GPUCpp")
        .def(py::init<const py::tuple &, bool, bool, float, float, int, float>(),
             py::arg("frame_shape"),
             py::arg("input_is_gray"),
             py::arg("enable_timing") = true,
             py::arg("detect_scale") = 1.0f,
             py::arg("learning_rate") = 0.05f,
             py::arg("reference_window_frames") = 1,
             py::arg("mog2_var_threshold") = 20.0f)
        .def_static("is_available", &BackgroundSubtractorMOG2GPUCpp::is_available)
        .def("detect", &BackgroundSubtractorMOG2GPUCpp::detect)
        .def("on_reanchor", &BackgroundSubtractorMOG2GPUCpp::on_reanchor)
        .def("timing_summary", &BackgroundSubtractorMOG2GPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("reset_timing", &BackgroundSubtractorMOG2GPUCpp::reset_timing)
        .def("get_last_mask", &BackgroundSubtractorMOG2GPUCpp::get_last_mask)
        .def("get_last_cc_mask", &BackgroundSubtractorMOG2GPUCpp::get_last_cc_mask)
        .def("get_last_bb_mask", &BackgroundSubtractorMOG2GPUCpp::get_last_bb_mask);
}
