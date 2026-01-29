#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class RegisterAndDetectGPUCpp {
public:
    RegisterAndDetectGPUCpp(
        const py::array &reference,
        bool input_is_gray,
        float downscale_factor = 1.0f,
        float detect_scale = 1.0f,
        float learning_rate = 0.05f,
        int reference_window_frames = 1,
        float mog2_var_threshold = 20.0f,
        float phase_response_threshold = 0.1f,
        int reanchor_boost_frames = 0,
        const std::string &reanchor_log_path = "",
        float reanchor_boost_lr = 0.0f,
        float max_shift = 0.0f,
        bool phase_use_cached_fft = true,
        bool enable_timing = true
    )
        : input_is_gray_(input_is_gray),
          downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          detect_scale_(detect_scale > 0.0f ? detect_scale : 1.0f),
          scale_inv_(1.0f / detect_scale_),
          learning_rate_(learning_rate),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          mog2_history_(reference_window_frames_),
          mog2_var_threshold_(mog2_var_threshold),
          phase_response_threshold_(phase_response_threshold),
          reanchor_boost_frames_(reanchor_boost_frames > 0 ? reanchor_boost_frames : 0),
          max_shift_(max_shift > 0.0f ? max_shift : 0.0f),
          reanchor_boost_lr_(reanchor_boost_lr > 0.0f ? reanchor_boost_lr : learning_rate_),
          phase_use_cached_fft_(phase_use_cached_fft),
          timing_enabled_(enable_timing) {
        cv::Mat ref_mat = ensure_mat(reference);
        set_geometry(ref_mat);
        set_reference_mat(ref_mat, false);
        init_detector_buffers();

        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,   // history length (frames)
            mog2_var_threshold,    // variance threshold for foreground
            false  // disable shadow detection for speed
        );

        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        gaussian_filter_ = cv::cuda::createGaussianFilter(
            type,
            type,
            cv::Size(3, 3),
            0.0
        );

        if (!reanchor_log_path.empty()) {
            reanchor_log_.open(reanchor_log_path, std::ios::out);
            if (reanchor_log_.is_open()) {
                reanchor_log_ << "frame,response,phase_ok,dx,dy,reason\n";
                reanchor_log_.flush();
            }
        }

        if (timing_enabled_) {
            timing_ = {
                {"calls", 0.0},
                {"registration", 0.0},
                {"blur", 0.0},
                {"gray", 0.0},
                {"diff", 0.0},
                {"bgsub", 0.0},
                {"contours", 0.0},
                {"upload", 0.0},
                {"download", 0.0},
                {"bbox_creation", 0.0},
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
            out["registration_ms_avg"] = (timing_["registration"] / calls) * 1000.0;
            out["blur_ms_avg"] = (timing_["blur"] / calls) * 1000.0;
            out["gray_ms_avg"] = (timing_["gray"] / calls) * 1000.0;
            out["diff_ms_avg"] = (timing_["diff"] / calls) * 1000.0;
            out["bgsub_ms_avg"] = (timing_["bgsub"] / calls) * 1000.0;
            out["contours_ms_avg"] = (timing_["contours"] / calls) * 1000.0;
            out["upload_ms_avg"] = (timing_["upload"] / calls) * 1000.0;
            out["download_ms_avg"] = (timing_["download"] / calls) * 1000.0;
            out["bbox_creation_ms_avg"] = (timing_["bbox_creation"] / calls) * 1000.0;
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

    py::tuple register_and_detect(const py::array &frame) {
        const auto start = now();
        cv::Mat frame_cpu = ensure_mat(frame);
        if (input_is_gray_ && frame_cpu.channels() != 1) {
            throw std::runtime_error("Expected grayscale frame for GPU register/detect.");
        }
        if (!input_is_gray_ && frame_cpu.channels() != 3) {
            throw std::runtime_error("Expected BGR frame for GPU register/detect.");
        }

        frame_count_ += 1;
        const auto t_reg_start = now();
        cv::cuda::GpuMat reg_gpu = register_frame_gpu(frame_cpu);
        if (timing_enabled_) {
            timing_["registration"] += elapsed_seconds(t_reg_start, now());
        }

        detect_from_registered(reg_gpu);

        // CPU contour extraction from the downloaded low-res mask.
        py::array detections = finalize_detections();
        if (timing_enabled_) {
            const auto t_dl_start = now();
            reg_gpu.download(registered_cpu_);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            reg_gpu.download(registered_cpu_);
        }
        if (timing_enabled_) {
            const auto end = now();
            timing_["total"] += elapsed_seconds(start, end);
            timing_["calls"] += 1.0;
        }

        return py::make_tuple(detections, mat_to_array(registered_cpu_));
    }

    py::array get_last_mask() {
        if (mask_cpu_.empty()) {
            return py::array();
        }
        cv::Mat mask_out;
        if (detect_scale_ == 1.0f) {
            mask_out = mask_cpu_;
        } else {
            cv::resize(mask_cpu_, mask_out, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
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

    py::array get_reference_events() const {
        py::array_t<int32_t> out(reference_events_.size());
        auto buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(reference_events_.size()); ++i) {
            buf(i) = reference_events_[static_cast<size_t>(i)];
        }
        return out;
    }

    py::tuple get_last_shift() const {
        return py::make_tuple(last_dx_, last_dy_);
    }

    py::dict get_last_registration_debug() const {
        py::dict out;
        out["phase_response_threshold"] = phase_response_threshold_;
        out["response"] = last_response_;
        out["phase_ok"] = last_phase_ok_;
        out["dx"] = last_dx_;
        out["dy"] = last_dy_;
        out["reanchor_reason"] = last_reanchor_reason_;
        return out;
    }

private:
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

    void set_geometry(const cv::Mat &reference) {
        h_ = reference.rows;
        w_ = reference.cols;
        margin_ = static_cast<int>(std::min(h_, w_) * 0.01f);
        cropped_h_ = h_ - margin_ * 2;
        cropped_w_ = w_ - margin_ * 2;
    }

    void set_reference_mat(const cv::Mat &reference, bool record_event) {
        gray_ref_gpu_ = prepare_gray(reference);
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
        }
        if (record_event) {
            reference_events_.push_back(static_cast<int>(frame_count_));
            if (reanchor_boost_frames_ > 0) {
                reanchor_boost_left_ = reanchor_boost_frames_;
            }
            log_reanchor_event();
        }
    }

    void set_reference_gray(const cv::cuda::GpuMat &gray_f) {
        gray_ref_gpu_ = gray_f.clone();
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
        }
    }

    cv::cuda::GpuMat prepare_gray(const cv::Mat &frame_cpu) {
        // Upload and convert to float32 grayscale (optionally downscaled) for phase correlation.
        if (timing_enabled_) {
            const auto t_up_start = now();
            frame_gpu_.upload(frame_cpu);
            timing_["upload"] += elapsed_seconds(t_up_start, now());
        } else {
            frame_gpu_.upload(frame_cpu);
        }
        if (input_is_gray_) {
            gray_gpu_ = frame_gpu_;
        } else {
            cv::cuda::cvtColor(frame_gpu_, gray_gpu_, cv::COLOR_BGR2GRAY);
        }
        cv::cuda::GpuMat scaled;
        if (downscale_factor_ != 1.0f) {
            cv::cuda::resize(
                gray_gpu_,
                scaled,
                cv::Size(),
                downscale_factor_,
                downscale_factor_,
                cv::INTER_AREA
            );
        } else {
            scaled = gray_gpu_;
        }
        cv::cuda::GpuMat gray_f;
        scaled.convertTo(gray_f, CV_32F);
        return gray_f;
    }

    bool phase_correlate_gpu(
        const cv::cuda::GpuMat &gray_gpu,
        float &dx,
        float &dy,
        float &response
    ) {
        // GPU phase correlation: compute cross-power spectrum and locate peak shift.
        if (gray_ref_gpu_.empty()) {
            return false;
        }
        cv::cuda::GpuMat ref_fft;
        if (phase_use_cached_fft_) {
            if (ref_fft_gpu_.empty()) {
                cv::cuda::dft(gray_ref_gpu_, ref_fft_gpu_, gray_ref_gpu_.size());
            }
            ref_fft = ref_fft_gpu_;
        } else {
            cv::cuda::dft(gray_ref_gpu_, ref_fft, gray_ref_gpu_.size());
        }

        cv::cuda::GpuMat cur_fft;
        cv::cuda::dft(gray_gpu, cur_fft, gray_gpu.size());

        cv::cuda::GpuMat cps;
        cv::cuda::mulSpectrums(cur_fft, ref_fft, cps, 0, true);

        std::vector<cv::cuda::GpuMat> planes;
        cv::cuda::split(cps, planes);

        cv::cuda::GpuMat mag;
        cv::cuda::magnitude(planes[0], planes[1], mag);
        cv::cuda::addWeighted(mag, 1.0, mag, 0.0, 1e-9, mag);
        cv::cuda::divide(planes[0], mag, planes[0]);
        cv::cuda::divide(planes[1], mag, planes[1]);
        cv::cuda::merge(planes, cps);

        cv::cuda::GpuMat corr;
        cv::cuda::dft(
            cps,
            corr,
            cps.size(),
            cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT
        );

        cv::Mat corr_cpu;
        if (timing_enabled_) {
            const auto t_dl_start = now();
            corr.download(corr_cpu);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            corr.download(corr_cpu);
        }
        double max_val = 0.0;
        cv::Point max_loc;
        cv::minMaxLoc(corr_cpu, nullptr, &max_val, nullptr, &max_loc);
        const int h = corr_cpu.rows;
        const int w = corr_cpu.cols;
        const float shift_x = (max_loc.x <= w / 2) ? static_cast<float>(max_loc.x)
                                                   : static_cast<float>(max_loc.x - w);
        const float shift_y = (max_loc.y <= h / 2) ? static_cast<float>(max_loc.y)
                                                   : static_cast<float>(max_loc.y - h);
        dx = shift_x / downscale_factor_;
        dy = shift_y / downscale_factor_;
        response = static_cast<float>(max_val);
        last_response_ = response;
        return true;
    }

    cv::cuda::GpuMat register_frame_gpu(const cv::Mat &frame_cpu) {
        // Compute translation and warp on GPU.
        float dx = 0.0f;
        float dy = 0.0f;
        float response = 0.0f;
        cv::cuda::GpuMat gray_f = prepare_gray(frame_cpu);
        last_phase_ok_ = false;
        last_reanchor_reason_ = "none";
        bool phase_ok = phase_correlate_gpu(gray_f, dx, dy, response) &&
                        response >= phase_response_threshold_;
        if (phase_ok && max_shift_ > 0.0f) {
            if (std::abs(dx) > max_shift_ || std::abs(dy) > max_shift_) {
                phase_ok = false;
                last_reanchor_reason_ = "max_shift";
            }
        }
        bool ok = phase_ok;
        last_phase_ok_ = phase_ok;
        if (!ok) {
            if (!phase_ok) {
                last_reanchor_reason_ = "phase";
            }
        }
        if (!ok) {
            last_dx_ = 0.0f;
            last_dy_ = 0.0f;
            if (last_reanchor_reason_ == "none") {
                last_reanchor_reason_ = "fallback_fail";
            }
            reg_gpu_ = crop_gpu(frame_gpu_);
        } else {
            last_dx_ = dx;
            last_dy_ = dy;
            cv::Mat warp = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, dx, 0.0f, 1.0f, dy);
            cv::cuda::warpAffine(
                frame_gpu_,
                reg_gpu_,
                warp,
                cv::Size(w_, h_),
                cv::INTER_LINEAR
            );
            reg_gpu_ = crop_gpu(reg_gpu_);
        }
        if (should_update_reference()) {
            set_reference_gray(gray_f);
            record_reference_event(ok ? "window" : "window_fallback");
            reset_background_model();
        }
        return reg_gpu_;
    }

    cv::cuda::GpuMat crop_gpu(const cv::cuda::GpuMat &mat) const {
        if (margin_ == 0) {
            return mat;
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return mat(roi);
    }

    void log_reanchor_event() {
        if (!reanchor_log_.is_open()) {
            return;
        }
        reanchor_log_
            << frame_count_ << ","
            << last_response_ << ","
            << (last_phase_ok_ ? 1 : 0) << ","
            << last_dx_ << ","
            << last_dy_ << ","
            << last_reanchor_reason_
            << "\n";
        reanchor_log_.flush();
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void record_reference_event(const std::string &reason) {
        last_reanchor_reason_ = reason;
        reference_events_.push_back(static_cast<int>(frame_count_ + 1));
        if (reanchor_boost_frames_ > 0) {
            reanchor_boost_left_ = reanchor_boost_frames_;
        }
        log_reanchor_event();
    }

    void reset_background_model() {
        backsub_ = cv::cuda::createBackgroundSubtractorMOG2(
            mog2_history_,
            mog2_var_threshold_,
            false
        );
        prev_valid_ = false;
    }

    void init_detector_buffers() {
        int type = input_is_gray_ ? CV_8UC1 : CV_8UC3;
        if (detect_scale_ != 1.0f) {
            scaled_h_ = std::max(1, static_cast<int>(std::round(cropped_h_ * detect_scale_)));
            scaled_w_ = std::max(1, static_cast<int>(std::round(cropped_w_ * detect_scale_)));
        } else {
            scaled_h_ = cropped_h_;
            scaled_w_ = cropped_w_;
        }
        zeros_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        zeros_gpu_.setTo(cv::Scalar(0));

        frame_small_gpu_.create(scaled_h_, scaled_w_, type);
        blur_small_gpu_.create(scaled_h_, scaled_w_, type);
        gray_small_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        diff_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        fg_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        prev_gray_gpu_.create(scaled_h_, scaled_w_, CV_8UC1);
        mask_cpu_.create(scaled_h_, scaled_w_, CV_8UC1);
    }

    void detect_from_registered(const cv::cuda::GpuMat &reg_gpu) {
        // Detection pipeline on the registered GPU frame.
        const auto start = now();
        upload_and_scale(reg_gpu);
        gaussian_blur();
        const auto t_blur = now();
        if (timing_enabled_) {
            timing_["blur"] += elapsed_seconds(start, t_blur);
        }

        calc_diff();
        const auto t_diff = now();
        if (timing_enabled_) {
            timing_["gray"] += elapsed_seconds(t_blur, t_diff);
            timing_["diff"] += elapsed_seconds(t_blur, t_diff);
        }

        float lr = learning_rate_;
        if (reanchor_boost_left_ > 0) {
            lr = reanchor_boost_lr_;
        }
        backsub_->apply(blur_small_gpu_, fg_gpu_, lr);
        const auto t_fg = now();
        if (timing_enabled_) {
            timing_["bgsub"] += elapsed_seconds(t_diff, t_fg);
        }

        fg_gpu_.copyTo(mask_gpu_);
        if (reanchor_boost_left_ > 0) {
            reanchor_boost_left_ -= 1;
        }

        // contours timing is tracked in finalize_detections (CPU)
    }

    void upload_and_scale(const cv::cuda::GpuMat &frame_gpu) {
        // Downscale registered frame for detector if requested.
        if (detect_scale_ == 1.0f) {
            frame_small_gpu_ = frame_gpu;
            return;
        }
        cv::cuda::resize(
            frame_gpu,
            frame_small_gpu_,
            cv::Size(scaled_w_, scaled_h_),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    void gaussian_blur() {
        // Apply prebuilt Gaussian filter on the scaled frame.
        gaussian_filter_->apply(frame_small_gpu_, blur_small_gpu_);
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
        if (prev_valid_) {
            cv::cuda::absdiff(prev_gray_gpu_, gray_small_gpu_, diff_gpu_);
            cv::cuda::threshold(diff_gpu_, diff_gpu_, 25.0, 255.0, cv::THRESH_BINARY);
        } else {
            prev_valid_ = true;
            gray_small_gpu_.copyTo(prev_gray_gpu_);
            diff_gpu_ = zeros_gpu_;
            return;
        }
        gray_small_gpu_.copyTo(prev_gray_gpu_);
    }

    py::array finalize_detections() {
        // CPU contour extraction from the downloaded low-res mask.
        if (timing_enabled_) {
            const auto t_dl_start = now();
            mask_gpu_.download(mask_cpu_);
            timing_["download"] += elapsed_seconds(t_dl_start, now());
        } else {
            mask_gpu_.download(mask_cpu_);
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
        const auto t_contours_start = now();
        cv::findContours(mask_cpu_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (timing_enabled_) {
            const auto t_contours_end = now();
            timing_["contours"] += elapsed_seconds(t_contours_start, t_contours_end);
        }

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
            cv::resize(cc_mask_cpu_, cc_mask_cpu_, cv::Size(cropped_w_, cropped_h_), 0.0, 0.0, cv::INTER_NEAREST);
        }

        const auto t_cpu_start = now();
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
            bb_mask_cpu_ = cv::Mat::zeros(cv::Size(cropped_w_, cropped_h_), CV_8UC3);
        }
        for (const auto &bbox : detections) {
            cv::rectangle(bb_mask_cpu_, bbox, cv::Scalar(0, 0, 255), 1);
        }

        py::array detections_arr(
            py::dtype::of<int32_t>(),
            {static_cast<py::ssize_t>(detections.size()), static_cast<py::ssize_t>(4)}
        );
        auto buf = detections_arr.mutable_unchecked<int32_t, 2>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(detections.size()); ++i) {
            const cv::Rect &r = detections[i];
            buf(i, 0) = r.x;
            buf(i, 1) = r.y;
            buf(i, 2) = r.width;
            buf(i, 3) = r.height;
        }
        if (timing_enabled_) {
            const auto t_cpu_end = now();
            timing_["bbox_creation"] += elapsed_seconds(t_cpu_start, t_cpu_end);
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
    float downscale_factor_ = 1.0f;
    float detect_scale_ = 1.0f;
    float scale_inv_ = 1.0f;
    float learning_rate_ = 0.05f;
    int reference_window_frames_ = 1;
    int mog2_history_ = 400;
    float mog2_var_threshold_ = 20.0f;
    float phase_response_threshold_ = 0.1f;
    int reanchor_boost_frames_ = 0;
    int reanchor_boost_left_ = 0;
    float max_shift_ = 0.0f;
    float reanchor_boost_lr_ = 0.0f;
    bool phase_use_cached_fft_ = true;
    std::ofstream reanchor_log_;
    float last_response_ = 0.0f;
    bool last_phase_ok_ = false;
    std::string last_reanchor_reason_ = "none";

    bool timing_enabled_ = true;
    std::map<std::string, double> timing_;

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    int cropped_h_ = 0;
    int cropped_w_ = 0;
    int scaled_h_ = 0;
    int scaled_w_ = 0;

    std::size_t frame_count_ = 0;

    cv::cuda::GpuMat frame_gpu_;
    cv::cuda::GpuMat gray_gpu_;
    cv::cuda::GpuMat gray_ref_gpu_;
    cv::cuda::GpuMat ref_fft_gpu_;
    cv::cuda::GpuMat reg_gpu_;

    cv::Ptr<cv::cuda::BackgroundSubtractorMOG2> backsub_;
    cv::Ptr<cv::cuda::Filter> gaussian_filter_;
    cv::Mat morph_kernel_;
    int close_iterations_ = 1;
    int open_iterations_ = 1;
    int dilate_iterations_ = 2;
    double min_contour_area_ = 50.0;
    double max_contour_area_ = 1500.0;

    cv::cuda::GpuMat frame_small_gpu_;
    cv::cuda::GpuMat blur_small_gpu_;
    cv::cuda::GpuMat gray_small_gpu_;
    cv::cuda::GpuMat diff_gpu_;
    cv::cuda::GpuMat fg_gpu_;
    cv::cuda::GpuMat mask_gpu_;
    cv::cuda::GpuMat zeros_gpu_;
    cv::cuda::GpuMat prev_gray_gpu_;
    bool prev_valid_ = false;

    cv::Mat mask_cpu_;
    cv::Mat cc_mask_cpu_;
    cv::Mat bb_mask_cpu_;
    cv::Mat registered_cpu_;
    std::vector<int> reference_events_;
    float last_dx_ = 0.0f;
    float last_dy_ = 0.0f;

};

struct KalmanBBox {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

class KalmanIoUTrackerCpp {
public:
    struct Entity {
        int id = 0;
        KalmanBBox bbox;
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
        std::vector<KalmanBBox> obs = parse_observations(observations);
        const int num_obs = static_cast<int>(obs.size());
        std::vector<bool> used_mask(num_obs, false);
        std::map<int, Entity> updated_tracks;

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
    static void init_kalman_state(cv::KalmanFilter &kf, const KalmanBBox &bbox) {
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

    void apply_match(Entity &entity, const KalmanBBox &bbox) {
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

    static float iou(const KalmanBBox &a, const KalmanBBox &b) {
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
        const std::vector<KalmanBBox> &obs
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

    static std::vector<KalmanBBox> parse_observations(const py::array &observations) {
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
        std::vector<KalmanBBox> out;
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

PYBIND11_MODULE(_register_detect_cpp, m) {
    m.doc() = "GPU register+detect pipeline (C++/pybind11).";

    py::class_<RegisterAndDetectGPUCpp>(m, "RegisterAndDetectGPUCpp")
        .def(py::init<
             const py::array &,
             bool,
             float,
             float,
             float,
             int,
             float,
             float,
             int,
             const std::string &,
             float,
             float,
             bool,
             bool
         >(),
         py::arg("reference"),
         py::arg("input_is_gray"),
         py::arg("downscale_factor") = 1.0f,
         py::arg("detect_scale") = 1.0f,
         py::arg("learning_rate") = 0.05f,
         py::arg("reference_window_frames") = 1,
         py::arg("mog2_var_threshold") = 20.0f,
         py::arg("phase_response_threshold") = 0.1f,
         py::arg("reanchor_boost_frames") = 0,
         py::arg("reanchor_log_path") = "",
         py::arg("reanchor_boost_lr") = 0.0f,
         py::arg("max_shift") = 0.0f,
         py::arg("phase_use_cached_fft") = true,
         py::arg("enable_timing") = true)
        .def_static("is_available", &RegisterAndDetectGPUCpp::is_available)
        .def("register_and_detect", &RegisterAndDetectGPUCpp::register_and_detect)
        .def("timing_summary", &RegisterAndDetectGPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("reset_timing", &RegisterAndDetectGPUCpp::reset_timing)
        .def("get_last_mask", &RegisterAndDetectGPUCpp::get_last_mask)
        .def("get_last_cc_mask", &RegisterAndDetectGPUCpp::get_last_cc_mask)
        .def("get_last_bb_mask", &RegisterAndDetectGPUCpp::get_last_bb_mask)
        .def("get_reference_events", &RegisterAndDetectGPUCpp::get_reference_events)
        .def("get_last_shift", &RegisterAndDetectGPUCpp::get_last_shift)
        .def("get_last_registration_debug", &RegisterAndDetectGPUCpp::get_last_registration_debug);

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
