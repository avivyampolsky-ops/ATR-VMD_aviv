#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class DirectTranslationGPUCpp {
public:
    DirectTranslationGPUCpp(
        const py::array &reference,
        float downscale_factor = 1.0f,
        float phase_response_threshold = 0.1f,
        bool phase_use_cached_fft = true,
        int reference_window_frames = 1,
        bool enable_timing = true
    )
        : downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          phase_response_threshold_(phase_response_threshold),
          phase_use_cached_fft_(phase_use_cached_fft),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          timing_enabled_(enable_timing) {
        set_reference(reference);
    }

    static bool is_available() {
        try {
            return cv::cuda::getCudaEnabledDeviceCount() > 0;
        } catch (const cv::Exception &) {
            return false;
        }
    }

    void set_reference(const py::array &reference) {
        cv::Mat ref_mat = ensure_mat(reference);
        set_geometry(ref_mat);
        set_reference_mat(ref_mat);
    }

    py::tuple get_shape() const {
        return py::make_tuple(h_ - margin_ * 2, w_ - margin_ * 2);
    }

    void reset_timing() {
        timing_calls_ = 0;
        timing_total_seconds_ = 0.0;
    }

    py::dict timing_summary(bool reset = false) {
        py::dict out;
        if (!timing_enabled_) {
            out["enabled"] = false;
            return out;
        }
        out["enabled"] = true;
        out["calls"] = timing_calls_;
        if (timing_calls_ > 0) {
            out["registration"] = (timing_total_seconds_ / timing_calls_) * 1000.0;
        } else {
            out["registration"] = 0.0;
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    py::array get_reference_events() const {
        py::array_t<int32_t> out(reference_events_.size());
        auto buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(reference_events_.size()); ++i) {
            buf(i) = reference_events_[static_cast<size_t>(i)];
        }
        return out;
    }

    py::tuple register_frame(const py::array &frame) {
        const auto start = now();
        cv::Mat frame_mat = ensure_mat(frame);
        if (input_is_gray_ && frame_mat.channels() != 1) {
            throw std::runtime_error("Expected grayscale frame for GPU translation.");
        }
        if (!input_is_gray_ && frame_mat.channels() != 3) {
            throw std::runtime_error("Expected BGR frame for GPU translation.");
        }

        frame_count_ += 1;

        cv::Mat reg_cpu;
        float dx = 0.0f;
        float dy = 0.0f;
        bool ok = register_with_fallback(frame_mat, reg_cpu, dx, dy);
        (void)ok;

        if (timing_enabled_) {
            timing_calls_ += 1;
            timing_total_seconds_ += elapsed_seconds(start);
        }

        return py::make_tuple(mat_to_array(reg_cpu), dx, dy);
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
        input_is_gray_ = reference.channels() == 1;
    }

    void set_reference_mat(const cv::Mat &reference) {
        gray_ref_gpu_ = prepare_gray(reference);
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(
                gray_ref_gpu_,
                ref_fft_gpu_,
                gray_ref_gpu_.size()
            );
        }
    }

    void set_reference_gray(const cv::cuda::GpuMat &gray_f) {
        gray_ref_gpu_ = gray_f.clone();
        ref_fft_gpu_.release();
        if (phase_use_cached_fft_) {
            cv::cuda::dft(
                gray_ref_gpu_,
                ref_fft_gpu_,
                gray_ref_gpu_.size()
            );
        }
    }

    cv::cuda::GpuMat prepare_gray(const cv::Mat &frame_cpu) {
        // Upload and convert to float32 grayscale (optionally downscaled) for phase correlation.
        frame_gpu_.upload(frame_cpu);
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
                cv::cuda::dft(
                    gray_ref_gpu_,
                    ref_fft_gpu_,
                    gray_ref_gpu_.size()
                );
            }
            ref_fft = ref_fft_gpu_;
        } else {
            cv::cuda::dft(
                gray_ref_gpu_,
                ref_fft,
                gray_ref_gpu_.size()
            );
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
        corr.download(corr_cpu);
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
        return true;
    }

    cv::cuda::GpuMat warp_frame_gpu(float dx, float dy) {
        // Apply translation warp to the cached GPU frame.
        cv::Mat warp = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, dx, 0.0f, 1.0f, dy);
        cv::cuda::GpuMat reg_gpu;
        cv::cuda::warpAffine(
            frame_gpu_,
            reg_gpu,
            warp,
            cv::Size(w_, h_),
            cv::INTER_LINEAR
        );
        if (margin_ == 0) {
            return reg_gpu;
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return reg_gpu(roi);
    }

    bool register_with_fallback(
        const cv::Mat &frame_cpu,
        cv::Mat &reg_cpu,
        float &dx,
        float &dy
    ) {
        float response = 0.0f;
        cv::cuda::GpuMat gray_f = prepare_gray(frame_cpu);
        // phase_correlate_gpu sets dx, dy, response; it returns false if reference is not set.
        // If phase correlation fails or response is low, reanchor to prev_frame and retry once.
        if (!phase_correlate_gpu(gray_f, dx, dy, response) ||
            response < phase_response_threshold_) {
            reg_cpu = crop_cpu(frame_cpu);
            dx = 0.0f;
            dy = 0.0f;
            if (should_update_reference()) {
                set_reference_gray(gray_f);
                record_reference_event("window_fallback");
            }
            return false;
        }

        // If it worked on the first try, apply the computed shift and download to CPU.
        cv::cuda::GpuMat reg_gpu = warp_frame_gpu(dx, dy);
        reg_gpu.download(reg_cpu);
        if (should_update_reference()) {
            set_reference_gray(gray_f);
            record_reference_event("window");
        }
        return true;
    }

    cv::Mat crop_cpu(const cv::Mat &frame_cpu) const {
        // CPU crop to remove borders (same margin as GPU path).
        if (margin_ == 0) {
            return frame_cpu.clone();
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return frame_cpu(roi).clone();
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void record_reference_event(const std::string &reason) {
        (void)reason;
        reference_events_.push_back(static_cast<int>(frame_count_ + 1));
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    bool input_is_gray_ = true;

    float downscale_factor_ = 1.0f;
    float phase_response_threshold_ = 0.1f;
    bool phase_use_cached_fft_ = true;
    int reference_window_frames_ = 1;

    bool timing_enabled_ = true;
    std::size_t timing_calls_ = 0;
    double timing_total_seconds_ = 0.0;

    std::size_t frame_count_ = 0;
    std::vector<int> reference_events_;

    cv::cuda::GpuMat frame_gpu_;
    cv::cuda::GpuMat gray_gpu_;
    cv::cuda::GpuMat gray_ref_gpu_;
    cv::cuda::GpuMat ref_fft_gpu_;
};

class DirectTranslationCPUCpp {
public:
    DirectTranslationCPUCpp(
        const py::array &reference,
        float downscale_factor = 1.0f,
        float phase_response_threshold = 0.1f,
        bool phase_use_cached_fft = true,
        int reference_window_frames = 1,
        bool enable_timing = true
    )
        : downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          phase_response_threshold_(phase_response_threshold),
          phase_use_cached_fft_(phase_use_cached_fft),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          timing_enabled_(enable_timing) {
        set_reference(reference);
    }

    void set_reference(const py::array &reference) {
        cv::Mat ref_mat = ensure_mat(reference);
        set_geometry(ref_mat);
        set_reference_mat(ref_mat);
    }

    py::tuple get_shape() const {
        return py::make_tuple(h_ - margin_ * 2, w_ - margin_ * 2);
    }

    void reset_timing() {
        timing_calls_ = 0;
        timing_total_seconds_ = 0.0;
    }

    py::dict timing_summary(bool reset = false) {
        py::dict out;
        if (!timing_enabled_) {
            out["enabled"] = false;
            return out;
        }
        out["enabled"] = true;
        out["calls"] = timing_calls_;
        if (timing_calls_ > 0) {
            out["registration"] = (timing_total_seconds_ / timing_calls_) * 1000.0;
        } else {
            out["registration"] = 0.0;
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    py::array get_reference_events() const {
        py::array_t<int32_t> out(reference_events_.size());
        auto buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(reference_events_.size()); ++i) {
            buf(i) = reference_events_[static_cast<size_t>(i)];
        }
        return out;
    }

    py::tuple register_frame(const py::array &frame) {
        const auto start = now();
        cv::Mat frame_mat = ensure_mat(frame);
        if (input_is_gray_ && frame_mat.channels() != 1) {
            throw std::runtime_error("Expected grayscale frame for CPU translation.");
        }
        if (!input_is_gray_ && frame_mat.channels() != 3) {
            throw std::runtime_error("Expected BGR frame for CPU translation.");
        }

        frame_count_ += 1;

        cv::Mat reg_cpu;
        float dx = 0.0f;
        float dy = 0.0f;
        bool ok = register_with_fallback(frame_mat, reg_cpu, dx, dy);
        (void)ok;

        if (timing_enabled_) {
            timing_calls_ += 1;
            timing_total_seconds_ += elapsed_seconds(start);
        }

        return py::make_tuple(mat_to_array(reg_cpu), dx, dy);
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
        input_is_gray_ = reference.channels() == 1;
    }

    cv::Mat prepare_gray(const cv::Mat &frame_cpu) {
        // CPU grayscale + optional downscale to float32 for phase correlation.
        cv::Mat gray;
        if (input_is_gray_) {
            gray = frame_cpu;
        } else {
            cv::cvtColor(frame_cpu, gray, cv::COLOR_BGR2GRAY);
        }
        cv::Mat scaled;
        if (downscale_factor_ != 1.0f) {
            cv::resize(
                gray,
                scaled,
                cv::Size(),
                downscale_factor_,
                downscale_factor_,
                cv::INTER_AREA
            );
        } else {
            scaled = gray;
        }
        cv::Mat gray_f;
        scaled.convertTo(gray_f, CV_32F);
        return gray_f;
    }

    void set_reference_mat(const cv::Mat &reference) {
        gray_ref_cpu_ = prepare_gray(reference);
        cv::createHanningWindow(
            hann_window_,
            cv::Size(gray_ref_cpu_.cols, gray_ref_cpu_.rows),
            CV_32F
        );
        if (phase_use_cached_fft_) {
            cv::Mat windowed = gray_ref_cpu_.mul(hann_window_);
            cv::dft(windowed, ref_fft_cpu_, cv::DFT_COMPLEX_OUTPUT);
        } else {
            ref_fft_cpu_.release();
        }
    }

    void set_reference_gray(const cv::Mat &gray_f) {
        gray_ref_cpu_ = gray_f.clone();
        if (hann_window_.empty() || hann_window_.size() != gray_ref_cpu_.size()) {
            cv::createHanningWindow(
                hann_window_,
                cv::Size(gray_ref_cpu_.cols, gray_ref_cpu_.rows),
                CV_32F
            );
        }
        if (phase_use_cached_fft_) {
            cv::Mat windowed = gray_ref_cpu_.mul(hann_window_);
            cv::dft(windowed, ref_fft_cpu_, cv::DFT_COMPLEX_OUTPUT);
        } else {
            ref_fft_cpu_.release();
        }
    }

    bool phase_correlate_cpu(
        const cv::Mat &gray_cpu,
        float &dx,
        float &dy,
        float &response
    ) {
        // CPU phaseCorrelate wrapper with response extraction.
        if (gray_ref_cpu_.empty()) {
            return false;
        }
        double response_d = 0.0;
        cv::Point2d shift = cv::phaseCorrelate(gray_ref_cpu_, gray_cpu, hann_window_, &response_d);
        response = static_cast<float>(response_d);
        dx = static_cast<float>(shift.x / downscale_factor_);
        dy = static_cast<float>(shift.y / downscale_factor_);
        return true;
    }

    bool register_with_fallback(
        const cv::Mat &frame_cpu,
        cv::Mat &reg_cpu,
        float &dx,
        float &dy
    ) {
        float response = 0.0f;
        cv::Mat gray_f = prepare_gray(frame_cpu);
        if (!phase_correlate_cpu(gray_f, dx, dy, response) ||
            response < phase_response_threshold_) {
            reg_cpu = crop_cpu(frame_cpu);
            dx = 0.0f;
            dy = 0.0f;
            if (should_update_reference()) {
                set_reference_gray(gray_f);
                record_reference_event("window_fallback");
            }
            return false;
        }

        reg_cpu = warp_cpu(frame_cpu, dx, dy);
        if (should_update_reference()) {
            set_reference_gray(gray_f);
            record_reference_event("window");
        }
        return true;
    }

    cv::Mat warp_cpu(const cv::Mat &frame_cpu, float dx, float dy) {
        // Apply translation warp on CPU and crop margins.
        cv::Mat warp = (cv::Mat_<float>(2, 3) << 1.0f, 0.0f, dx, 0.0f, 1.0f, dy);
        cv::Mat reg_cpu;
        cv::warpAffine(frame_cpu, reg_cpu, warp, cv::Size(w_, h_), cv::INTER_LINEAR);
        if (margin_ == 0) {
            return reg_cpu;
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return reg_cpu(roi).clone();
    }

    cv::Mat crop_cpu(const cv::Mat &frame_cpu) const {
        if (margin_ == 0) {
            return frame_cpu.clone();
        }
        cv::Rect roi(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2);
        return frame_cpu(roi).clone();
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void record_reference_event(const std::string &reason) {
        (void)reason;
        reference_events_.push_back(static_cast<int>(frame_count_ + 1));
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    bool input_is_gray_ = true;

    float downscale_factor_ = 1.0f;
    float phase_response_threshold_ = 0.1f;
    bool phase_use_cached_fft_ = true;
    int reference_window_frames_ = 1;

    bool timing_enabled_ = true;
    std::size_t timing_calls_ = 0;
    double timing_total_seconds_ = 0.0;

    std::size_t frame_count_ = 0;
    std::vector<int> reference_events_;

    cv::Mat gray_ref_cpu_;
    cv::Mat hann_window_;
    cv::Mat ref_fft_cpu_;
};

class HomographyTranslationCPUCpp {
public:
    HomographyTranslationCPUCpp(
        const py::array &reference,
        const std::string &feature_type = "ORB",
        const std::string &matcher_type = "BF",
        float downscale_factor = 1.0f,
        float knn_ratio = 0.75f,
        float ransac_reproj_threshold = 5.0f,
        int min_inliers = 0,
        int reference_window_frames = 1,
        bool enable_timing = true
    )
        : downscale_factor_(downscale_factor > 0.0f ? downscale_factor : 1.0f),
          knn_ratio_(knn_ratio),
          ransac_reproj_threshold_(ransac_reproj_threshold),
          min_inliers_(min_inliers > 0 ? min_inliers : 0),
          reference_window_frames_(reference_window_frames > 0 ? reference_window_frames : 1),
          timing_enabled_(enable_timing),
          feature_type_(feature_type),
          matcher_type_(matcher_type) {
        // Default feature/matcher are ORB + BF when not specified.
        set_reference(reference, false, "init");
    }

    static bool is_available() {
        return true;
    }

    void set_reference(const py::array &reference, bool record_event = true,
                       const std::string &reason = "manual") {
        cv::Mat ref_mat = ensure_mat(reference);
        set_geometry(ref_mat);
        set_reference_mat(ref_mat);
        if (record_event) {
            reference_events_.push_back(static_cast<int>(frame_count_));
            last_reference_reason_ = reason;
        }
    }

    void set_reference_mat(const cv::Mat &reference, bool record_event,
                           const std::string &reason) {
        set_geometry(reference);
        set_reference_mat(reference);
        if (record_event) {
            reference_events_.push_back(static_cast<int>(frame_count_));
            last_reference_reason_ = reason;
        }
    }

    py::tuple get_shape() const {
        return py::make_tuple(h_ - margin_ * 2, w_ - margin_ * 2);
    }

    void reset_timing() {
        if (!timing_enabled_) {
            return;
        }
        for (auto &kv : timing_) {
            kv.second = 0.0;
        }
    }

    py::dict timing_summary(bool reset = false) {
        py::dict out;
        if (!timing_enabled_) {
            out["enabled"] = false;
            return out;
        }
        out["enabled"] = true;
        const double calls = timing_["calls"];
        out["calls"] = static_cast<int>(calls);
        if (calls > 0.0) {
            out["registration"] = (timing_["registration"] / calls) * 1000.0;
            out["gray_ms_avg"] = (timing_["gray"] / calls) * 1000.0;
            out["keypoints_ms_avg"] = (timing_["keypoints"] / calls) * 1000.0;
            out["match_ms_avg"] = (timing_["match"] / calls) * 1000.0;
            out["homography_ms_avg"] = (timing_["homography"] / calls) * 1000.0;
            out["warp_ms_avg"] = (timing_["warp"] / calls) * 1000.0;
            out["re_registration_calls"] = static_cast<int>(re_registration_calls_);
        }
        if (reset) {
            reset_timing();
        }
        return out;
    }

    py::array get_reference_events() const {
        py::array_t<int32_t> out(reference_events_.size());
        auto buf = out.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(reference_events_.size()); ++i) {
            buf(i) = reference_events_[static_cast<size_t>(i)];
        }
        return out;
    }

    py::dict get_last_homography_debug() const {
        py::dict out;
        out["fail_reason"] = last_fail_reason_;
        out["match_count"] = last_match_count_;
        out["inlier_count"] = last_inlier_count_;
        out["overlap_ratio"] = last_overlap_;
        out["reference_reason"] = last_reference_reason_;
        return out;
    }

    py::array register_frame(const py::array &frame) {
        cv::Mat frame_mat = ensure_mat(frame);
        frame_count_ += 1;
        cv::Mat out = register_with_fallback(frame_mat);
        return mat_to_array(out);
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
        input_is_gray_ = reference.channels() == 1;
        if (downscale_factor_ != 1.0f) {
            float s = downscale_factor_;
            scale_matrix_ = (cv::Mat_<float>(3, 3) <<
                s, 0.0f, 0.0f,
                0.0f, s, 0.0f,
                0.0f, 0.0f, 1.0f);
            scale_matrix_inv_ = (cv::Mat_<float>(3, 3) <<
                1.0f / s, 0.0f, 0.0f,
                0.0f, 1.0f / s, 0.0f,
                0.0f, 0.0f, 1.0f);
        } else {
            scale_matrix_ = cv::Mat::eye(3, 3, CV_32F);
            scale_matrix_inv_ = cv::Mat::eye(3, 3, CV_32F);
        }
        frame_corners_ = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(static_cast<float>(w_), 0.0f),
            cv::Point2f(static_cast<float>(w_), static_cast<float>(h_)),
            cv::Point2f(0.0f, static_cast<float>(h_))
        };
        frame_area_ = static_cast<float>(w_ * h_);
    }

    void set_reference_mat(const cv::Mat &reference) {
        gray_ref_ = prepare_gray(reference);
        build_feature_extractor();
        extract_features(gray_ref_, kp_ref_, descriptors_ref_);
    }

    void set_reference_from_features(
        const cv::Mat &gray,
        const std::vector<cv::KeyPoint> &kp,
        const cv::Mat &descriptors
    ) {
        gray_ref_ = gray;
        kp_ref_ = kp;
        descriptors_ref_ = descriptors.clone();
    }

    cv::Mat prepare_gray(const cv::Mat &frame) const {
        cv::Mat gray;
        if (input_is_gray_) {
            gray = frame;
        } else {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        if (downscale_factor_ != 1.0f) {
            cv::Mat scaled;
            cv::resize(gray, scaled, cv::Size(), downscale_factor_, downscale_factor_, cv::INTER_AREA);
            return scaled;
        }
        return gray;
    }

    void build_feature_extractor() {
        // Map feature_type_ string to a concrete extractor (ORB/FAST+BRIEF/AKAZE/BRISK/SIFT).
        std::string type = feature_type_;
        if (type.empty()) {
            type = "ORB";
        }
        std::string upper;
        upper.reserve(type.size());
        for (char c : type) {
            upper.push_back(static_cast<char>(std::toupper(c)));
        }
        feature_type_ = upper;

        if (feature_type_ == "SIFT") {
            feature_extractor_ = cv::SIFT::create(500);
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_L2;
        } else if (feature_type_ == "AKAZE") {
            feature_extractor_ = cv::AKAZE::create();
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else if (feature_type_ == "BRISK") {
            feature_extractor_ = cv::BRISK::create();
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else if (feature_type_ == "FAST_BRIEF") {
            fast_detector_ = cv::FastFeatureDetector::create(20, true);
            descriptor_extractor_ = cv::xfeatures2d::BriefDescriptorExtractor::create(32);
            feature_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        } else {
            feature_extractor_ = cv::ORB::create(500);
            descriptor_extractor_.release();
            norm_type_ = cv::NORM_HAMMING;
        }

        std::string matcher_type = matcher_type_;
        if (matcher_type.empty()) {
            matcher_type = "BF";
        }
        std::string matcher_upper;
        matcher_upper.reserve(matcher_type.size());
        for (char c : matcher_type) {
            matcher_upper.push_back(static_cast<char>(std::toupper(c)));
        }
        matcher_type_ = matcher_upper;
    }

    void extract_features(const cv::Mat &gray,
                          std::vector<cv::KeyPoint> &kps,
                          cv::Mat &descriptors) {
        if (feature_type_ == "FAST_BRIEF") {
            if (!fast_detector_ || !descriptor_extractor_) {
                throw std::runtime_error("FAST_BRIEF requires xfeatures2d.");
            }
            fast_detector_->detect(gray, kps);
            if (kps.size() > 500) {
                std::nth_element(kps.begin(), kps.begin() + 500, kps.end(),
                                 [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                                     return a.response > b.response;
                                 });
                kps.resize(500);
            }
            descriptor_extractor_->compute(gray, kps, descriptors);
        } else if (feature_extractor_) {
            feature_extractor_->detectAndCompute(gray, cv::noArray(), kps, descriptors);
        }
    }

    std::vector<cv::DMatch> match_descriptors(const cv::Mat &descriptors_frame) {
        std::vector<cv::DMatch> matches;
        if (descriptors_ref_.empty() || descriptors_frame.empty()) {
            return matches;
        }
        // KNN/FLANN use ratio test; BF returns best matches sorted by distance.
        if (matcher_type_ == "KNN") {
            cv::BFMatcher matcher(norm_type_);
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(descriptors_frame, descriptors_ref_, knn, 2);
            for (const auto &pair : knn) {
                if (pair.size() < 2) {
                    continue;
                }
                if (pair[0].distance < knn_ratio_ * pair[1].distance) {
                    matches.push_back(pair[0]);
                }
            }
        } else if (matcher_type_ == "FLANN") {
            cv::FlannBasedMatcher matcher;
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(descriptors_frame, descriptors_ref_, knn, 2);
            for (const auto &pair : knn) {
                if (pair.size() < 2) {
                    continue;
                }
                if (pair[0].distance < knn_ratio_ * pair[1].distance) {
                    matches.push_back(pair[0]);
                }
            }
        } else {
            cv::BFMatcher matcher(norm_type_, true);
            matcher.match(descriptors_frame, descriptors_ref_, matches);
            std::sort(matches.begin(), matches.end(),
                      [](const cv::DMatch &a, const cv::DMatch &b) {
                          return a.distance < b.distance;
                      });
        }
        return matches;
    }

    cv::Mat calc_homography(const std::vector<cv::KeyPoint> &kp_frame,
                             const std::vector<cv::DMatch> &matches) {
        if (matches.size() < 4) {
            last_inlier_count_ = 0;
            return cv::Mat();
        }
        std::vector<cv::Point2f> pts1;
        std::vector<cv::Point2f> pts2;
        pts1.reserve(matches.size());
        pts2.reserve(matches.size());
        for (const auto &m : matches) {
            pts1.push_back(kp_frame[m.queryIdx].pt);
            pts2.push_back(kp_ref_[m.trainIdx].pt);
        }
        cv::Mat mask;
        cv::Mat H = cv::findHomography(
            pts1,
            pts2,
            cv::RANSAC,
            ransac_reproj_threshold_,
            mask
        );
        if (!mask.empty()) {
            last_inlier_count_ = static_cast<int>(cv::countNonZero(mask));
        } else {
            last_inlier_count_ = 0;
        }
        return H;
    }

    float overlap_ratio(const cv::Mat &H) const {
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(frame_corners_, dst, H);
        std::vector<cv::Point2f> hull;
        cv::convexHull(dst, hull);
        std::vector<cv::Point2f> ref_hull;
        cv::convexHull(frame_corners_, ref_hull);
        double inter_area = cv::intersectConvexConvex(hull, ref_hull, hull);
        if (inter_area <= 0.0) {
            return 0.0f;
        }
        return static_cast<float>(inter_area / frame_area_);
    }

    cv::Mat scale_homography(const cv::Mat &H) const {
        if (downscale_factor_ == 1.0f) {
            return H;
        }
        cv::Mat H32;
        if (H.type() != CV_32F) {
            H.convertTo(H32, CV_32F);
        } else {
            H32 = H;
        }
        return scale_matrix_inv_ * H32 * scale_matrix_;
    }

    cv::Mat register_with_fallback(const cv::Mat &frame_mat) {
        auto start = now();
        cv::Mat gray = prepare_gray(frame_mat);
        auto t_gray = now();
        std::vector<cv::KeyPoint> kp_frame;
        cv::Mat descriptors_frame;
        auto keypoint_start = t_gray;
        extract_features(gray, kp_frame, descriptors_frame);
        if (timing_enabled_) {
            timing_["gray"] += elapsed_seconds(start, t_gray);
            timing_["keypoints"] += elapsed_seconds(keypoint_start, now());
        }

        cv::Mat reg = register_frame_internal(frame_mat, kp_frame, descriptors_frame);
        if (!reg.empty()) {
            if (descriptors_frame.rows >= 4) {
                if (should_update_reference()) {
                    set_reference_from_features(gray, kp_frame, descriptors_frame);
                    record_reference_event("window");
                }
            }
            record_timing(start, now());
            return reg;
        }

        record_timing(start, now());
        if (descriptors_frame.rows >= 4) {
            if (should_update_reference()) {
                set_reference_from_features(gray, kp_frame, descriptors_frame);
                record_reference_event("window_fallback");
            }
        }
        cv::Mat cropped = crop_frame(frame_mat);
        return cropped;
    }

    cv::Mat register_frame_internal(
        const cv::Mat &frame_mat,
        const std::vector<cv::KeyPoint> &kp_frame,
        const cv::Mat &descriptors_frame
    ) {
        if (descriptors_frame.empty() || kp_frame.size() < 4) {
            last_fail_reason_ = "descriptors";
            return cv::Mat();
        }
        auto match_start = now();
        std::vector<cv::DMatch> matches = match_descriptors(descriptors_frame);
        last_match_count_ = static_cast<int>(matches.size());
        if (timing_enabled_) {
            timing_["match"] += elapsed_seconds(match_start, now());
        }
        if (matches.empty()) {
            last_fail_reason_ = "match";
            return cv::Mat();
        }

        auto h_start = now();
        cv::Mat H = calc_homography(kp_frame, matches);
        if (!H.empty()) {
            H = scale_homography(H);
        }
        if (timing_enabled_) {
            timing_["homography"] += elapsed_seconds(h_start, now());
        }
        if (H.empty()) {
            last_fail_reason_ = "homography";
            return cv::Mat();
        }
        if (min_inliers_ && last_inlier_count_ < min_inliers_) {
            last_fail_reason_ = "inliers";
            return cv::Mat();
        }
        float overlap = overlap_ratio(H);
        last_overlap_ = overlap;

        auto warp_start = now();
        cv::Mat reg;
        cv::warpPerspective(frame_mat, reg, H, cv::Size(w_, h_));
        if (timing_enabled_) {
            timing_["warp"] += elapsed_seconds(warp_start, now());
        }

        last_fail_reason_.clear();
        return crop_frame(reg);
    }

    cv::Mat crop_frame(const cv::Mat &frame_mat) const {
        if (margin_ == 0) {
            return frame_mat.clone();
        }
        return frame_mat(cv::Rect(margin_, margin_, w_ - margin_ * 2, h_ - margin_ * 2)).clone();
    }

    static std::chrono::steady_clock::time_point now() {
        return std::chrono::steady_clock::now();
    }

    static double elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                                  const std::chrono::steady_clock::time_point &end) {
        return std::chrono::duration<double>(end - start).count();
    }

    void record_timing(const std::chrono::steady_clock::time_point &start,
                       const std::chrono::steady_clock::time_point &end) {
        if (!timing_enabled_) {
            return;
        }
        timing_["calls"] += 1.0;
        timing_["registration"] += elapsed_seconds(start, end);
    }

    bool should_update_reference() const {
        return reference_window_frames_ > 0 && (frame_count_ % reference_window_frames_ == 0);
    }

    void record_reference_event(const std::string &reason) {
        reference_events_.push_back(static_cast<int>(frame_count_ + 1));
        last_reference_reason_ = reason;
    }

    bool input_is_gray_ = true;
    float downscale_factor_ = 1.0f;
    float knn_ratio_ = 0.75f;
    float ransac_reproj_threshold_ = 5.0f;
    int min_inliers_ = 0;
    int reference_window_frames_ = 1;
    bool timing_enabled_ = true;

    std::string feature_type_;
    std::string matcher_type_;
    int norm_type_ = cv::NORM_HAMMING;
    cv::Ptr<cv::Feature2D> feature_extractor_;
    cv::Ptr<cv::Feature2D> descriptor_extractor_;
    cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    cv::Mat gray_ref_;
    std::vector<cv::KeyPoint> kp_ref_;
    cv::Mat descriptors_ref_;

    cv::Mat scale_matrix_;
    cv::Mat scale_matrix_inv_;

    int h_ = 0;
    int w_ = 0;
    int margin_ = 0;
    std::vector<cv::Point2f> frame_corners_;
    float frame_area_ = 0.0f;

    std::size_t frame_count_ = 0;

    std::map<std::string, double> timing_{
        {"calls", 0.0},
        {"registration", 0.0},
        {"gray", 0.0},
        {"keypoints", 0.0},
        {"match", 0.0},
        {"homography", 0.0},
        {"warp", 0.0},
    };
    int re_registration_calls_ = 0;

    std::string last_fail_reason_;
    int last_match_count_ = 0;
    int last_inlier_count_ = 0;
    float last_overlap_ = 0.0f;

    std::vector<int> reference_events_;
    std::string last_reference_reason_;
};

class HomographyTranslationGPUCpp : public HomographyTranslationCPUCpp {
public:
    using HomographyTranslationCPUCpp::HomographyTranslationCPUCpp;

    static bool is_available() {
        try {
            return cv::cuda::getCudaEnabledDeviceCount() > 0;
        } catch (const cv::Exception &) {
            return false;
        }
    }
};

PYBIND11_MODULE(_translation_gpu_cpp, m) {
    m.doc() = "GPU translation via CUDA phase correlation (C++/pybind11).";

    py::class_<DirectTranslationGPUCpp>(m, "DirectTranslationGPUCpp")
        .def(py::init<
             const py::array &,
             float,
             float,
             bool,
             int,
             bool
         >(),
         py::arg("reference"),
         py::arg("downscale_factor") = 1.0f,
         py::arg("phase_response_threshold") = 0.1f,
         py::arg("phase_use_cached_fft") = true,
         py::arg("reference_window_frames") = 1,
         py::arg("enable_timing") = true)
        .def_static("is_available", &DirectTranslationGPUCpp::is_available)
        .def("set_reference", &DirectTranslationGPUCpp::set_reference)
        .def("get_shape", &DirectTranslationGPUCpp::get_shape)
        .def("reset_timing", &DirectTranslationGPUCpp::reset_timing)
        .def("timing_summary", &DirectTranslationGPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("register_frame", &DirectTranslationGPUCpp::register_frame)
        .def("get_reference_events", &DirectTranslationGPUCpp::get_reference_events);

    py::class_<DirectTranslationCPUCpp>(m, "DirectTranslationCPUCpp")
        .def(py::init<
             const py::array &,
             float,
             float,
             bool,
             int,
             bool
        >(),
        py::arg("reference"),
        py::arg("downscale_factor") = 1.0f,
        py::arg("phase_response_threshold") = 0.1f,
        py::arg("phase_use_cached_fft") = true,
        py::arg("reference_window_frames") = 1,
        py::arg("enable_timing") = true)
        .def("set_reference", &DirectTranslationCPUCpp::set_reference)
        .def("get_shape", &DirectTranslationCPUCpp::get_shape)
        .def("reset_timing", &DirectTranslationCPUCpp::reset_timing)
        .def("timing_summary", &DirectTranslationCPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("register_frame", &DirectTranslationCPUCpp::register_frame)
        .def("get_reference_events", &DirectTranslationCPUCpp::get_reference_events);

    py::class_<HomographyTranslationGPUCpp>(m, "HomographyTranslationGPUCpp")
        .def(py::init<
             const py::array &,
             const std::string &,
             const std::string &,
             float,
             float,
             float,
             int,
             int,
             bool
         >(),
         py::arg("reference"),
         py::arg("feature_type") = "ORB",
         py::arg("matcher_type") = "BF",
         py::arg("downscale_factor") = 1.0f,
         py::arg("knn_ratio") = 0.75f,
         py::arg("ransac_reproj_threshold") = 5.0f,
         py::arg("min_inliers") = 0,
         py::arg("reference_window_frames") = 1,
         py::arg("enable_timing") = true)
        .def_static("is_available", &HomographyTranslationGPUCpp::is_available)
        .def("set_reference", &HomographyTranslationGPUCpp::set_reference,
             py::arg("reference"),
             py::arg("record_event") = true,
             py::arg("reason") = "manual")
        .def("get_shape", &HomographyTranslationGPUCpp::get_shape)
        .def("reset_timing", &HomographyTranslationGPUCpp::reset_timing)
        .def("timing_summary", &HomographyTranslationGPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("register_frame", &HomographyTranslationGPUCpp::register_frame)
        .def("get_reference_events", &HomographyTranslationGPUCpp::get_reference_events)
        .def("get_last_homography_debug", &HomographyTranslationGPUCpp::get_last_homography_debug);

    py::class_<HomographyTranslationCPUCpp>(m, "HomographyTranslationCPUCpp")
        .def(py::init<
             const py::array &,
             const std::string &,
             const std::string &,
             float,
             float,
             float,
             int,
             int,
             bool
        >(),
        py::arg("reference"),
        py::arg("feature_type") = "ORB",
        py::arg("matcher_type") = "BF",
        py::arg("downscale_factor") = 1.0f,
        py::arg("knn_ratio") = 0.75f,
        py::arg("ransac_reproj_threshold") = 5.0f,
        py::arg("min_inliers") = 0,
        py::arg("reference_window_frames") = 1,
        py::arg("enable_timing") = true)
        .def("set_reference", &HomographyTranslationCPUCpp::set_reference,
             py::arg("reference"),
             py::arg("record_event") = true,
             py::arg("reason") = "manual")
        .def("get_shape", &HomographyTranslationCPUCpp::get_shape)
        .def("reset_timing", &HomographyTranslationCPUCpp::reset_timing)
        .def("timing_summary", &HomographyTranslationCPUCpp::timing_summary,
             py::arg("reset") = false)
        .def("register_frame", &HomographyTranslationCPUCpp::register_frame)
        .def("get_reference_events", &HomographyTranslationCPUCpp::get_reference_events)
        .def("get_last_homography_debug", &HomographyTranslationCPUCpp::get_last_homography_debug);
}
