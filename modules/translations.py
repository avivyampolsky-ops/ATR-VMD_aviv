import cv2
import numpy as np
from time import perf_counter
from numpy import (
    float32 as np_float32
)

try:
    from ._translation_gpu_cpp import (
        DirectTranslationGPUCpp as _DirectTranslationGPUCpp,
        DirectTranslationCPUCpp as _DirectTranslationCPUCpp,
        HomographyTranslationGPUCpp as _HomographyTranslationGPUCpp,
        HomographyTranslationCPUCpp as _HomographyTranslationCPUCpp,
    )
except Exception:
    _DirectTranslationGPUCpp = None
    _DirectTranslationCPUCpp = None
    _HomographyTranslationGPUCpp = None
    _HomographyTranslationCPUCpp = None

from .features import (
    FeatureExtractorType,
    OrbFeatureExtractor,
    FastBriefFeatureExtractor,
    AkazeFeatureExtractor,
    BriskFeatureExtractor,
    SiftFeatureExtractor,
)
from .matchers import MatcherType, BFMatcher, KNNMatcher, FlannMatcher


class DirectTranslation:
    def __init__(
        self,
        reference,
        method="phase",
        downscale_factor=1.0,
        enable_timing=True,
        phase_response_threshold=0.1,
        phase_use_cached_fft=True,
        reference_window_frames=1,
        use_cuda=False,
        use_cpp=True,
    ):
        if str(method).lower() != "phase":
            raise ValueError(f"Unknown translation method: {method}")
        impl = None
        if use_cpp and use_cuda:
            if _DirectTranslationGPUCpp is None:
                raise RuntimeError("DirectTranslation: C++ GPU module is not available.")
            if not _DirectTranslationGPUCpp.is_available():
                raise RuntimeError("DirectTranslation: C++ GPU module not available on this device.")
        if use_cpp and not use_cuda and _DirectTranslationCPUCpp is None:
            raise RuntimeError("DirectTranslation: C++ CPU module is not available.")
        if use_cpp and use_cuda and _DirectTranslationGPUCpp is not None:
            try:
                impl = DirectTranslationGPUCppWrapper(
                    reference,
                    downscale_factor=downscale_factor,
                    enable_timing=enable_timing,
                    phase_response_threshold=phase_response_threshold,
                    phase_use_cached_fft=phase_use_cached_fft,
                )
                print('Using GPU Cpp:', impl is not None)
            except Exception:
                raise

        if impl is None and use_cpp and not use_cuda and _DirectTranslationCPUCpp is not None:
            try:
                impl = DirectTranslationCPUCppWrapper(
                    reference,
                    downscale_factor=downscale_factor,
                    enable_timing=enable_timing,
                    phase_response_threshold=phase_response_threshold,
                    phase_use_cached_fft=phase_use_cached_fft,
                )
                print('Using CPU Cpp:', impl is not None)
            except Exception:
                raise

        if impl is None:
            impl = DirectTranslationCPU(
                reference,
                method=method,
                downscale_factor=downscale_factor,
                enable_timing=enable_timing,
                phase_response_threshold=phase_response_threshold,
                phase_use_cached_fft=phase_use_cached_fft,
                reference_window_frames=reference_window_frames,
            )
            print('Using Python CPU')
            
        self._impl = impl

    def __getattr__(self, name):
        return getattr(self._impl, name)


class _DirectTranslationBase:
    def __init__(
        self,
        reference,
        method="phase",
        downscale_factor=1.0,
        enable_timing=True,
        phase_response_threshold=0.1,
        phase_use_cached_fft=True,
        reference_window_frames=1,
        defer_reference=False,
    ):
        self.method = str(method).lower()
        if self.method != "phase":
            raise ValueError(f"Unknown translation method: {method}")

        self._downscale_factor = float(downscale_factor) if downscale_factor else 1.0
        # Import here to avoid circular imports and to defer dependency until init time.
        from . import video_reader
        self._input_is_gray = bool(video_reader.input_is_gray())
        # phase correlation
        self._phase_response_threshold = float(phase_response_threshold)
        self._phase_use_cached_fft = bool(phase_use_cached_fft)
        self._warp_flags = cv2.INTER_LINEAR

        self._frame_count = 0
        self._prev_frame = None
        self._last_shift = (0.0, 0.0)
        self._last_fail_reason = None
        self._phase_response_total = 0.0
        self._reference_events = []
        self._last_reference_reason = None

        self._reg_stats = {
            "first_try": {"calls": 0, "total": 0.0},
            "fallback": {"calls": 0, "total": 0.0},
            "re_registration_calls": 0,
        }
        self._reference_window_frames = max(1, int(reference_window_frames))

        self._timing_enabled = bool(enable_timing)
        self._timing = None
        if self._timing_enabled:
            self._timing = {
                "calls": 0,
                "registration": 0.0,
                "gray": 0.0,
                "shift": 0.0,
                "warp": 0.0,
            }
        self._register = self._register_timed if self._timing_enabled else self._register_fast

        self._hann_window = None
        self._ref_fft = None

        if not defer_reference:
            self._set_reference(reference)

    def set_reference(self, reference):
        return self._set_reference(reference)

    def _set_geometry(self, reference):
        self.h, self.w = reference.shape[0:2]
        self.margin = int(min(self.h, self.w) * 0.01)

    def get_shape(self):
        return (self.h - self.margin * 2, self.w - self.margin * 2)

    def _crop_frame(self, reg):
        if self.margin == 0:
            return reg
        if isinstance(reg, cv2.cuda_GpuMat):
            reg = reg.rowRange(self.margin, self.h - self.margin)
            return reg.colRange(self.margin, self.w - self.margin)
        return reg[self.margin:-self.margin, self.margin:-self.margin]

    def reset_timing(self):
        if not self._timing_enabled:
            return
        for key in self._timing:
            self._timing[key] = 0
        for bucket in ("first_try", "fallback"):
            self._reg_stats[bucket]["calls"] = 0
            self._reg_stats[bucket]["total"] = 0.0
        self._reg_stats["re_registration_calls"] = 0
        self._phase_response_total = 0.0

    def timing_summary(self, reset=False):
        if not self._timing_enabled:
            return {"enabled": False}
        calls = self._timing["calls"]
        if calls == 0:
            summary = {"calls": 0}
        else:
            summary = {
                "calls": calls,
                "registration": (self._timing["registration"] / calls) * 1000.0,
                "gray_ms_avg": (self._timing["gray"] / calls) * 1000.0,
                "shift_ms_avg": (self._timing["shift"] / calls) * 1000.0,
                "warp_ms_avg": (self._timing["warp"] / calls) * 1000.0,
                "re_registration_calls": self._reg_stats["re_registration_calls"],
                "registration_first_try_ms_avg": (
                    (self._reg_stats["first_try"]["total"] / self._reg_stats["first_try"]["calls"]) * 1000.0
                    if self._reg_stats["first_try"]["calls"] else 0.0
                ),
                "registration_fallback_ms_avg": (
                    (self._reg_stats["fallback"]["total"] / self._reg_stats["fallback"]["calls"]) * 1000.0
                    if self._reg_stats["fallback"]["calls"] else 0.0
                ),
            }
            summary["phase_response_avg"] = self._phase_response_total / calls
        if reset:
            self.reset_timing()
        return summary

    def _record_total_time(self, elapsed):
        self._timing["calls"] += 1
        self._timing["registration"] += elapsed

    def _register_fast(self, frame, warp, timing=None):
        if warp is None:
            return None

        reg = self._warp(frame, warp)
        reg = self._crop_frame(reg)
        if isinstance(reg, cv2.cuda_GpuMat):
            return reg.download()
        return reg

    def _register_timed(self, frame, warp, timing):
        start = timing["start"]
        t_shift = timing["t_shift"]
        if warp is None:
            self._record_total_time(perf_counter() - start)
            return None

        reg = self._warp(frame, warp)
        reg = self._crop_frame(reg)
        if isinstance(reg, cv2.cuda_GpuMat):
            reg = reg.download()

        t_warp = perf_counter()
        self._timing["warp"] += t_warp - t_shift
        self._record_total_time(t_warp - start)

        return reg

    def _prepare_shift(self, frame):
        if self._timing_enabled:
            start = perf_counter()
            gray = self._prepare_gray(frame)
            t_gray = perf_counter()
            self._timing["gray"] += t_gray - start
            warp = self._estimate_shift_phase(gray)
            t_shift = perf_counter()
            self._timing["shift"] += t_shift - t_gray
            timing = {"start": start, "t_shift": t_shift}
        else:
            gray = self._prepare_gray(frame)
            warp = self._estimate_shift_phase(gray)
            timing = None
        return gray, warp, timing

    def _register_with_fallback(self, frame, prev_frame):
        total_start = perf_counter() if self._timing_enabled else None
        gray, warp, timing = self._prepare_shift(frame)
        out = self._register(frame, warp, timing)
        if out is not None:
            if self._timing_enabled:
                self._reg_stats["first_try"]["calls"] += 1
                self._reg_stats["first_try"]["total"] += perf_counter() - total_start
            if self._should_update_reference():
                self._record_reference_update(gray, reason="window")
            return out

        if self._should_update_reference():
            self._record_reference_update(gray, reason="window_fallback")
        return frame[self.margin:-self.margin, self.margin:-self.margin]

    def register_frame(self, frame):
        self._frame_count += 1
        prev_frame = self._prev_frame
        out = self._register_with_fallback(frame, prev_frame)

        self._prev_frame = frame
        return out

    def _should_update_reference(self):
        return (self._frame_count % self._reference_window_frames) == 0

    def get_reference_events(self):
        return np.array(self._reference_events, dtype=np.int32)

    def _record_reference_update(self, gray, reason):
        self._set_reference_from_gray(gray)
        # Event applies to the next frame (first one registered to the new reference).
        self._reference_events.append(self._frame_count + 1)
        self._last_reference_reason = reason

    def register_frame_with_shift(self, frame):
        reg = self.register_frame(frame)
        dx, dy = self._last_shift
        return reg, dx, dy

    def _estimate_shift_phase(self, gray):
        shift, response = self._estimate_shift(gray)
        if response < self._phase_response_threshold:
            self._last_fail_reason = "phase_response"
            self._last_shift = (0.0, 0.0)
            return None
        self._phase_response_total += response
        dx = shift[0] / self._downscale_factor
        dy = shift[1] / self._downscale_factor
        self._last_shift = (float(dx), float(dy))
        return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    def get_last_shift(self):
        return self._last_shift

    def _prepare_gray(self, frame):
        raise NotImplementedError

    def _set_reference(self, reference):
        raise NotImplementedError

    def _set_reference_from_gray(self, gray):
        raise NotImplementedError

    def _estimate_shift(self, gray):
        raise NotImplementedError

    def _warp(self, frame, warp):
        raise NotImplementedError


class DirectTranslationCPU(_DirectTranslationBase):
    def __init__(self, reference, **kwargs):
        super().__init__(reference, defer_reference=True, **kwargs)
        self._to_gray = self._to_gray_passthrough if self._input_is_gray else self._to_gray_bgr
        self._scale_gray = (
            self._scale_gray_passthrough
            if self._downscale_factor == 1.0
            else self._scale_gray_cpu
        )
        self._set_reference(reference)

    def _to_gray_passthrough(self, frame):
        return frame

    def _to_gray_bgr(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _scale_gray_passthrough(self, gray):
        return gray

    def _scale_gray_cpu(self, gray):
        return cv2.resize(
            gray,
            None,
            fx=self._downscale_factor,
            fy=self._downscale_factor,
            interpolation=cv2.INTER_AREA,
        )

    def _prepare_gray(self, frame):
        gray = self._scale_gray(self._to_gray(frame))
        return gray.astype(np.float32, copy=False)

    def _set_reference(self, reference):
        self._set_geometry(reference)
        self._gray_ref_cpu = self._prepare_gray(reference)
        if self.method == "phase":
            self._hann_window = cv2.createHanningWindow(
                (self._gray_ref_cpu.shape[1], self._gray_ref_cpu.shape[0]),
                cv2.CV_32F,
            )
            if self._phase_use_cached_fft:
                windowed = self._gray_ref_cpu * self._hann_window
                self._ref_fft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)

    def _set_reference_from_gray(self, gray):
        self._gray_ref_cpu = gray
        if self._hann_window is None or self._hann_window.shape != gray.shape:
            self._hann_window = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]),
                cv2.CV_32F,
            )
        if self._phase_use_cached_fft:
            windowed = gray * self._hann_window
            self._ref_fft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)

    def _estimate_shift(self, gray):
        if self._phase_use_cached_fft and self._ref_fft is not None:
            return self._phase_correlate_cached(gray)
        return cv2.phaseCorrelate(self._gray_ref_cpu, gray, self._hann_window)

    def _phase_correlate_cached(self, gray):
        return cv2.phaseCorrelate(self._gray_ref_cpu, gray, self._hann_window)

    def _warp(self, frame, warp):
        return cv2.warpAffine(frame, warp, (self.w, self.h), flags=self._warp_flags)


class DirectTranslationGPUCppWrapper:
    def __init__(
        self,
        reference,
        downscale_factor=1.0,
        enable_timing=True,
        phase_response_threshold=0.1,
        phase_use_cached_fft=True,
        reference_window_frames=1,
    ):
        if _DirectTranslationGPUCpp is None:
            raise RuntimeError("C++ GPU translation module is not available.")
        self._impl = _DirectTranslationGPUCpp(
            reference,
            downscale_factor=downscale_factor,
            phase_response_threshold=phase_response_threshold,
            phase_use_cached_fft=phase_use_cached_fft,
            reference_window_frames=reference_window_frames,
            enable_timing=enable_timing,
        )

    def register_frame(self, frame):
        reg, dx, dy = self._impl.register_frame(frame)
        self._last_shift = (float(dx), float(dy))
        return reg

    def register_frame_with_shift(self, frame):
        reg, dx, dy = self._impl.register_frame(frame)
        self._last_shift = (float(dx), float(dy))
        return reg, dx, dy

    def set_reference(self, reference):
        return self._impl.set_reference(reference)

    def get_shape(self):
        return self._impl.get_shape()

    def get_last_shift(self):
        return getattr(self, "_last_shift", (0.0, 0.0))

    def get_reference_events(self):
        return self._impl.get_reference_events()

    def reset_timing(self):
        return self._impl.reset_timing()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)


class DirectTranslationCPUCppWrapper:
    def __init__(
        self,
        reference,
        downscale_factor=1.0,
        enable_timing=True,
        phase_response_threshold=0.1,
        phase_use_cached_fft=True,
        reference_window_frames=1,
    ):
        if _DirectTranslationCPUCpp is None:
            raise RuntimeError("C++ CPU translation module is not available.")
        self._impl = _DirectTranslationCPUCpp(
            reference,
            downscale_factor=downscale_factor,
            phase_response_threshold=phase_response_threshold,
            phase_use_cached_fft=phase_use_cached_fft,
            reference_window_frames=reference_window_frames,
            enable_timing=enable_timing,
        )

    def register_frame(self, frame):
        reg, dx, dy = self._impl.register_frame(frame)
        self._last_shift = (float(dx), float(dy))
        return reg

    def register_frame_with_shift(self, frame):
        reg, dx, dy = self._impl.register_frame(frame)
        self._last_shift = (float(dx), float(dy))
        return reg, dx, dy

    def set_reference(self, reference):
        return self._impl.set_reference(reference)

    def get_shape(self):
        return self._impl.get_shape()

    def get_last_shift(self):
        return getattr(self, "_last_shift", (0.0, 0.0))

    def get_reference_events(self):
        return self._impl.get_reference_events()

    def reset_timing(self):
        return self._impl.reset_timing()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)


def _feature_type_name(feature_extractor):
    if feature_extractor is None:
        return "ORB"
    if isinstance(feature_extractor, FeatureExtractorType):
        return feature_extractor.name
    name = getattr(feature_extractor, "__class__", type("x", (), {})).__name__
    upper = name.upper()
    if "FAST" in upper and "BRIEF" in upper:
        return "FAST_BRIEF"
    if "SIFT" in upper:
        return "SIFT"
    if "AKAZE" in upper:
        return "AKAZE"
    if "BRISK" in upper:
        return "BRISK"
    return "ORB"


def _matcher_type_name(matcher):
    if matcher is None:
        return "BF"
    if isinstance(matcher, MatcherType):
        return matcher.name
    name = getattr(matcher, "__class__", type("x", (), {})).__name__
    upper = name.upper()
    if "KNN" in upper:
        return "KNN"
    if "FLANN" in upper:
        return "FLANN"
    return "BF"



class HomographyTranslation:
    def __init__(self, reference,
                 matcher=MatcherType.BF,
                 feature_extractor=FeatureExtractorType.ORB,
                 downscale_factor=1.0,
                 knn_ratio=0.75,
                 ransac_reproj_threshold=5.0,
                 min_inliers=0,
                 enable_timing=True,
                 use_cuda=False,
                 use_cpp=True):
        impl = None
        if use_cpp and use_cuda:
            if _HomographyTranslationGPUCpp is None:
                raise RuntimeError("HomographyTranslation: C++ GPU module is not available.")
            if not _HomographyTranslationGPUCpp.is_available():
                raise RuntimeError("HomographyTranslation: C++ GPU module not available on this device.")
            impl = HomographyTranslationGPUCppWrapper(
                reference,
                matcher=matcher,
                feature_extractor=feature_extractor,
                downscale_factor=downscale_factor,
                knn_ratio=knn_ratio,
                ransac_reproj_threshold=ransac_reproj_threshold,
                min_inliers=min_inliers,
                enable_timing=enable_timing,
            )
        elif use_cpp and not use_cuda:
            if _HomographyTranslationCPUCpp is None:
                raise RuntimeError("HomographyTranslation: C++ CPU module is not available.")
            impl = HomographyTranslationCPUCppWrapper(
                reference,
                matcher=matcher,
                feature_extractor=feature_extractor,
                downscale_factor=downscale_factor,
                knn_ratio=knn_ratio,
                ransac_reproj_threshold=ransac_reproj_threshold,
                min_inliers=min_inliers,
                enable_timing=enable_timing,
            )
        else:
            impl = HomographyTranslationCPU(
                reference,
                matcher=matcher,
                feature_extractor=feature_extractor,
                downscale_factor=downscale_factor,
                knn_ratio=knn_ratio,
                ransac_reproj_threshold=ransac_reproj_threshold,
                min_inliers=min_inliers,
                enable_timing=enable_timing,
            )
        self._impl = impl

    def __getattr__(self, name):
        return getattr(self._impl, name)


class HomographyTranslationCPU:
    def __init__(self, reference,
                 matcher=MatcherType.BF,
                 feature_extractor=FeatureExtractorType.ORB,
                 downscale_factor=1.0,
                 knn_ratio=0.75,
                 ransac_reproj_threshold=5.0,
                 min_inliers=0,
                 enable_timing=True,
                 reference_window_frames=1):
        self.feature_extractor = self._build_feature_extractor(feature_extractor)
        self._norm_type = self.feature_extractor.norm_type
        self._flann_index_params = self.feature_extractor.flann_index_params
        self.matcher = self._build_matcher(matcher)
        self._downscale_factor = float(downscale_factor) if downscale_factor else 1.0
        # Import here to avoid circular imports and to defer dependency until init time.
        from . import video_reader
        self._to_gray = self._to_gray_passthrough if video_reader.input_is_gray() else self._to_gray_bgr
        self._knn_ratio = float(knn_ratio)
        self._ransac_reproj_threshold = float(ransac_reproj_threshold)
        self._min_inliers = int(min_inliers) if min_inliers else 0

        s = self._downscale_factor
        self._scale_matrix = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        self._scale_matrix_inv = np.array([[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        self._frame_count = 0
        self._prev_frame = None
        self._reference_events = []
        self._last_reference_reason = None
        self.set_reference(reference, record_event=False, reason="init")
        self._reference_window_frames = max(1, int(reference_window_frames))

        self._timing_enabled = bool(enable_timing)
        self._register = self._register_frame_timed if self._timing_enabled else self._register_frame_fast
        self._last_fail_reason = None
        self._reg_stats = {
            "first_try": {"calls": 0, "total": 0.0},
            "fallback": {"calls": 0, "total": 0.0},
            "re_registration_calls": 0,
        }
        self._timing = None
        if self._timing_enabled:
            self._timing = {
                "calls": 0,
                "registration": 0.0,
                "gray": 0.0,
                "keypoints": 0.0,
                "match": 0.0,
                "homography": 0.0,
                "warp": 0.0,
            }
        self._last_match_count = 0
        self._last_inlier_count = 0
        self._last_overlap = 0.0

    def _build_matcher(self, matcher):
        if matcher is None:
            return BFMatcher(norm_type=self._norm_type)
        if isinstance(matcher, MatcherType):
            if matcher == MatcherType.BF:
                return BFMatcher(norm_type=self._norm_type)
            if matcher == MatcherType.KNN:
                return KNNMatcher(norm_type=self._norm_type)
            if matcher == MatcherType.FLANN:
                return FlannMatcher(index_params=self._flann_index_params)
        return matcher

    def _build_feature_extractor(self, feature_extractor):
        if feature_extractor is None:
            return OrbFeatureExtractor()
        if isinstance(feature_extractor, FeatureExtractorType):
            if feature_extractor == FeatureExtractorType.ORB:
                return OrbFeatureExtractor()
            if feature_extractor == FeatureExtractorType.FAST_BRIEF:
                return FastBriefFeatureExtractor()
            if feature_extractor == FeatureExtractorType.AKAZE:
                return AkazeFeatureExtractor()
            if feature_extractor == FeatureExtractorType.BRISK:
                return BriskFeatureExtractor()
            if feature_extractor == FeatureExtractorType.SIFT:
                return SiftFeatureExtractor()
        return feature_extractor
        
    def set_reference(self, reference, record_event=True, reason="manual"):
        # Set frame geometry parameters
        self.h, self.w = reference.shape[0:2]
        self._frame_corners = np_float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        self.frame_area = float(self.w * self.h)
        self.margin = int(min(self.h, self.w) * 0.01)
        # Extract features from reference frame
        gray_full = self._to_gray(reference)
        self.gray_ref = self._scale_gray(gray_full)
        self.kp_ref, self.descriptors_ref = self.feature_extractor.get_keypoints_and_descriptors(self.gray_ref)
        if record_event:
            self._reference_events.append(self._frame_count)
            self._last_reference_reason = reason

    def _to_gray_passthrough(self, frame):
        return frame

    def _to_gray_bgr(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _scale_gray(self, gray):
        if self._downscale_factor == 1.0:
            return gray
        return cv2.resize(
            gray,
            None,
            fx=self._downscale_factor,
            fy=self._downscale_factor,
            interpolation=cv2.INTER_AREA,
        )

    def _scale_homography(self, H):
        if self._downscale_factor == 1.0:
            return H
        return self._scale_matrix_inv @ H @ self._scale_matrix

    def get_shape(self):
        return (self.h-self.margin*2, self.w-self.margin*2)

    def get_reference_events(self):
        return np.array(self._reference_events, dtype=np.int32)
    
    def reset_timing(self):
        if not self._timing_enabled:
            return
        for key in self._timing:
            self._timing[key] = 0
        for bucket in ("first_try", "fallback"):
            self._reg_stats[bucket]["calls"] = 0
            self._reg_stats[bucket]["total"] = 0.0
        self._reg_stats["re_registration_calls"] = 0

    def timing_summary(self, reset=False):
        if not self._timing_enabled:
            return {"enabled": False}
        calls = self._timing["calls"]
        if calls == 0:
            summary = {"calls": 0}
        else:
            summary = {
                "calls": calls,
                "registration": (self._timing["registration"] / calls) * 1000.0,
                "gray_ms_avg": (self._timing["gray"] / calls) * 1000.0,
                "keypoints_ms_avg": (self._timing["keypoints"] / calls) * 1000.0,
                "match_ms_avg": (self._timing["match"] / calls) * 1000.0,
                "homography_ms_avg": (self._timing["homography"] / calls) * 1000.0,
                "warp_ms_avg": (self._timing["warp"] / calls) * 1000.0,
                "re_registration_calls": self._reg_stats["re_registration_calls"],
                "registration_first_try_ms_avg": (
                    (self._reg_stats["first_try"]["total"] / self._reg_stats["first_try"]["calls"]) * 1000.0
                    if self._reg_stats["first_try"]["calls"] else 0.0
                ),
                "registration_fallback_ms_avg": (
                    (self._reg_stats["fallback"]["total"] / self._reg_stats["fallback"]["calls"]) * 1000.0
                    if self._reg_stats["fallback"]["calls"] else 0.0
                ),
            }
        if reset:
            self.reset_timing()
        return summary

    def _record_total_time(self, elapsed):
        self._timing["calls"] += 1
        self._timing["registration"] += elapsed

    def _register_with_fallback(self, frame, prev_frame):
        total_start = perf_counter() if self._timing_enabled else None
        gray, kp_frame, descriptors_frame, timing = self._prepare_features(frame)
        out = self._register(frame, prev_frame, kp_frame, descriptors_frame, timing)

        if out is not None:
            if self._timing_enabled:
                self._reg_stats["first_try"]["calls"] += 1
                self._reg_stats["first_try"]["total"] += perf_counter() - total_start
            if descriptors_frame is not None and len(descriptors_frame) >= 4:
                if self._should_update_reference():
                    # Use current frame features as the next reference to avoid re-extracting.
                    self._record_reference_update(gray, kp_frame, descriptors_frame, reason="window")
            return out

        if descriptors_frame is not None and len(descriptors_frame) >= 4:
            if self._should_update_reference():
                # Keep the reference moving even when alignment fails.
                self._record_reference_update(gray, kp_frame, descriptors_frame, reason="window_fallback")
        return frame[self.margin:-self.margin, self.margin:-self.margin]

    def register_frame(self, frame):
        self._frame_count += 1
        prev_frame = self._prev_frame
        out = self._register_with_fallback(frame, prev_frame)

        self._prev_frame = frame
        return out

    def _should_update_reference(self):
        return (self._frame_count % self._reference_window_frames) == 0

    def _record_reference_update(self, gray, kp, descriptors, reason):
        self._set_reference_from_features(gray, kp, descriptors)
        # Event applies to the next frame (first one registered to the new reference).
        self._reference_events.append(self._frame_count + 1)
        self._last_reference_reason = reason

    def _register_frame_fast(self, frame, prev_frame, kp_frame, descriptors_frame, timing=None):
        # Skip alignment if features are missing or too sparse.
        if descriptors_frame is None or len(descriptors_frame) < 4:
            self._last_fail_reason = "descriptors"
            return None
        
        # Match descriptors and sort by distance (best first).
        matches = self.matcher.match(descriptors_frame, self.descriptors_ref)
        # Some matchers already apply ratio filtering (e.g., KNNMatcher); only filter when pairs are provided.
        if matches and isinstance(matches[0], (list, tuple)):
            matches = [
                m[0]
                for m in matches
                if len(m) >= 2 and m[0].distance < self._knn_ratio * m[1].distance
            ]
        self._last_match_count = len(matches)
        if not matches:
            self._last_fail_reason = "match"
            return None
       
        # Build matched point arrays for homography estimation.
        H = self.calc_homography_matrix(matches, kp_frame)
        if H is not None:
            H = self._scale_homography(H)
        if H is None:
            self._last_fail_reason = "homography"
            return None
        if self._min_inliers and self._last_inlier_count < self._min_inliers:
            self._last_fail_reason = "inliers"
            return None
        overlap_ratio = self._overlap_ratio(H)
        self._last_overlap = overlap_ratio
        self._last_fail_reason = None

        # Warp current frame into the reference view.
        reg = cv2.warpPerspective(frame, H, (self.w, self.h))

        # Trim borders that can contain warping artifacts.
        out = reg[self.margin:-self.margin, self.margin:-self.margin]
        #     return None
        return out

    def _register_frame_timed(self, frame, prev_frame, kp_frame, descriptors_frame, timing):
        start = timing["start"]
        t_keypoints = timing["t_keypoints"]
        # Skip alignment if features are missing or too sparse.
        if descriptors_frame is None or len(descriptors_frame) < 4:
            self._last_fail_reason = "descriptors"
            self._record_total_time(perf_counter() - start)
            return None
        
        # Brute-force match descriptors and sort by distance (best first).
        matches = self.matcher.match(descriptors_frame, self.descriptors_ref)
        # Some matchers already apply ratio filtering (e.g., KNNMatcher); only filter when pairs are provided.
        if matches and isinstance(matches[0], (list, tuple)):
            matches = [
                m[0]
                for m in matches
                if len(m) >= 2 and m[0].distance < self._knn_ratio * m[1].distance
            ]
        self._last_match_count = len(matches)
        t_match = perf_counter()
        self._timing["match"] += t_match - t_keypoints

        H = self.calc_homography_matrix(matches, kp_frame)
        if H is not None:
            H = self._scale_homography(H)
        t_h = perf_counter()
        self._timing["homography"] += t_h - t_match
        if H is None:
            self._last_fail_reason = "homography"
            self._record_total_time(perf_counter() - start)
            return None
        if self._min_inliers and self._last_inlier_count < self._min_inliers:
            self._last_fail_reason = "inliers"
            self._record_total_time(perf_counter() - start)
            return None
        overlap_ratio = self._overlap_ratio(H)
        self._last_overlap = overlap_ratio
        self._last_fail_reason = None

        # Warp current frame into the reference view.
        reg = cv2.warpPerspective(frame, H, (self.w, self.h))
        t_warp = perf_counter()
        self._timing["warp"] += t_warp - t_h

        # Trim borders that can contain warping artifacts.
        out = reg[self.margin:-self.margin, self.margin:-self.margin]
        self._record_total_time(perf_counter() - start)
        return out

    def _prepare_features(self, frame):
        if self._timing_enabled:
            start = perf_counter()
            gray_full = self._to_gray(frame)
            gray = self._scale_gray(gray_full)
            t_gray = perf_counter()
            self._timing["gray"] += t_gray - start
            kp_frame, descriptors_frame = self._extract_features(gray)
            t_keypoints = perf_counter()
            self._timing["keypoints"] += t_keypoints - t_gray
            timing = {"start": start, "t_keypoints": t_keypoints}
        else:
            gray_full = self._to_gray(frame)
            gray = self._scale_gray(gray_full)
            kp_frame, descriptors_frame = self._extract_features(gray)
            timing = None
        return gray, kp_frame, descriptors_frame, timing
    
    def calc_homography_matrix(self, matches, kp_frame):
        if matches is None or len(matches) < 4:
            self._last_inlier_count = 0
            return None

        # Build matched point arrays for homography estimation.
        pts1 = np_float32([kp_frame[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np_float32([self.kp_ref[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        # Estimate homography with cv2.RANSAC to reject outliers.
        H, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            self._ransac_reproj_threshold
        )
        if mask is not None:
            self._last_inlier_count = int(mask.sum())
        else:
            self._last_inlier_count = 0
        return H

    def get_last_homography_debug(self):
        return {
            "fail_reason": self._last_fail_reason,
            "match_count": self._last_match_count,
            "inlier_count": self._last_inlier_count,
            "overlap_ratio": self._last_overlap,
            "reference_reason": self._last_reference_reason,
        }

    def _set_reference_from_features(self, gray, kp, descriptors):
        self.gray_ref = gray
        self.kp_ref = kp
        self.descriptors_ref = descriptors


    def _overlap_ratio(self, H):
        dst = cv2.perspectiveTransform(self._frame_corners, H)
        dst_hull = cv2.convexHull(dst)
        inter_area, _ = cv2.intersectConvexConvex(dst_hull, self._frame_corners)
        if inter_area <= 0:
            return 0.0
        return inter_area / self.frame_area

    def _extract_features(self, gray):
        return self.feature_extractor.get_keypoints_and_descriptors(gray)


class HomographyTranslationGPUCppWrapper:
    def __init__(
        self,
        reference,
        matcher=MatcherType.BF,
        feature_extractor=FeatureExtractorType.ORB,
        downscale_factor=1.0,
        knn_ratio=0.75,
        ransac_reproj_threshold=5.0,
        min_inliers=0,
        reference_window_frames=1,
        enable_timing=True,
    ):
        if _HomographyTranslationGPUCpp is None:
            raise RuntimeError("C++ GPU homography module is not available.")
        self._impl = _HomographyTranslationGPUCpp(
            reference,
            _feature_type_name(feature_extractor),
            _matcher_type_name(matcher),
            downscale_factor=downscale_factor,
            knn_ratio=knn_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            min_inliers=min_inliers,
            reference_window_frames=reference_window_frames,
            enable_timing=enable_timing,
        )

    def register_frame(self, frame):
        return self._impl.register_frame(frame)

    def set_reference(self, reference):
        return self._impl.set_reference(reference)

    def get_shape(self):
        return self._impl.get_shape()

    def reset_timing(self):
        return self._impl.reset_timing()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)

    def get_reference_events(self):
        return self._impl.get_reference_events()

    def get_last_homography_debug(self):
        return self._impl.get_last_homography_debug()


class HomographyTranslationCPUCppWrapper:
    def __init__(
        self,
        reference,
        matcher=MatcherType.BF,
        feature_extractor=FeatureExtractorType.ORB,
        downscale_factor=1.0,
        knn_ratio=0.75,
        ransac_reproj_threshold=5.0,
        min_inliers=0,
        reference_window_frames=1,
        enable_timing=True,
    ):
        if _HomographyTranslationCPUCpp is None:
            raise RuntimeError("C++ CPU homography module is not available.")
        self._impl = _HomographyTranslationCPUCpp(
            reference,
            _feature_type_name(feature_extractor),
            _matcher_type_name(matcher),
            downscale_factor=downscale_factor,
            knn_ratio=knn_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            min_inliers=min_inliers,
            reference_window_frames=reference_window_frames,
            enable_timing=enable_timing,
        )

    def register_frame(self, frame):
        return self._impl.register_frame(frame)

    def set_reference(self, reference):
        return self._impl.set_reference(reference)

    def get_shape(self):
        return self._impl.get_shape()

    def reset_timing(self):
        return self._impl.reset_timing()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)

    def get_reference_events(self):
        return self._impl.get_reference_events()

    def get_last_homography_debug(self):
        return self._impl.get_last_homography_debug()
