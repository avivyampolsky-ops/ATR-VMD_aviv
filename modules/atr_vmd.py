from .registrator import Registrator
from .detector import Detector, DetectorType
from .kalman import KalmanIoUTracker
import cv2
from time import perf_counter


class _TimingProxy:
    def __init__(self, impl, combined, kind):
        # Proxy wrapper to expose per-component timing summaries when a combined C++ pipeline is used.
        self._impl = impl
        self._combined = combined
        self._kind = kind

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def timing_summary(self, reset=False):
        if self._combined is None:
            return self._impl.timing_summary(reset=reset)
        summary = self._combined.timing_summary(reset=reset)
        if self._kind == "registrator":
            out = {
                "enabled": summary.get("enabled", True),
                "calls": summary.get("calls", 0),
            }
            if "registration_ms_avg" in summary:
                out["registration"] = summary["registration_ms_avg"]
            return out
        if self._kind == "detector":
            out = {
                "enabled": summary.get("enabled", True),
                "calls": summary.get("calls", 0),
            }
            for key in (
                "blur_ms_avg",
                "gray_ms_avg",
                "diff_ms_avg",
                "bgsub_ms_avg",
                "contours_ms_avg",
                "bbox_creation_ms_avg",
                "upload_ms_avg",
                "download_ms_avg",
                "total_ms_avg",
            ):
                if key in summary:
                    out[key] = summary[key]
            return out
        return summary

try:
    from ._register_detect_cpp import RegisterAndDetectGPUCpp as _RegisterAndDetectGPUCpp
except Exception:
    _RegisterAndDetectGPUCpp = None
try:
    from ._register_detect_cpp import KalmanIoUTrackerCpp as _KalmanIoUTrackerCpp
except Exception:
    _KalmanIoUTrackerCpp = None

class ATR_VMD:
    def __init__(self, reference, config_loader):
        cfg = config_loader
        registration_mode = cfg.get("registration.mode", "translation")
        matcher = cfg.matcher()
        feature_extractor = cfg.feature_extractor()
        translation_method = cfg.get("registration.translation.method", "phase")
        phase_use_cached_fft = cfg.get("registration.translation.phase_use_cached_fft", True)
        phase_response_threshold = cfg.get("registration.translation.phase_response_threshold", 0.1)
        knn_ratio = cfg.get("registration.homography.knn_ratio", 0.75)
        ransac_reproj_threshold = cfg.get("registration.homography.ransac_reproj_threshold", 5.0)
        min_inliers = cfg.get("registration.homography.min_inliers", 0)
        downscale_factor = cfg.get("registration.downscale_factor", 1.0)
        reference_window_frames = cfg.get("registration.reference_window_frames", 1)
        detect_scale = cfg.get("detection.detect_scale", 1.0)
        learning_rate = cfg.get("detection.learning_rate", 0.05)
        mog2_var_threshold = cfg.get("detection.mog2_var_threshold", 20.0)
        debug_mode = cfg.get("debug.debug_mode", True)
        reanchor_log_path = cfg.get("debug.reanchor_log_path", "") if debug_mode else ""
        max_shift = cfg.get("registration.translation.max_shift", 0.0)
        use_cuda = cfg.get("general.use_cuda", False)
        use_cpp = cfg.get("general.use_cpp", True)
        enable_timing = debug_mode and cfg.get("debug.enable_timing", True)
        draw_output = cfg.get("debug.draw_output", False)
        show_pipeline = cfg.get("debug.show_pipeline", False)

        self.draw_output = draw_output
        
        self.registrator = Registrator(
            reference,
            matcher=matcher,
            feature_extractor=feature_extractor,
            registration_mode=registration_mode,
            translation_method=translation_method,
            phase_use_cached_fft=phase_use_cached_fft,
            phase_response_threshold=phase_response_threshold,
            knn_ratio=knn_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            min_inliers=min_inliers,
            downscale_factor=downscale_factor,
            reference_window_frames=reference_window_frames,
            use_cuda=use_cuda,
            use_cpp=use_cpp,
            enable_timing=enable_timing,
        )
        self.shape_with_margin = self.registrator.get_shape()
        self.detector = Detector(
            frame_shape=self.shape_with_margin,
            detector_type=DetectorType.MOG2,
            enable_timing=enable_timing,
            detect_scale=detect_scale,
            learning_rate=learning_rate,
            reference_window_frames=reference_window_frames,
            mog2_var_threshold=mog2_var_threshold,
            use_cuda=use_cuda,
            use_cpp=use_cpp,
        )
        self._register_and_detect = None
        if (use_cpp and use_cuda
            and registration_mode == "translation"
            and _RegisterAndDetectGPUCpp is not None):
            try:
                from . import video_reader
                input_is_gray = bool(video_reader.input_is_gray())
                if _RegisterAndDetectGPUCpp.is_available():
                    self._register_and_detect = _RegisterAndDetectGPUCpp(
                        reference,
                        input_is_gray,
                        downscale_factor=downscale_factor,
                        detect_scale=detect_scale,
                        learning_rate=learning_rate,
                        reference_window_frames=reference_window_frames,
                        mog2_var_threshold=mog2_var_threshold,
                        phase_response_threshold=phase_response_threshold,
                        reanchor_log_path=reanchor_log_path,
                        max_shift=max_shift,
                        phase_use_cached_fft=phase_use_cached_fft,
                        enable_timing=enable_timing,
                    )
            except Exception:
                if use_cpp:
                    raise
                print("ATR_VMD: C++ register+detect init failed, using Python pipeline")
        if use_cpp and use_cuda and registration_mode == "translation" and self._register_and_detect is None:
            raise RuntimeError("ATR_VMD: C++ register+detect pipeline requested but unavailable.")

        if self._register_and_detect is not None:
            self.registrator = _TimingProxy(self.registrator, self._register_and_detect, kind="registrator")
            self.detector = _TimingProxy(self.detector, self._register_and_detect, kind="detector")

        self.prev_gray = None
        self.frame_count = 0
        self._last_detections = None
        self._last_shift = (0.0, 0.0)
        self._prev_shift = None
        kalman_iou_thresh = cfg.get("tracker.kalman.iou_thresh", 0.05)
        kalman_max_lost = cfg.get("tracker.kalman.max_lost", 8)
        kalman_min_move = cfg.get("tracker.kalman.min_move", 2.0)
        kalman_ema_alpha = cfg.get("tracker.kalman.ema_alpha", 0.3)
        kalman_dist_gate_scale = cfg.get("tracker.kalman.dist_gate_scale", 2.0)
        kalman_easy_iou_thresh = cfg.get("tracker.kalman.easy_iou_thresh", 0.6)

        if use_cpp and _KalmanIoUTrackerCpp is not None:
            self.tracker = _KalmanIoUTrackerCpp(
                kalman_iou_thresh,
                int(kalman_max_lost),
                kalman_min_move,
                kalman_ema_alpha,
                kalman_dist_gate_scale,
                kalman_easy_iou_thresh,
            )
        else:
            self.tracker = KalmanIoUTracker(
                iou_thresh=kalman_iou_thresh,
                max_lost=int(kalman_max_lost),
                min_move=kalman_min_move,
                ema_alpha=kalman_ema_alpha,
                dist_gate_scale=kalman_dist_gate_scale,
                easy_iou_thresh=kalman_easy_iou_thresh,
            )

        self._timing_enabled = bool(enable_timing)
        self._tracker_timing = None
        if self._timing_enabled:
            self._tracker_timing = {"calls": 0, "update": 0.0}
        if show_pipeline:
            self._print_pipeline_summary(registration_mode, use_cuda)

    def _print_pipeline_summary(self, registration_mode, use_cuda):
        if self._register_and_detect is not None:
            print("")
            print("Pipeline:")
            print("  register+detect: translation | cpp | gpu")
            print("")
            return

        reg_impl = self.registrator
        if isinstance(reg_impl, _TimingProxy):
            reg_impl = reg_impl._impl
        reg_backend = getattr(reg_impl, "_impl", reg_impl)
        reg_name = reg_backend.__class__.__name__
        reg_lang = "cpp" if "Cpp" in reg_name else "python"
        if registration_mode == "homography":
            reg_device = "cpu"
        else:
            reg_device = "gpu" if "GPU" in reg_name else "cpu"

        det_impl = self.detector
        if isinstance(det_impl, _TimingProxy):
            det_impl = det_impl._impl
        det_backend = getattr(det_impl, "background_substractor", det_impl)
        det_name = det_backend.__class__.__name__
        det_lang = "cpp" if "Cpp" in det_name else "python"
        det_device = "gpu" if "GPU" in det_name else ("gpu" if use_cuda and det_lang == "cpp" else "cpu")

        tracker_impl = self.tracker
        tracker_name = tracker_impl.__class__.__name__
        tracker_lang = "cpp" if "Cpp" in tracker_name else "python"
        tracker_device = "cpu"

        print("")
        print("Pipeline:")
        print(f"  registration: {registration_mode} | {reg_lang} | {reg_device}")
        print(f"  detection: mog2 | {det_lang} | {det_device}")
        print(f"  tracker: kalman | {tracker_lang} | {tracker_device}")
        print("")

    def reset_tracker_timing(self):
        if not self._timing_enabled:
            return
        for key in self._tracker_timing:
            self._tracker_timing[key] = 0

    def tracker_timing_summary(self, reset=False):
        if not self._timing_enabled:
            return {"enabled": False}
        calls = self._tracker_timing["calls"]
        summary = {"enabled": True, "calls": calls}
        if calls:
            summary["update_ms_avg"] = (self._tracker_timing["update"] / calls) * 1000.0
        if reset:
            self.reset_tracker_timing()
        return summary

    def register_frame(self, frame):
        return self.registrator.register_frame(frame)
    
    def detect_objects(self, frame):
        return self.detector.detect(frame)

    def get_last_mask(self):
        if self._register_and_detect is not None:
            getter = getattr(self._register_and_detect, "get_last_mask", None)
            if getter is None:
                return None
            return getter()
        getter = getattr(self.detector, "get_last_mask", None)
        if getter is None:
            return None
        return getter()

    def get_last_cc_mask(self):
        if self._register_and_detect is not None:
            getter = getattr(self._register_and_detect, "get_last_cc_mask", None)
            if getter is None:
                return None
            return getter()
        getter = getattr(self.detector, "get_last_cc_mask", None)
        if getter is None:
            return None
        return getter()

    def get_last_bb_mask(self):
        if self._register_and_detect is not None:
            getter = getattr(self._register_and_detect, "get_last_bb_mask", None)
            if getter is None:
                return None
            return getter()
        getter = getattr(self.detector, "get_last_bb_mask", None)
        if getter is None:
            return None
        return getter()

    def get_reference_events(self):
        if self._register_and_detect is not None:
            getter = getattr(self._register_and_detect, "get_reference_events", None)
            if getter is None:
                return None
            return getter()
        getter = getattr(self.registrator, "get_reference_events", None)
        if getter is None:
            return None
        return getter()

    def register_and_detect(self, frame):
        if self._register_and_detect is not None:
            detections, registered_frame = self._register_and_detect.register_and_detect(frame)
            getter = getattr(self._register_and_detect, "get_last_shift", None)
            if getter is not None:
                self._last_shift = getter()
            else:
                self._last_shift = (0.0, 0.0)
            ref_events = self.get_reference_events()
            if ref_events is not None and len(ref_events):
                current_reg_frame = self.frame_count + 1
                if int(ref_events[-1]) == current_reg_frame:
                    if hasattr(self.tracker, "reset"):
                        self.tracker.reset()
                    self._prev_shift = None
            return detections, registered_frame
        registered_frame, dx, dy = self.registrator.register_frame_with_shift(frame)
        self._last_shift = (float(dx), float(dy))
        ref_events = self.get_reference_events()
        if ref_events is not None and len(ref_events):
            current_reg_frame = self.frame_count + 1
            if int(ref_events[-1]) == current_reg_frame:
                self.detector.on_reanchor()
                if hasattr(self.tracker, "reset"):
                    self.tracker.reset()
                self._prev_shift = None
        detections = self.detect_objects(registered_frame)
        return detections, registered_frame
    
    def process_frame(self, frame):
        detections, registered_frame = self.register_and_detect(frame)
        self._last_registered_frame = registered_frame
        self._last_detections = detections
        if self._prev_shift is not None:
            dx = float(self._last_shift[0]) - float(self._prev_shift[0])
            dy = float(self._last_shift[1]) - float(self._prev_shift[1])
            if hasattr(self.tracker, "apply_shift"):
                # Compensate tracker state for registration jumps so IoU matching stays stable.
                self.tracker.apply_shift(dx, dy)
        self._prev_shift = self._last_shift
        if self._timing_enabled:
            start = perf_counter()
            tracks = self.tracker.update(detections)
            elapsed = perf_counter() - start
            self._tracker_timing["calls"] += 1
            self._tracker_timing["update"] += elapsed
        else:
            tracks = self.tracker.update(detections)
    
        self.frame_count += 1
        return self.tracker.get_tracks()

    def get_last_detections(self):
        return self._last_detections

    def get_last_shift(self):
        return self._last_shift

    def get_last_registered_frame(self):
        return getattr(self, "_last_registered_frame", None)

    def get_registration_debug(self):
        if self._register_and_detect is not None:
            getter = getattr(self._register_and_detect, "get_last_registration_debug", None)
            if getter is None:
                return None
            return getter()
        return None
