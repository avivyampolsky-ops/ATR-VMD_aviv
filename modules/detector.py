import cv2
import numpy as np
from enum import Enum
from time import perf_counter

try:
    from ._detector_cpp import BackgroundSubtractorMOG2GPUCpp as _BackgroundSubtractorMOG2GPUCpp
except Exception:
    _BackgroundSubtractorMOG2GPUCpp = None

class BackgroundSubtractorMOG2:
    def __init__(
        self,
        frame_shape,
        enable_timing=True,
        detect_scale=1.0,
        learning_rate=0.05,
        reference_window_frames=None,
        mog2_var_threshold=20.0,
    ):
        self._mog2_history = int(reference_window_frames) if reference_window_frames else 400
        self._mog2_var_threshold = float(mog2_var_threshold)
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=self._mog2_history,
            varThreshold=self._mog2_var_threshold,
            detectShadows=False
        )
        self.prev_gray = None
        self._detect_scale = float(detect_scale) if detect_scale else 1.0
        self._scale_inv = 1.0 / self._detect_scale
        self._learning_rate = float(learning_rate)
        if self._detect_scale != 1.0:
            h, w = frame_shape[:2]
            h = max(1, int(round(h * self._detect_scale)))
            w = max(1, int(round(w * self._detect_scale)))
            self._scaled_shape = (h, w)
        else:
            self._scaled_shape = frame_shape[:2]
        self.zeros = np.zeros(self._scaled_shape, dtype=np.uint8)
        from . import video_reader
        input_is_gray = video_reader.input_is_gray()
        self._to_gray = self._to_gray_passthrough if input_is_gray else self._to_gray_bgr
        self._timing_enabled = bool(enable_timing)
        self._timing = None
        self._last_mask = None
        self._full_shape = frame_shape[:2]
        self._border_margin = 10
        # Mask cleanup for bolder blobs and less speckle noise.
        self._min_contour_area = 50
        self._max_contour_area = 1500
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._close_iterations = 1
        self._open_iterations = 1
        self._dilate_iterations = 2
        if self._timing_enabled:
            self._timing = {
                "calls": 0,
                "blur": 0.0,
                "gray": 0.0,
                "diff": 0.0,
                "bgsub": 0.0,
                "contours": 0.0,
                "total": 0.0,
            }
        self._last_cc_mask = None
        self._last_bb_mask = None

    def on_reanchor(self):
        # Reset MOG2 model to avoid foreground flood after reference changes.
        self.backsub = cv2.createBackgroundSubtractorMOG2(
            history=self._mog2_history,
            varThreshold=self._mog2_var_threshold,
            detectShadows=False,
        )
        self.prev_gray = None

    def detect(self, frame):
        if not self._timing_enabled:
            return self._detect_fast(frame)
        return self._detect_timed(frame)

    def _detect_fast(self, frame):
        if self._detect_scale != 1.0:
            frame_small = cv2.resize(
                frame,
                (self._scaled_shape[1], self._scaled_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_small = frame
        # Foreground detection from frame differencing + background subtraction.
        blur_small = cv2.GaussianBlur(frame_small, (3, 3), 0)
        gray = self._to_gray(blur_small)

        # Mask: background subtraction only.
        fg = self.backsub.apply(blur_small, learningRate=self._learning_rate) # TODO: tune. higher adapts faster, lower adapts slower.

        mask = self._postprocess_mask(fg)
        self._last_mask = mask

        # Find contours and filter by area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea
        contour_bbox = cv2.boundingRect
        detections = []
        for c in contours:
            area = contour_area(c)
            if not (self._min_contour_area < area < self._max_contour_area):
                continue
            bbox = contour_bbox(c)
            if not bbox:
                continue
            if not (0.2 <= (bbox[2] / bbox[3]) <= 5.0):
                continue
            if self._detect_scale != 1.0:
                x, y, w, h = bbox
                x = int(round(x * self._scale_inv))
                y = int(round(y * self._scale_inv))
                w = int(round(w * self._scale_inv))
                h = int(round(h * self._scale_inv))
                bbox = (x, y, w, h)
            if self._touches_border(bbox):
                continue
            detections.append(bbox)

        cc_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        for i, c in enumerate(contours):
            color = ((37 * i) % 255, (17 * i) % 255, (97 * i) % 255)
            cv2.drawContours(cc_mask, [c], 0, color, thickness=cv2.FILLED)
        self._last_cc_mask = cc_mask
        bb_mask = self._last_cc_mask.copy()
        for x, y, w, h in detections:
            cv2.rectangle(bb_mask, (x, y), (x + w, y + h), (0, 0, 255), 1)
        self._last_bb_mask = bb_mask

        if detections:
            detections_arr = np.array(detections, dtype=np.int32)
        else:
            detections_arr = np.empty((0, 4), dtype=np.int32)
        return detections_arr

    def _detect_timed(self, frame):
        start = perf_counter()

        if self._detect_scale != 1.0:
            frame_small = cv2.resize(
                frame,
                (self._scaled_shape[1], self._scaled_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            frame_small = frame
        blur_small = cv2.GaussianBlur(frame_small, (3, 3), 0)
        t_blur = perf_counter()
        self._timing["blur"] += t_blur - start

        gray = self._to_gray(blur_small)
        t_gray = perf_counter()
        self._timing["gray"] += t_gray - t_blur

        fg = self.backsub.apply(blur_small, learningRate=self._learning_rate) # TODO: tune. higher adapts faster, lower adapts slower.
        t_fg = perf_counter()
        self._timing["bgsub"] += t_fg - t_gray

        mask = self._postprocess_mask(fg)
        self._last_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea
        contour_bbox = cv2.boundingRect
        detections = []
        for c in contours:
            area = contour_area(c)
            if not (self._min_contour_area < area < self._max_contour_area):
                continue
            bbox = contour_bbox(c)
            if not bbox:
                continue
            if not (0.2 <= (bbox[2] / bbox[3]) <= 5.0):
                continue
            if self._detect_scale != 1.0:
                x, y, w, h = bbox
                x = int(round(x * self._scale_inv))
                y = int(round(y * self._scale_inv))
                w = int(round(w * self._scale_inv))
                h = int(round(h * self._scale_inv))
                bbox = (x, y, w, h)
            if self._touches_border(bbox):
                continue
            detections.append(bbox)
        t_contours = perf_counter()
        self._timing["contours"] += t_contours - t_fg

        cc_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        for i, c in enumerate(contours):
            color = ((37 * i) % 255, (17 * i) % 255, (97 * i) % 255)
            cv2.drawContours(cc_mask, [c], 0, color, thickness=cv2.FILLED)
        self._last_cc_mask = cc_mask
        bb_mask = self._last_cc_mask.copy()
        for x, y, w, h in detections:
            cv2.rectangle(bb_mask, (x, y), (x + w, y + h), (0, 0, 255), 1)
        self._last_bb_mask = bb_mask

        self._record_total_time(t_contours - start)
        if detections:
            detections_arr = np.array(detections, dtype=np.int32)
        else:
            detections_arr = np.empty((0, 4), dtype=np.int32)
        return detections_arr

    def calc_diff(self, gray):
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        else:
            diff = self.zeros

        self.prev_gray = gray

        return diff

    def _to_gray_passthrough(self, frame):
        return frame

    def _to_gray_bgr(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _postprocess_mask(self, mask):
        # Connect components, then remove remaining speckle noise.
        if self._close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self._morph_kernel,
                iterations=self._close_iterations,
            )
        if self._open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self._morph_kernel,
                iterations=self._open_iterations,
            )
        if self._dilate_iterations > 0:
            mask = cv2.dilate(mask, self._morph_kernel, iterations=self._dilate_iterations)
        return mask

    def _touches_border(self, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return True
        h_full, w_full = self._full_shape
        margin = self._border_margin
        return (
            x <= margin
            or y <= margin
            or (x + w) >= (w_full - margin)
            or (y + h) >= (h_full - margin)
        )


    def timing_summary(self, reset=False):
        if not self._timing_enabled:
            return {"enabled": False}
        calls = self._timing["calls"]
        if calls == 0:
            summary = {"calls": 0}
        else:
            summary = {
                "calls": calls,
                "total_ms_avg": (self._timing["total"] / calls) * 1000.0,
                "blur_ms_avg": (self._timing["blur"] / calls) * 1000.0,
                "gray_ms_avg": (self._timing["gray"] / calls) * 1000.0,
                "diff_ms_avg": (self._timing["diff"] / calls) * 1000.0,
                "bgsub_ms_avg": (self._timing["bgsub"] / calls) * 1000.0,
                "contours_ms_avg": (self._timing["contours"] / calls) * 1000.0,
            }
        if reset:
            self.reset_timing()
        return summary

    def get_last_mask(self):
        if self._last_mask is None:
            return None
        if self._detect_scale == 1.0:
            return self._last_mask
        return cv2.resize(
            self._last_mask,
            (self._full_shape[1], self._full_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def get_last_cc_mask(self):
        if self._last_cc_mask is None:
            return None
        if self._detect_scale == 1.0:
            return self._last_cc_mask
        return cv2.resize(
            self._last_cc_mask,
            (self._full_shape[1], self._full_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def get_last_bb_mask(self):
        if self._last_bb_mask is None:
            return None
        if self._detect_scale == 1.0:
            return self._last_bb_mask
        return cv2.resize(
            self._last_bb_mask,
            (self._full_shape[1], self._full_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def reset_timing(self):
        if not self._timing_enabled:
            return
        for key in self._timing:
            self._timing[key] = 0

    def _record_total_time(self, elapsed):
        self._timing["calls"] += 1
        self._timing["total"] += elapsed

class DetectorType(Enum):
    MOG2 = 1

class Detector:
    def __init__(
        self,
        frame_shape,
        detector_type=DetectorType.MOG2,
        enable_timing=True,
        detect_scale=1.0,
        learning_rate=0.05,
        reference_window_frames=None,
        mog2_var_threshold=20.0,
        use_cuda=False,
        use_cpp=True,
    ):
        self.background_substractor = self._build_detector(
            detector_type,
            frame_shape,
            enable_timing,
            detect_scale,
            learning_rate,
            reference_window_frames,
            mog2_var_threshold,
            use_cuda,
            use_cpp,
        )

    def detect(self, frame):
        return self.background_substractor.detect(frame)

    def on_reanchor(self):
        handler = getattr(self.background_substractor, "on_reanchor", None)
        if handler is not None:
            handler()

    def timing_summary(self, reset=False):
        return self.background_substractor.timing_summary(reset=reset)

    def get_last_mask(self):
        getter = getattr(self.background_substractor, "get_last_mask", None)
        if getter is None:
            return None
        return getter()

    def get_last_cc_mask(self):
        getter = getattr(self.background_substractor, "get_last_cc_mask", None)
        if getter is None:
            return None
        return getter()

    def get_last_bb_mask(self):
        getter = getattr(self.background_substractor, "get_last_bb_mask", None)
        if getter is None:
            return None
        return getter()

    def _build_detector(
        self,
        detector_type,
        frame_shape,
        enable_timing,
        detect_scale,
        learning_rate,
        reference_window_frames,
        mog2_var_threshold,
        use_cuda,
        use_cpp,
    ):
        if detector_type == DetectorType.MOG2:
            if use_cpp and use_cuda:
                if _BackgroundSubtractorMOG2GPUCpp is None:
                    raise RuntimeError("Detector: C++ GPU module is not available.")
                if not _BackgroundSubtractorMOG2GPUCpp.is_available():
                    raise RuntimeError("Detector: C++ GPU module not available on this device.")
                return BackgroundSubtractorMOG2GPUCppWrapper(
                    frame_shape,
                    enable_timing=enable_timing,
                    detect_scale=detect_scale,
                    learning_rate=learning_rate,
                    reference_window_frames=reference_window_frames,
                    mog2_var_threshold=mog2_var_threshold,
                )
            return BackgroundSubtractorMOG2(
                frame_shape,
                enable_timing=enable_timing,
                detect_scale=detect_scale,
                learning_rate=learning_rate,
                reference_window_frames=reference_window_frames,
                mog2_var_threshold=mog2_var_threshold,
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")


class BackgroundSubtractorMOG2GPUCppWrapper:
    def __init__(
        self,
        frame_shape,
        enable_timing=True,
        detect_scale=1.0,
        learning_rate=0.05,
        reference_window_frames=None,
        mog2_var_threshold=20.0,
    ):
        if _BackgroundSubtractorMOG2GPUCpp is None:
            raise RuntimeError("C++ GPU detector module is not available.")
        from . import video_reader
        input_is_gray = bool(video_reader.input_is_gray())
        self._impl = _BackgroundSubtractorMOG2GPUCpp(
            frame_shape,
            input_is_gray,
            enable_timing,
            detect_scale,
            learning_rate,
            int(reference_window_frames) if reference_window_frames else 1,
            mog2_var_threshold,
        )

    def detect(self, frame):
        return self._impl.detect(frame)

    def on_reanchor(self):
        handler = getattr(self._impl, "on_reanchor", None)
        if handler is not None:
            handler()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)

    def reset_timing(self):
        return self._impl.reset_timing()

    def get_last_mask(self):
        return self._impl.get_last_mask()

    def get_last_cc_mask(self):
        return self._impl.get_last_cc_mask()

    def get_last_bb_mask(self):
        return self._impl.get_last_bb_mask()
