from .translations import (
    DirectTranslation,
    HomographyTranslation,
    HomographyTranslationCPU,
    HomographyTranslationCPUCppWrapper,
    HomographyTranslationGPUCppWrapper,
)
from .features import FeatureExtractorType
from .matchers import MatcherType


class Registrator:
    def __init__(
        self,
        reference,
        matcher=MatcherType.BF,
        feature_extractor=FeatureExtractorType.ORB,
        registration_mode="homography",
        translation_method="phase",
        phase_use_cached_fft=True,
        phase_response_threshold=0.1,
        knn_ratio=0.75,
        ransac_reproj_threshold=5.0,
        min_inliers=0,
        downscale_factor=1.0,
        reference_window_frames=1,
        use_cuda=False,
        use_cpp=True,
        enable_timing=True,
    ):
        mode = str(registration_mode).lower()
        if mode == "translation":
            self._impl = DirectTranslation(
                reference,
                method=translation_method,
                downscale_factor=downscale_factor,
                enable_timing=enable_timing,
                phase_use_cached_fft=phase_use_cached_fft,
                phase_response_threshold=phase_response_threshold,
                reference_window_frames=reference_window_frames,
                use_cuda=use_cuda,
                use_cpp=use_cpp,
            )
        else:
            if use_cpp and use_cuda:
                self._impl = HomographyTranslationGPUCppWrapper(
                    reference,
                    matcher=matcher,
                    feature_extractor=feature_extractor,
                    downscale_factor=downscale_factor,
                    knn_ratio=knn_ratio,
                    ransac_reproj_threshold=ransac_reproj_threshold,
                    min_inliers=min_inliers,
                    reference_window_frames=reference_window_frames,
                    enable_timing=enable_timing,
                )
            elif use_cpp and not use_cuda:
                self._impl = HomographyTranslationCPUCppWrapper(
                    reference,
                    matcher=matcher,
                    feature_extractor=feature_extractor,
                    downscale_factor=downscale_factor,
                    knn_ratio=knn_ratio,
                    ransac_reproj_threshold=ransac_reproj_threshold,
                    min_inliers=min_inliers,
                    reference_window_frames=reference_window_frames,
                    enable_timing=enable_timing,
                )
            else:
                self._impl = HomographyTranslationCPU(
                    reference,
                    matcher=matcher,
                    feature_extractor=feature_extractor,
                    downscale_factor=downscale_factor,
                    knn_ratio=knn_ratio,
                    ransac_reproj_threshold=ransac_reproj_threshold,
                    min_inliers=min_inliers,
                    enable_timing=enable_timing,
                    reference_window_frames=reference_window_frames,
                )

    def set_reference(self, reference):
        return self._impl.set_reference(reference)

    def get_shape(self):
        return self._impl.get_shape()

    def reset_timing(self):
        return self._impl.reset_timing()

    def timing_summary(self, reset=False):
        return self._impl.timing_summary(reset=reset)

    def register_frame(self, frame):
        return self._impl.register_frame(frame)

    def get_last_homography_debug(self):
        getter = getattr(self._impl, "get_last_homography_debug", None)
        if getter is None:
            return None
        return getter()

    def get_reference_events(self):
        getter = getattr(self._impl, "get_reference_events", None)
        if getter is None:
            return None
        return getter()

    def register_frame_with_shift(self, frame):
        method = getattr(self._impl, "register_frame_with_shift", None)
        if method is None:
            reg = self._impl.register_frame(frame)
            return reg, 0.0, 0.0
        return method(frame)

    def get_last_shift(self):
        getter = getattr(self._impl, "get_last_shift", None)
        if getter is None:
            return (0.0, 0.0)
        return getter()
