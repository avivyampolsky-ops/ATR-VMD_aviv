import cv2
from enum import Enum
from .matchers import _LSH_FLANN_PARAMS, _KDTREE_FLANN_PARAMS

class FeatureExtractorType(Enum):
    ORB = "orb"
    FAST_BRIEF = "fast_brief"
    AKAZE = "akaze"
    BRISK = "brisk"
    SIFT = "sift"

class FeatureExtractorBase:
    norm_type = cv2.NORM_HAMMING
    flann_index_params = _LSH_FLANN_PARAMS

    def get_keypoints_and_descriptors(self, image):
        raise NotImplementedError

class OrbFeatureExtractor(FeatureExtractorBase):
    norm_type = cv2.NORM_HAMMING
    flann_index_params = _LSH_FLANN_PARAMS

    def __init__(self):
        self.feature_extractor = cv2.ORB_create(500)

    def get_keypoints_and_descriptors(self, image):
        return self.feature_extractor.detectAndCompute(image, mask=None)

class FastBriefFeatureExtractor(FeatureExtractorBase):
    norm_type = cv2.NORM_HAMMING
    flann_index_params = _LSH_FLANN_PARAMS

    def __init__(self, nfeatures=500, fast_threshold=20, nonmax_suppression=True, brief_bytes=32):
        if not hasattr(cv2, "xfeatures2d"):
            raise RuntimeError("cv2.xfeatures2d is not available; BRIEF requires opencv-contrib.")
        self.nfeatures = nfeatures
        self.detector = cv2.FastFeatureDetector_create(
            threshold=fast_threshold,
            nonmaxSuppression=nonmax_suppression
        )
        self.extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=brief_bytes)

    def get_keypoints_and_descriptors(self, image):
        kp = self.detector.detect(image, None)
        if self.nfeatures is not None and len(kp) > self.nfeatures:
            kp = sorted(kp, key=lambda k: k.response, reverse=True)[:self.nfeatures]
        kp, desc = self.extractor.compute(image, kp)
        return kp, desc

class AkazeFeatureExtractor(FeatureExtractorBase):
    norm_type = cv2.NORM_HAMMING
    flann_index_params = _LSH_FLANN_PARAMS

    def __init__(
        self,
        nfeatures=500,
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size=0,
        descriptor_channels=3,
        threshold=0.001,
        nOctaves=4,
        nOctaveLayers=4,
        diffusivity=cv2.KAZE_DIFF_PM_G2,
    ):
        self.nfeatures = nfeatures
        self.feature_extractor = cv2.AKAZE_create(
            descriptor_type=descriptor_type,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=nOctaves,
            nOctaveLayers=nOctaveLayers,
            diffusivity=diffusivity,
        )

    def get_keypoints_and_descriptors(self, image):
        kp, desc = self.feature_extractor.detectAndCompute(image, mask=None)
        if kp is not None and self.nfeatures is not None and len(kp) > self.nfeatures:
            kp = sorted(kp, key=lambda k: k.response, reverse=True)[:self.nfeatures]
            kp, desc = self.feature_extractor.compute(image, kp)
        return kp, desc

class BriskFeatureExtractor(FeatureExtractorBase):
    norm_type = cv2.NORM_HAMMING
    flann_index_params = _LSH_FLANN_PARAMS

    def __init__(self, nfeatures=500, thresh=30, octaves=3, patternScale=1.0):
        self.nfeatures = nfeatures
        self.feature_extractor = cv2.BRISK_create(thresh=thresh, octaves=octaves, patternScale=patternScale)

    def get_keypoints_and_descriptors(self, image):
        kp, desc = self.feature_extractor.detectAndCompute(image, mask=None)
        if kp is not None and self.nfeatures is not None and len(kp) > self.nfeatures:
            kp = sorted(kp, key=lambda k: k.response, reverse=True)[:self.nfeatures]
            kp, desc = self.feature_extractor.compute(image, kp)
        return kp, desc

class SiftFeatureExtractor(FeatureExtractorBase):
    norm_type = cv2.NORM_L2
    flann_index_params = _KDTREE_FLANN_PARAMS

    def __init__(
        self,
        nfeatures=500,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    ):
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("cv2.SIFT_create is not available in this OpenCV build.")
        self.feature_extractor = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )

    def get_keypoints_and_descriptors(self, image):
        return self.feature_extractor.detectAndCompute(image, mask=None)
