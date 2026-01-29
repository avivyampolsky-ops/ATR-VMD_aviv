import cv2
from enum import Enum

_LSH_FLANN_PARAMS = dict(
    algorithm=6,
    table_number=6,
    key_size=12,
    multi_probe_level=1,
)
_KDTREE_FLANN_PARAMS = dict(
    algorithm=1,
    trees=5,
)

class MatcherType(Enum):
    BF = "bf"
    KNN = "knn"
    FLANN = "flann"

class BFMatcher:
    def __init__(self, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=True)

    def match(self, des1, des2):
        matches = self.matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)

class KNNMatcher:
    def __init__(self, ratio=0.75, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type)
        self.ratio = ratio

    def match(self, des1, des2):
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        try:
            for m, n in knn_matches:
                if m.distance < self.ratio * n.distance:
                    good.append(m)
            return sorted(good, key=lambda x: x.distance)
        except ValueError:
            return []

class FlannMatcher:
    def __init__(self, ratio=0.75, index_params=None, checks=50):
        if index_params is None:
            index_params = _LSH_FLANN_PARAMS
        search_params = dict(checks=checks)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.ratio = ratio

    def match(self, des1, des2):
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return sorted(good, key=lambda x: x.distance)
