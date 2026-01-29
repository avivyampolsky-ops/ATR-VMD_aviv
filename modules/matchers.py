import cv2
from enum import Enum

try:
    import torch
    import kornia
except ImportError:
    torch = None
    kornia = None

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
    KORNIA_SNN = "kornia_snn"

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

class KorniaMatcher:
    def __init__(self, ratio=0.8, dist_type="l2"):
        if torch is None or kornia is None:
            raise RuntimeError("KorniaMatcher requires 'torch' and 'kornia'.")
        self.ratio = ratio
        # kornia.feature.match_snn (Symmetric Nearest Neighbor) or match_mnn (Mutual Nearest Neighbor)
        # SIFT descriptors are L2 normalized, so "l2" distance is appropriate.
        self.dist_type = dist_type
        self.matcher = kornia.feature.DescriptorMatcher(match_mode='snn', th=ratio)

    def match(self, des1, des2):
        # kornia.feature.match_snn expects (N, D) tensors for direct matching.
        # Although DescriptorMatcher can handle batches, the error log shows match_snn
        # complaining about shape when fed (1, N, D).
        # We strip the batch dimension if present (assuming B=1 for this pipeline).

        if des1.ndim == 3 and des1.shape[0] == 1:
            des1 = des1.squeeze(0)
        if des2.ndim == 3 and des2.shape[0] == 1:
            des2 = des2.squeeze(0)

        # If we have legitimate batches > 1, this logic would need update,
        # but the current pipeline is strictly single-frame.

        # kornia.feature.DescriptorMatcher returns (dists, indices) or just indices?
        # The traceback says: dists, indices = self.matcher(des1, des2)
        # And match_snn returns (dists, idxs)
        # With 2D inputs (N, D), DescriptorMatcher likely returns 2D outputs or similar.

        dists, indices = self.matcher(des1, des2)

        # If input was 2D, output is (num_matches, 2)
        # indices contains pairs of (idx1, idx2)

        return indices, dists
