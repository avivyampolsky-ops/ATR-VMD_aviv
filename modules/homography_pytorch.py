import cv2
import numpy as np
from time import perf_counter

try:
    import torch
    import kornia
except ImportError:
    torch = None
    kornia = None

from .features import FeatureExtractorType, KorniaSiftFeatureExtractor
from .matchers import MatcherType, KorniaMatcher

class HomographyTranslationPyTorch:
    def __init__(self, reference,
                 matcher=MatcherType.KORNIA_SNN,
                 feature_extractor=FeatureExtractorType.KORNIA_SIFT,
                 downscale_factor=1.0,
                 knn_ratio=0.8,
                 ransac_reproj_threshold=5.0,
                 min_inliers=0,
                 reference_window_frames=1,
                 enable_timing=True,
                 device=None):
        if torch is None or kornia is None:
            raise RuntimeError("HomographyTranslationPyTorch requires 'torch' and 'kornia'.")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._downscale_factor = float(downscale_factor) if downscale_factor else 1.0
        self._knn_ratio = float(knn_ratio)
        self._ransac_threshold = float(ransac_reproj_threshold)
        self._min_inliers = int(min_inliers)
        self._reference_window_frames = max(1, int(reference_window_frames))
        self._timing_enabled = bool(enable_timing)

        # Initialize Kornia Components
        self.feature_extractor = KorniaSiftFeatureExtractor(nfeatures=2000, device=self.device) # Using default from class usually
        self.matcher = KorniaMatcher(ratio=self._knn_ratio)

        # State
        self._frame_count = 0
        self._prev_frame_tensor = None
        self._reference_events = []
        self._last_reference_reason = None
        self._last_fail_reason = None
        self._reg_stats = {
            "first_try": {"calls": 0, "total": 0.0},
            "fallback": {"calls": 0, "total": 0.0},
            "re_registration_calls": 0,
        }
        self._timing = {
            "calls": 0,
            "registration": 0.0,
            "upload_gray": 0.0,
            "features": 0.0,
            "match": 0.0,
            "homography": 0.0,
            "warp": 0.0,
        }

        # Set Reference
        self.set_reference(reference, record_event=False, reason="init")

    def set_reference(self, reference, record_event=True, reason="manual"):
        # Reference is expected to be HxW or HxWx3 numpy array
        self.h, self.w = reference.shape[:2]
        self.margin = int(min(self.h, self.w) * 0.01)

        # Convert to Tensor (B, C, H, W)
        ref_tensor = self._to_tensor(reference)

        # Extract Features
        # lafs: (1, N, 2, 3), descs: (1, N, 128)
        self.lafs_ref, self.descs_ref = self.feature_extractor.get_keypoints_and_descriptors(ref_tensor)
        self.ref_tensor = ref_tensor

        if record_event:
            self._reference_events.append(self._frame_count)
            self._last_reference_reason = reason

    def _to_tensor(self, img_np):
        t_img = kornia.image_to_tensor(img_np, keepdim=False).float().to(self.device)
        if t_img.max() > 1.0:
            t_img /= 255.0
        # If color, keep it for warping later, but features will convert to gray internally
        return t_img

    def register_frame(self, frame):
        # Enforce no_grad for the entire registration block to speed up inference
        if torch is not None:
             with torch.no_grad():
                return self._register_frame_impl(frame)
        return self._register_frame_impl(frame)

    def _register_frame_impl(self, frame):
        self._frame_count += 1

        # Timing start
        start = perf_counter() if self._timing_enabled else 0.0

        # 1. Upload & Preprocess
        frame_tensor = self._to_tensor(frame) # (1, C, H, W)

        if self._timing_enabled:
            t_upload = perf_counter()
            self._timing["upload_gray"] += t_upload - start

        # 2. Extract Features
        lafs_frame, descs_frame = self.feature_extractor.get_keypoints_and_descriptors(frame_tensor)

        if self._timing_enabled:
            t_feats = perf_counter()
            self._timing["features"] += t_feats - t_upload

        if descs_frame.shape[1] < 4:
            self._last_fail_reason = "descriptors"
            return self._crop_and_return(frame_tensor)

        # 3. Match
        # indices: (num_matches, 2), dists: (num_matches)
        indices, dists = self.matcher.match(descs_frame, self.descs_ref)

        if self._timing_enabled:
            t_match = perf_counter()
            self._timing["match"] += t_match - t_feats

        if indices.shape[0] < 4:
            self._last_fail_reason = "match"
            return self._crop_and_return(frame_tensor)

        # 4. Homography (RANSAC)
        # Get points from LAFs using indices
        # lafs are (1, N, 2, 3). We want the center points (last column of the 2x3 affine matrix)
        # points: (1, N, 2) -> (x, y)
        kps_frame = kornia.feature.get_laf_center(lafs_frame)
        kps_ref = kornia.feature.get_laf_center(self.lafs_ref)

        # Gather matched points
        # match_snn returns indices as pairs of (idx_in_desc1, idx_in_desc2)
        # descs_frame (1st arg) -> kps_frame -> indices[:, 0]
        # descs_ref (2nd arg)   -> kps_ref   -> indices[:, 1]

        src_pts = kps_frame[0, indices[:, 0]] # (M, 2)
        dst_pts = kps_ref[0, indices[:, 1]]   # (M, 2)

        # Kornia RANSAC
        try:
            H_res = kornia.geometry.ransac.find_homography_ransac(
                src_pts.unsqueeze(0),
                dst_pts.unsqueeze(0),
                ransac_thres=self._ransac_threshold
            )
            # H_res is (H, inliers)
            H = H_res[0]
            inliers = H_res[1]
        except Exception:
            self._last_fail_reason = "ransac"
            return self._crop_and_return(frame_tensor)

        inlier_count = int(inliers.sum().item())
        if self._min_inliers > 0 and inlier_count < self._min_inliers:
            self._last_fail_reason = "inliers"
            return self._crop_and_return(frame_tensor)

        if self._timing_enabled:
            t_h = perf_counter()
            self._timing["homography"] += t_h - t_match

        # 5. Warp
        # warp_perspective expects (B, C, H, W)
        warped_tensor = kornia.geometry.transform.warp_perspective(
            frame_tensor,
            H,
            dsize=(self.h, self.w)
        )

        if self._timing_enabled:
            t_warp = perf_counter()
            self._timing["warp"] += t_warp - t_h
            self._timing["calls"] += 1
            self._timing["registration"] += t_warp - start

        # Update reference logic (windowed)
        if self._should_update_reference():
             # We can reuse the features we just computed
             self.lafs_ref = lafs_frame
             self.descs_ref = descs_frame
             self.ref_tensor = frame_tensor
             self._reference_events.append(self._frame_count + 1)
             self._last_reference_reason = "window"

        # Return result (convert back to numpy for compatibility with rest of pipeline)
        return self._crop_and_return(warped_tensor, to_numpy=True)

    def _crop_and_return(self, tensor, to_numpy=True):
        # Crop margins
        if self.margin > 0:
            cropped = tensor[..., self.margin:-self.margin, self.margin:-self.margin]
        else:
            cropped = tensor

        if to_numpy:
            # (1, C, H, W) -> (H, W, C) or (H, W)
            out_np = kornia.tensor_to_image(cropped.byte() if isinstance(cropped, torch.ByteTensor) else (cropped * 255.0).byte())
            return out_np
        return cropped

    def _should_update_reference(self):
        return (self._frame_count % self._reference_window_frames) == 0

    def get_shape(self):
        return (self.h - self.margin * 2, self.w - self.margin * 2)

    def get_reference_events(self):
        return np.array(self._reference_events, dtype=np.int32)

    def reset_timing(self):
        if not self._timing_enabled:
            return
        for k in self._timing:
            self._timing[k] = 0

    def timing_summary(self, reset=False):
        if not self._timing_enabled:
            return {"enabled": False}
        calls = self._timing["calls"] or 1
        summary = {
            "enabled": True,
            "calls": self._timing["calls"],
            "registration": (self._timing["registration"] / calls) * 1000.0,
            "upload_gray": (self._timing["upload_gray"] / calls) * 1000.0,
            "features": (self._timing["features"] / calls) * 1000.0,
            "match": (self._timing["match"] / calls) * 1000.0,
            "homography": (self._timing["homography"] / calls) * 1000.0,
            "warp": (self._timing["warp"] / calls) * 1000.0,
        }
        if reset:
            self.reset_timing()
        return summary

    def get_last_homography_debug(self):
        return {
            "fail_reason": self._last_fail_reason,
            "reference_reason": self._last_reference_reason
        }

    def get_last_shift(self):
        return (0.0, 0.0)

    def register_frame_with_shift(self, frame):
        return self.register_frame(frame), 0.0, 0.0
