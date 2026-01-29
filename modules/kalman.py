from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Optional


class KalmanIoUTracker:
    def __init__(
        self,
        iou_thresh=0.05,
        max_lost=3,
        min_move=2,
        ema_alpha=0.3,
        dist_gate_scale=2.0,
        easy_iou_thresh=0.6
    ):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.min_move = min_move
        self.ema_alpha = ema_alpha
        self.dist_gate_scale = dist_gate_scale
        self.easy_iou_thresh = easy_iou_thresh

        self.current_id = 0
        self.tracks: Dict[int, KalmanIoUTracker.Entity] = {}

    class Entity:
        def __init__(self, entity_id: int, bbox: np.ndarray, kf: cv2.KalmanFilter,
                     age: int = 1, lost: int = 0, moving: bool = False, ema_area: float = 0.0):
            ''' Represents a tracked entity with Kalman filter state '''
            self.id = entity_id
            self.bbox = bbox
            self.kf = kf
            self.age = age
            self.lost = lost
            self.moving = moving
            bbox_area = float(bbox[2] * bbox[3])
            self.ema_area = bbox_area if ema_area == 0.0 else ema_area

            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    @staticmethod
    def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def iou_vectorized(boxA: np.ndarray, obs_x, obs_y, obs_w, obs_h, obs_area) -> np.ndarray:
        xA = np.maximum(boxA[0], obs_x)
        yA = np.maximum(boxA[1], obs_y)
        xB = np.minimum(boxA[0] + boxA[2], obs_x + obs_w)
        yB = np.minimum(boxA[1] + boxA[3], obs_y + obs_h)
        inter_w = np.maximum(0.0, xB - xA)
        inter_h = np.maximum(0.0, yB - yA)
        inter = inter_w * inter_h
        union = (boxA[2] * boxA[3]) + obs_area - inter
        return np.where(union > 0.0, inter / union, 0.0)

    @staticmethod
    def create_kalman():
        kf = cv2.KalmanFilter(4, 2)

        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def _apply_match(self, entity: Entity, bbox: np.ndarray) -> None:
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        entity.kf.correct(np.array([[cx], [cy]], np.float32))
        px = entity.bbox[0] + entity.bbox[2] // 2
        py = entity.bbox[1] + entity.bbox[3] // 2
        dist = np.hypot(cx - px, cy - py)

        ema_area = (
            self.ema_alpha * (bbox[2] * bbox[3]) # less weight to current observation's area
            + (1 - self.ema_alpha) * entity.ema_area # more weight to previous EMA area
        )

        entity.bbox = bbox
        entity.age += 1
        entity.lost = 0
        entity.moving = dist >= self.min_move
        entity.ema_area = ema_area

    def _build_iou_matrix(self, track_items, observations: np.ndarray) -> Optional[np.ndarray]:
        num_tracks = len(track_items)
        num_obs = observations.shape[0]
        if not num_tracks or not num_obs:
            return None

        # Building iou_matrix
        iou_matrix = np.full((num_tracks, num_obs), -1.0, dtype=np.float32)
        obs_x = observations[:, 0].astype(np.float32)
        obs_y = observations[:, 1].astype(np.float32)
        obs_w = observations[:, 2].astype(np.float32)
        obs_h = observations[:, 3].astype(np.float32)
        obs_area = obs_w * obs_h
        obs_cx = obs_x + (obs_w * 0.5)
        obs_cy = obs_y + (obs_h * 0.5)

        for t_idx, (_, entity) in enumerate(track_items):
            ema_area = entity.ema_area
            pred = entity.kf.statePre
            pred_cx, pred_cy = float(pred[0, 0]), float(pred[1, 0])
            dist_gate = self.dist_gate_scale * max(entity.bbox[2], entity.bbox[3])

            # ---------- EMA AREA GATE ----------
            area_mask = (obs_area >= 0.5 * ema_area) & (obs_area <= 2.0 * ema_area)
            # ---------- DISTANCE GATE ----------
            dist_mask = np.hypot(obs_cx - pred_cx, obs_cy - pred_cy) <= dist_gate
            valid = area_mask & dist_mask
            if not np.any(valid):
                continue

            iou = self.iou_vectorized(entity.bbox, obs_x, obs_y, obs_w, obs_h, obs_area)
            iou_matrix[t_idx, valid] = iou[valid]

        return iou_matrix

    def update(self, observations: np.ndarray):
        updated_tracks: Dict[int, KalmanIoUTracker.Entity] = {}
        used_mask = np.zeros(observations.shape[0], dtype=bool)

        # ---------- Predict ----------
        for entity in self.tracks.values():
            entity.kf.predict()

        # ---------- Match ----------
        # Stage 0: build IoU matrix with gating (area + distance)
        track_items = list(self.tracks.items())
        num_tracks = len(track_items)
        iou_matrix = self._build_iou_matrix(track_items, observations)

        unmatched_tracks = []
        if iou_matrix is not None:
            # Stage 1: easy greedy matches
            for t_idx, (track_id, entity) in enumerate(track_items):
                row = iou_matrix[t_idx]
                if used_mask.any():
                    masked = np.where(used_mask, -1.0, row)
                else:
                    masked = row
                best_idx = int(masked.argmax())
                best_iou = masked[best_idx]

                if best_iou >= self.easy_iou_thresh:
                    bbox = observations[best_idx]
                    self._apply_match(entity, bbox)
                    updated_tracks[track_id] = entity
                    used_mask[best_idx] = True
                else:
                    unmatched_tracks.append(t_idx)
        else:
            unmatched_tracks = list(range(num_tracks))

        # Stage 2: remaining greedy matches
        for t_idx in unmatched_tracks:
            track_id, entity = track_items[t_idx]
            best_iou = -1.0
            best_idx = -1
            if iou_matrix is not None:
                row = iou_matrix[t_idx]
                if used_mask.any():
                    masked = np.where(used_mask, -1.0, row)
                else:
                    masked = row
                best_idx = int(masked.argmax())
                best_iou = masked[best_idx]

            if best_iou >= self.iou_thresh:
                bbox = observations[best_idx]
                self._apply_match(entity, bbox)
                updated_tracks[track_id] = entity
                used_mask[best_idx] = True
            else:
                entity.lost += 1
                if entity.lost <= self.max_lost:
                    pred = entity.kf.statePre # predicted position
                    cx, cy = int(pred[0, 0]), int(pred[1, 0])
                    w, h = entity.bbox[2], entity.bbox[3]
                    pred_bbox = np.array([cx - w//2, cy - h//2, w, h], dtype=entity.bbox.dtype)
                    entity.bbox = pred_bbox
                    updated_tracks[track_id] = entity

        # ---------- New tracks ----------
        for i, bbox in enumerate(observations):
            if not used_mask[i]:

                updated_tracks[self.current_id] = self.Entity(
                    self.current_id,
                    bbox,
                    self.create_kalman()
                )
                
                self.current_id += 1

        self.tracks = updated_tracks
        return self.tracks

    def apply_shift(self, dx: float, dy: float) -> None:
        # Compensate tracker state for registration jumps so IoU matching stays stable.
        if dx == 0.0 and dy == 0.0:
            return
        for entity in self.tracks.values():
            entity.bbox = entity.bbox.astype(np.float32, copy=False)
            entity.bbox[0] += dx
            entity.bbox[1] += dy
            entity.kf.statePost[0, 0] += dx
            entity.kf.statePost[1, 0] += dy
            entity.kf.statePre[0, 0] += dx
            entity.kf.statePre[1, 0] += dy

    def reset(self) -> None:
        self.current_id = 0
        self.tracks = {}

    def get_tracks(self):
        return self.tracks
