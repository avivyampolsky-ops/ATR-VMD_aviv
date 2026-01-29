import os
import re
import cv2
import numpy as np


# Bundle input source state (cap/images/ref/writer) to keep main loop clean.
class DataStreamer:
    def __init__(self, data_path, fps_override, config):
        # Keep input options and initialize the video/image source immediately.
        self.data_path = data_path
        self.start_frame = max(0, int(config.get("debug.start_frame", 0)))
        end_frame = int(config.get("debug.end_frame", -1))
        self.end_frame = None if end_frame < 0 else end_frame
        self.input_is_gray = config.get("general.input_is_gray", False)
        self.fps_override = fps_override
        self.cap = None
        self.image_paths = None
        self.total_frames = None
        self.resolved_input_is_gray = None
        self.fps = None
        self.ref = None
        self._init_source()

    def _init_source(self):
        # Resolve input source (image folder vs video) and load the reference frame.
        if os.path.isdir(self.data_path):
            self.image_paths = self._list_image_files(self.data_path)
            if not self.image_paths:
                print(f"No images found in {self.data_path}")
                return
            self.total_frames = len(self.image_paths)
            if self.start_frame >= self.total_frames:
                print(f"start_frame {self.start_frame} exceeds total frames {self.total_frames}")
                return
            if self.end_frame is not None:
                self.end_frame = min(self.end_frame, self.total_frames - 1)
            self.ref = self._read_image(self.image_paths[self.start_frame], self.input_is_gray)
            if self.ref is None:
                print(f"Failed to read image: {self.image_paths[self.start_frame]}")
                return
            self.resolved_input_is_gray = (self.ref.ndim == 2)
            from modules import video_reader
            video_reader.init_from_frame(self.ref, input_is_gray=self.input_is_gray)
            self.fps = self.fps_override
            return

        self.cap = cv2.VideoCapture(self.data_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if self.start_frame and self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        if self.total_frames is not None:
            if self.start_frame >= self.total_frames:
                print(f"start_frame {self.start_frame} exceeds total frames {self.total_frames}")
                self.cap.release()
                self.cap = None
                return
            if self.end_frame is not None:
                self.end_frame = min(self.end_frame, self.total_frames - 1)
        from modules import video_reader
        ret, self.ref = video_reader.init_reader(self.cap, input_is_gray=self.input_is_gray)
        if not ret:
            print("Failed to load video")
            self.cap.release()
            self.cap = None
            return
        self.resolved_input_is_gray = video_reader.input_is_gray()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read_frame(self, frame_idx, input_is_gray):
        # Read the next frame from disk or video capture.
        if self.image_paths is not None:
            frame = self._read_image(self.image_paths[frame_idx], input_is_gray)
            return frame is not None, frame
        from modules import video_reader
        return video_reader.read_frame(self.cap)

    def get_frame(self, frame_idx):
        # Read a frame and stop when bounds are exceeded or reading fails.
        if self.image_paths is not None:
            if self.end_frame is not None and frame_idx > self.end_frame:
                return None
            if frame_idx >= len(self.image_paths):
                return None
            ret, frame = self.read_frame(frame_idx, self.input_is_gray)
            if not ret:
                print(f"Failed to read image: {self.image_paths[frame_idx]}")
                return None
            return frame
        ret, frame = self.read_frame(frame_idx, self.input_is_gray)
        if not ret:
            return None
        if self.end_frame is not None and frame_idx > self.end_frame:
            return None
        if self.total_frames is not None and frame_idx >= self.total_frames:
            return None
        return frame

    def get_grame(self, frame_idx):
        # Backwards-compatible alias for get_frame.
        return self.get_frame(frame_idx)

    def _list_image_files(self, dir_path):
        # List image files in natural sort order for stable frame indexing.
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = []
        for name in os.listdir(dir_path):
            if name.lower().endswith(exts):
                files.append(os.path.join(dir_path, name))

        def _natural_key(path):
            base = os.path.basename(path)
            parts = re.split(r"(\d+)", base)
            key = []
            for part in parts:
                if part.isdigit():
                    key.append(int(part))
                else:
                    key.append(part.lower())
            return key

        files.sort(key=_natural_key)
        return files

    def _read_image(self, path, input_is_gray):
        # Read a single image with explicit grayscale or color flag.
        flag = cv2.IMREAD_GRAYSCALE if input_is_gray else cv2.IMREAD_COLOR
        return cv2.imread(path, flag)

    def close(self):
        # Release any open capture handle.
        if self.cap is not None:
            self.cap.release()


class DebugUtils:
    def __init__(self, config, data_path, out_dir, stream_fps, stream_is_gray):
        # Initialize optional debug writers that depend on runtime settings.
        self.registration_mode = config.get("registration.mode", "translation")
        self.start_frame = max(0, int(config.get("debug.start_frame", 0)))
        end_frame = int(config.get("debug.end_frame", -1))
        self.end_frame = None if end_frame < 0 else end_frame
        self.total_frames = None
        self.video_base = os.path.splitext(os.path.basename(os.path.normpath(data_path)))[0]
        self.out_dir = os.path.join(out_dir, f"{self.video_base}")
        self.draw_output = config.get("debug.draw_output", True)
        self.debug_mode = config.get("debug.debug_mode", True)
        self.enable_timing = config.get("debug.enable_timing", True)
        self.track_min_age = int(config.get("tracker.kalman.track_min_age", 3))
        self.frames_dir = os.path.join(self.out_dir, f"frames_{self.video_base}")
        self.mog2_dir = os.path.join(self.out_dir, f"mog2_masks_{self.video_base}")
        self.bb_dir = os.path.join(self.out_dir, f"bb_masks_{self.video_base}")
        if self.debug_mode:
            os.makedirs(self.out_dir, exist_ok=True)
        if self.debug_mode and self.draw_output:
            os.makedirs(self.frames_dir, exist_ok=True)
            os.makedirs(self.mog2_dir, exist_ok=True)
            os.makedirs(self.bb_dir, exist_ok=True)
        self.log_path = ""
        if self.debug_mode:
            self.log_path = os.path.join(self.out_dir, f"reanchor_log_{self.video_base}.txt")
            config.set_runtime("debug.reanchor_log_path", self.log_path)
        self._writer = None
        self._stream_fps = stream_fps
        self._stream_is_gray = stream_is_gray
        self._homography_log = None
        self.loop_calls = 0
        if self.debug_mode and self.registration_mode == "homography":
            min_inliers = config.get("registration.homography.min_inliers", 0)
            self._homography_log = DebugUtils.HomographyLog(self.out_dir, self.video_base, min_inliers)

    def set_stream_info(self, data_streamer):
        # Sync bounds derived from the input source (e.g., after clamping end_frame).
        self.total_frames = data_streamer.total_frames
        self.start_frame = data_streamer.start_frame
        self.end_frame = data_streamer.end_frame

    def write_homography_log(self, frame_idx, homography_debug):
        # Persist homography diagnostics when the log is enabled.
        if self.debug_mode and self._homography_log is not None:
            self._homography_log.write(frame_idx, homography_debug)

    def close_homography_log(self):
        # Close the homography log if it was created.
        if self._homography_log is not None:
            self._homography_log.close()
            self._homography_log = None

    def close_writer(self):
        # Close the debug video writer if it was created.
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def close(self):
        # Close any debug outputs owned by this helper.
        if not self.debug_mode:
            return
        self.close_writer()
        self.close_homography_log()

    # List image files in natural sort order for stable frame indexing.
    def _list_image_files(self, dir_path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = []
        for name in os.listdir(dir_path):
            if name.lower().endswith(exts):
                files.append(os.path.join(dir_path, name))

        def _natural_key(path):
            base = os.path.basename(path)
            parts = re.split(r"(\d+)", base)
            key = []
            for part in parts:
                if part.isdigit():
                    key.append(int(part))
                else:
                    key.append(part.lower())
            return key

        files.sort(key=_natural_key)
        return files

    # Read a single image with explicit grayscale or color flag.
    def _read_image(self, path, input_is_gray):
        flag = cv2.IMREAD_GRAYSCALE if input_is_gray else cv2.IMREAD_COLOR
        return cv2.imread(path, flag)

    # Build a side-by-side video from saved frames and bbox masks for debugging.
    def build_frame_vs_mask_video(self, out_dir, video_base, fps=30.0):
        if not self.debug_mode:
            return
        frames_dir = os.path.join(out_dir, f"frames_{video_base}")
        bb_dir = os.path.join(out_dir, f"bb_masks_{video_base}")
        if not os.path.isdir(frames_dir) or not os.path.isdir(bb_dir):
            print("Missing frames or bb_masks folder.")
            return

        frame_paths = self._list_image_files(frames_dir)
        if not frame_paths:
            print("No frames found.")
            return

        first = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            print("Failed to read first frame.")
            return
        h, w = first.shape[:2]
        out_path = os.path.join(out_dir, f"side_by_side_{video_base}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w * 2, h),
        )

        frame_map = {}
        mask_map = {}
        for path in frame_paths:
            name = os.path.basename(path)
            match = re.search(r"frame_(\d+)", name)
            if match:
                frame_map[int(match.group(1))] = path
        for path in self._list_image_files(bb_dir):
            name = os.path.basename(path)
            match = re.search(r"mask_(\d+)", name)
            if match:
                mask_map[int(match.group(1))] = path

        if not frame_map or not mask_map:
            print("Missing frame or mask indices.")
            writer.release()
            return

        sorted_frames = sorted(frame_map.keys())
        sorted_masks = sorted(mask_map.keys())
        offset_candidates = []
        for f_idx in sorted_frames:
            if f_idx in mask_map:
                offset_candidates.append(f_idx - f_idx)
                break
        if not offset_candidates:
            offset_candidates.append(sorted_frames[0] - sorted_masks[0])
        offset = offset_candidates[0]

        for f_idx in sorted_frames:
            frame_path = frame_map.get(f_idx)
            mask_path = mask_map.get(f_idx - offset)
            if frame_path is None or mask_path is None:
                continue
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if frame is None or mask is None:
                continue
            if frame.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(
                    mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            combined = cv2.hconcat([frame, mask])
            writer.write(combined)

        writer.release()

    # Render a timeline image that marks reference-reset events.
    def write_reference_events_graph(self, ref_events):
        if not self.debug_mode:
            return
        if ref_events is None or not len(ref_events) or not self.loop_calls:
            return
        width = min(2000, self.loop_calls)
        height = 200
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        span = max(1, self.loop_calls - 1)
        for ev in ref_events:
            frame_global = self.start_frame + int(ev)
            if self.end_frame is not None and frame_global > self.end_frame:
                continue
            rel = frame_global - self.start_frame
            if rel < 0 or rel >= self.loop_calls:
                continue
            x = int(round((rel / span) * (width - 1)))
            cv2.line(graph, (x, 0), (x, height - 1), (0, 0, 255), 1)
        cv2.imwrite(
            os.path.join(self.out_dir, f"reference_events_{self.video_base}.png"),
            graph,
        )

    # Add on-frame text for registration state, homography stats, and shifts.
    def annotate_registration_debug(self, frame, frame_idx, registration_mode, ref_events, homography_debug, shift):
        if not self.debug_mode:
            return frame
        if frame is None:
            return frame
        out = frame.copy()
        if ref_events is not None and frame_idx in ref_events:
            label = "REANCHOR"
            if homography_debug is not None:
                ref_reason = homography_debug.get("reference_reason")
                if ref_reason:
                    label = f"REANCHOR {ref_reason}"
            color = 255 if out.ndim == 2 else (0, 0, 255)
            cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if registration_mode == "homography" and homography_debug is not None:
            reason = homography_debug.get("fail_reason") or "ok"
            matches = homography_debug.get("match_count", 0)
            inliers = homography_debug.get("inlier_count", 0)
            overlap = homography_debug.get("overlap_ratio", 0.0)
            color = 255 if out.ndim == 2 else (0, 255, 255)
            cv2.putText(
                out,
                f"HOMO {reason} m={matches} i={inliers} ov={overlap:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        elif shift is not None:
            dx, dy = shift
            color = 255 if out.ndim == 2 else (0, 255, 255)
            cv2.putText(
                out,
                f"SHIFT dx={dx:.2f} dy={dy:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return out

    def draw_tracks(self, frame, tracks):
        # Draw tracker boxes and IDs on top of the provided frame.
        if not self.debug_mode or frame is None or tracks is None:
            return frame
        out = frame.copy()
        color = 255 if out.ndim == 2 else (0, 255, 0)
        for track_id, entity in tracks.items():
            if entity.age < self.track_min_age or not entity.moving:
                continue
            x, y, w, h = entity.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                out,
                f"ID {track_id} EMA:{int(entity.ema_area)}",
                (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return out

    # Persist frame/mask snapshots and bbox overlays for offline inspection.
    def write_debug_images(self, out, frame_idx, frames_dir, mog2_dir, bb_dir, tracker):
        if not self.debug_mode:
            return
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg"), out)
        mask = tracker.get_last_mask()
        if mask is not None and mask.size:
            cv2.imwrite(os.path.join(mog2_dir, f"mask_{frame_idx:06d}.jpg"), mask)
            detections = tracker.get_last_detections()
            cc_mask = tracker.get_last_cc_mask()
            if cc_mask is not None and cc_mask.size and detections is not None:
                if cc_mask.ndim == 2:
                    bb_mask = cv2.cvtColor(cc_mask, cv2.COLOR_GRAY2BGR)
                else:
                    bb_mask = cc_mask.copy()
                for x, y, w, h in detections:
                    cv2.rectangle(bb_mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(bb_dir, f"mask_{frame_idx:06d}.jpg"), bb_mask)

    def _append_registrator_summary(self, lines, summary):
        if not summary.get("enabled", True):
            return
        lines.append("Registrator summary (avg ms/frame):")
        if summary.get("calls", 0) == 0:
            lines.append("  no calls recorded")
            return
        lines.append(f"  calls: {summary['calls']}")
        if "gray_ms_avg" in summary:
            lines.append(f"  gray: {summary['gray_ms_avg']:.3f}")
        if "keypoints_ms_avg" in summary:
            lines.append(f"  keypoints: {summary['keypoints_ms_avg']:.3f}")
        if "match_ms_avg" in summary:
            lines.append(f"  match: {summary['match_ms_avg']:.3f}")
        if "homography_ms_avg" in summary:
            lines.append(f"  homography: {summary['homography_ms_avg']:.3f}")
        if "shift_ms_avg" in summary:
            lines.append(f"  shift: {summary['shift_ms_avg']:.3f}")
        if "warp_ms_avg" in summary:
            lines.append(f"  warp: {summary['warp_ms_avg']:.3f}")
        if "registration" in summary:
            lines.append(f"  registration: {summary['registration']:.3f}")
            lines.append(f"  total: {summary['registration']:.3f}")
        if "re_registration_calls" in summary:
            lines.append(f"  re-registrations: {summary['re_registration_calls']}")
        if "registration_first_try_ms_avg" in summary:
            lines.append(f"  registration (first try): {summary['registration_first_try_ms_avg']:.3f}")
        if "registration_fallback_ms_avg" in summary:
            lines.append(f"  registration (fallback): {summary['registration_fallback_ms_avg']:.3f}")
        if "phase_response_avg" in summary:
            lines.append(f"  phase response avg: {summary['phase_response_avg']:.3f}")
        if "ecc_cc_avg" in summary:
            lines.append(f"  ecc cc avg: {summary['ecc_cc_avg']:.3f}")

    def _append_detector_summary(self, lines, summary):
        if not summary.get("enabled", True):
            return
        lines.append("Detector summary (avg ms/frame):")
        if summary.get("calls", 0) == 0:
            lines.append("  no calls recorded")
            return
        lines.append(f"  calls: {summary['calls']}")
        lines.append(f"  blur: {summary['blur_ms_avg']:.3f}")
        lines.append(f"  gray: {summary['gray_ms_avg']:.3f}")
        lines.append(f"  diff: {summary['diff_ms_avg']:.3f}")
        lines.append(f"  bgsub: {summary['bgsub_ms_avg']:.3f}")
        lines.append(f"  contours: {summary['contours_ms_avg']:.3f}")
        if "bbox_creation_ms_avg" in summary:
            lines.append(f"  bbox_creation: {summary['bbox_creation_ms_avg']:.3f}")
        if "upload_ms_avg" in summary:
            lines.append(f"  upload: {summary['upload_ms_avg']:.3f}")
        if "download_ms_avg" in summary:
            lines.append(f"  download: {summary['download_ms_avg']:.3f}")
        lines.append(f"  total: {summary['total_ms_avg']:.3f}")

    def _append_tracker_summary(self, lines, summary):
        if not summary.get("enabled", True):
            return
        lines.append("KalmanIoUTracker summary (avg ms/frame):")
        if summary.get("calls", 0) == 0:
            lines.append("  no calls recorded")
            return
        lines.append(f"  calls: {summary['calls']}")
        lines.append(f"  update: {summary['update_ms_avg']:.3f}")
        lines.append(f"  total: {summary['update_ms_avg']:.3f}")

    # Emit run summaries and optionally build the debug side-by-side video.
    def print_run_summary(self, tracker, loop_total, fps):
        if not self.debug_mode:
            return
        lines = []
        self._append_registrator_summary(lines, tracker.registrator.timing_summary())
        self._append_detector_summary(lines, tracker.detector.timing_summary())
        self._append_tracker_summary(lines, tracker.tracker_timing_summary())
        if self.loop_calls:
            lines.append(f"Loop avg (ms/frame): {(loop_total / self.loop_calls) * 1000.0:.3f}")
        if self.draw_output:
            self.build_frame_vs_mask_video(self.out_dir, self.video_base, fps=fps)
        lines.append("Done")
        for line in lines:
            print(line)
        if self.enable_timing and lines:
            timing_path = os.path.join(self.out_dir, f"timing_summary_{self.video_base}.txt")
            with open(timing_path, "w", encoding="ascii") as log_file:
                log_file.write("\n".join(lines) + "\n")

    # Compute the expected number of frames for progress reporting.
    def compute_remaining_frames(self):
        if self.end_frame is not None:
            return max(0, self.end_frame - self.start_frame + 1)
        if self.total_frames is not None:
            return max(0, self.total_frames - self.start_frame)
        return None

    # Check frame bounds for image lists and video streams.
    def should_stop_processing(self, frame_idx):
        if self.end_frame is not None and frame_idx > self.end_frame:
            return True
        if self.total_frames is not None and frame_idx >= self.total_frames:
            return True
        return False

    # Create a video writer for debug output.
    def create_video_writer(self, out_dir, video_base, fps, shape, is_gray):
        return cv2.VideoWriter(
            os.path.join(out_dir, f"tracked_{video_base}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (shape[1], shape[0]),
            isColor=not is_gray,
        )

    # Lazily attach an output writer once the output shape is known.
    def set_shape(self, shape):
        if not self.debug_mode or not self.draw_output:
            return
        if self._writer is None and self._stream_fps is not None and self._stream_is_gray is not None:
            self._writer = self.create_video_writer(
                self.out_dir,
                self.video_base,
                self._stream_fps,
                shape,
                self._stream_is_gray,
            )

    # Handle per-frame debug overlays and file output when enabled.
    def process_image_outputs(self, tracker, frame_idx, ref_events, homography_debug):
        if not self.debug_mode or not self.draw_output:
            return
        out = self.annotate_registration_debug(
            tracker.get_last_registered_frame(),
            tracker.frame_count,
            self.registration_mode,
            ref_events,
            homography_debug,
            tracker.get_last_shift() if self.registration_mode != "homography" else None,
        )
        out = self.draw_tracks(out, tracker.tracker.get_tracks())
        self.write_debug_images(out, frame_idx, self.frames_dir, self.mog2_dir, self.bb_dir, tracker)
        if self._writer is not None:
            try:
                self._writer.write(out)
            except Exception as e:
                print(f"Error writing frame {tracker.frame_count:06d}: {e}")
        return

    def note_loop_call(self):
        # Track how many frames were processed for summary/graphs.
        self.loop_calls += 1

    # Minimal homography debug logger that owns the file handle.
    class HomographyLog:
        def __init__(self, out_dir, video_base, min_inliers):
            # Open a per-run log file with a header and keep the threshold for context.
            self._min_inliers = min_inliers
            log_path = os.path.join(out_dir, f"homography_log_{video_base}.txt")
            self._log_file = open(log_path, "w", encoding="ascii")
            self._log_file.write(
                "frame,match_count,inlier_count,overlap_ratio,fail_reason,min_inliers\n"
            )
            self._log_file.flush()

        def write(self, frame_idx, homography_debug):
            # Append one row of homography diagnostics for the current frame.
            if self._log_file is None or homography_debug is None:
                return
            self._log_file.write(
                f"{frame_idx},"
                f"{homography_debug.get('match_count', 0)},"
                f"{homography_debug.get('inlier_count', 0)},"
                f"{homography_debug.get('overlap_ratio', 0.0):.6f},"
                f"{homography_debug.get('fail_reason') or 'ok'},"
                f"{self._min_inliers}\n"
            )
            self._log_file.flush()

        def close(self):
            # Close the log file if it is still open.
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
