
import argparse
import os
from time import perf_counter
from tqdm import tqdm
from modules.atr_vmd import ATR_VMD
from config_loader import ConfigLoader
from debug_utils import DebugUtils, DataStreamer

def main(data_path, out_dir, config_path):
    config = ConfigLoader(config_path)

    fps_override = config.get("debug.fps", 30.0)
    
    ## ------ Data streamer class ------ ##
    data_streamer = DataStreamer(
        data_path, fps_override, config
    )
    if data_streamer.ref is None:
        return

    reference_window_ms = float(config.get("registration.reference_window_ms", 500.0))
    reference_window_frames = max(1, int(round(data_streamer.fps * (reference_window_ms / 1000.0))))
    config.set_runtime("registration.reference_window_frames", reference_window_frames)
    config.set_runtime("detection.learning_rate", min(1.0, 1.0 / reference_window_frames))
    
    ## ------ Debug utilities class ------ ##
    debug_utils = DebugUtils(
        config,
        data_path,
        out_dir,
        data_streamer.fps,
        data_streamer.resolved_input_is_gray,
    )
    debug_utils.set_stream_info(data_streamer)

    tracker = ATR_VMD(
        data_streamer.ref,
        config,
    )

    loop_total_time = 0.0
    debug_utils.set_shape(tracker.shape_with_margin)

    ref_event_set = set()
    ref_event_len = 0
    with tqdm(total=debug_utils.compute_remaining_frames(), unit="frame") as pbar:
        frame_idx = data_streamer.start_frame
        
        ## ------ Main processing loop ------ ##
        while True:
            frame = data_streamer.get_frame(frame_idx)
            if frame is None:
                break

            ## ------ Process frame ------ ##
            loop_start = perf_counter()
            tracks = tracker.process_frame(frame)
            loop_total_time += perf_counter() - loop_start

            ## ------ Debug processing ------ ##
            debug_utils.note_loop_call()
            
            homography_debug = None
            if debug_utils.registration_mode == "homography":
                homography_debug = tracker.registrator.get_last_homography_debug()
                # Debug: persist homography diagnostics for this frame.
                debug_utils.write_homography_log(tracker.frame_count, homography_debug)

            ref_events = tracker.get_reference_events()
            if ref_events is not None:
                if len(ref_events) != ref_event_len:
                    ref_event_set = set(int(ev) for ev in ref_events)
                    ref_event_len = len(ref_events)

            debug_utils.process_image_outputs(
                tracker,
                frame_idx,
                ref_event_set if ref_events is not None else None,
                homography_debug,
            )
            pbar.update(1)
            frame_idx += 1

    debug_utils.close()
    data_streamer.close()

    # Debug: write a reference-event timeline image for quick inspection.
    debug_utils.write_reference_events_graph(
        tracker.get_reference_events(),
    )

    # Debug: print timing summaries and build side-by-side debug video.
    debug_utils.print_run_summary(
        tracker,
        loop_total_time,
        data_streamer.fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/home/nvidia/workspace/oded/FireArrow/data/videos/06.01.2026/Shahed_02.mkv")
    parser.add_argument("--out-dir", default="/home/nvidia/workspace/oded/FireArrow/ATR-VMD/results")
    parser.add_argument(
        "--config-path",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(
        args.data_path,
        args.out_dir,
        args.config_path,
    )
