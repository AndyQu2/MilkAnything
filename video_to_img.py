#!/usr/bin/env python
# coding=utf-8
"""
High-Performance Video Frame Extraction CLI Tool
=================================================
A parallelized utility designed to recursively scan a target input directory for video assets,
extract frames at a user-defined cadence, and output them into a single, flattened target folder.

Optimizations include:
- Multi-processing via `multiprocessing.Pool` using sub-process workers.
- Frame indexing acceleration utilizing `cv2.CAP_PROP_POS_FRAMES`.
- Checkpoint resumption capabilities to skip pre-processed assets.
"""

import os
import cv2
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_single_video(video_path, output_root, target_fps=None, frame_stride=None, overwrite=False):
    """
    Worker function responsible for extracting frames from a single video asset.

    Args:
        video_path (str): Absolute or relative filesystem path to the target video.
        output_root (str): The flat directory where extracted frame images are saved.
        target_fps (float, optional): The target extraction frame rate per second.
        frame_stride (int, optional): The static index step interval between extracted frames.
        overwrite (bool): If True, re-processes the video even if prior outputs exist.

    Returns:
        dict: Operational metadata tracking the video name, execution status, and frame counts.
    """
    try:
        # Extract the base filename without extension to use as a primary prefix key
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Checkpoint Check: Verify if frame 0 already exists in the destination to implement resume mode
        first_frame_name = f"{video_name}_frame_000000.jpg"
        first_frame_path = os.path.join(output_root, first_frame_name)
        if not overwrite and os.path.exists(first_frame_path):
            return {"video": video_name, "status": "skipped", "saved_count": 0}

        # Instantiate the OpenCV VideoCapture object to interface with the media file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video": video_name, "status": "corrupted", "saved_count": 0}

        # Query metadata properties embedded within the video container
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Fallback safeguard: If FPS metadata is invalid or missing, default to a standard 25.0 FPS
        if original_fps <= 0:
            original_fps = 25.0

        # Calculate the dynamic stride if a target FPS is requested
        if target_fps is not None:
            # Stride dictates how many structural frames to jump over per loop iteration
            stride = max(1, int(original_fps / target_fps))
        elif frame_stride is not None:
            stride = frame_stride
        else:
            stride = 1

        saved_count = 0
        frame_idx = 0

        # Primary extraction loop over the absolute frame boundary of the video stream
        while frame_idx < total_frames:
            # Physically seek the decoder pointer to the absolute index layout position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break  # Terminate decoding loop if frame collection fails or EOF is met

            # Format the output name with a zero-padded string to preserve alphanumeric ordering
            img_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
            img_path = os.path.join(output_root, img_name)

            # Persist the decompressed array surface matrix onto the file system as an image
            cv2.imwrite(img_path, frame)

            saved_count += 1
            frame_idx += stride  # Shift pointer target forward by the computational stride

        # Release file descriptor assets back to the underlying operating system
        cap.release()
        return {"video": video_name, "status": "success", "saved_count": saved_count}

    except Exception as e:
        # Intercept local exceptions to prevent global worker termination crashes
        return {"video": os.path.basename(video_path), "status": f"failed (Error: {str(e)})", "saved_count": 0}


def video_path_generator(video_dir):
    """
    Generator expression that lazily walks through directories to surface media paths.
    Avoids loading full filesystem structures into local system RAM concurrently.
    """
    valid_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.ts')
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                yield os.path.join(root, file)


def main():
    # Setup standard Command Line Interface schema parser definitions
    parser = argparse.ArgumentParser(
        description="A high-performance CLI tool to extract frames from massive video directories into a single flat folder."
    )

    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing source videos."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Path to the flat directory where all extracted frames will be saved."
    )

    # Establish an exclusive constraint layout context: User must choose either static FPS or absolute stride
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target frame rate per second to extract (e.g., 2 means 2 frames per second)."
    )
    group.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Target frame stride (e.g., 5 means extract 1 frame every 5 frames)."
    )

    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=min(8, cpu_count()),
        help="Number of concurrent processes to run (defaults to min(8, physical_cpu_cores))."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite and re-process videos even if their frames already exist in the output folder."
    )

    args = parser.parse_args()

    # Ensure target output tree constructs exist safely before processing begins
    os.makedirs(args.output_dir, exist_ok=True)

    target_fps = args.fps
    frame_stride = args.stride

    # Enforce default behavior constraint adjustments if parameter selections are empty
    if target_fps is None and frame_stride is None:
        print("[Info] No extraction rate specified. Defaulting to --fps 1.0 (1 frame per second).")
        target_fps = 1.0

    print("==================================================")
    print("🚀 Starting Video Frame Extraction CLI (Flat Folder)...")
    print(f"📂 Input Directory  : {args.input_dir}")
    print(f"📂 Output Directory : {args.output_dir} (ALL FRAMES GO HERE)")
    print(f"⚡ Parallel Jobs     : {args.jobs}")
    if target_fps:
        print(f"🎯 Sampling Rate    : {target_fps} FPS")
    else:
        print(f"🎯 Sampling Rate    : Every {frame_stride} frame(s) (stride)")
    print(f"🔄 Overwrite Mode   : {'Enabled' if args.overwrite else 'Disabled (Resume Mode)'}")
    print("==================================================")

    # Freeze shared operational keyword settings configurations into a callable partial reference mapping
    worker_func = partial(
        extract_single_video,
        output_root=args.output_dir,
        target_fps=target_fps,
        frame_stride=frame_stride,
        overwrite=args.overwrite
    )

    # Instantiate the lazy path data-stream reader mapping structure
    video_stream = video_path_generator(args.input_dir)

    success_cnt = 0
    skipped_cnt = 0
    failed_cnt = 0
    total_frames_saved = 0

    # Initialize the high-concurrency multiprocessing process engine pool
    with Pool(processes=args.jobs) as pool:
        # Utilize imap_unordered for highly efficient scheduling; returns results out of order to prevent line blocking.
        # chunksize=10 mitigates process allocation thrashing over extensive loops.
        for result in pool.imap_unordered(worker_func, video_stream, chunksize=10):
            status = result["status"]
            video_name = result["video"]

            if status == "success":
                success_cnt += 1
                total_frames_saved += result["saved_count"]
                print(f"[SUCCESS] {video_name} ➜ Extracted {result['saved_count']} frames.")
            elif status == "skipped":
                skipped_cnt += 1
            else:
                failed_cnt += 1
                print(f"[FAILED]  {video_name} ➜ Reason: {status}")

    # Generate analytical overview statistics summaries for administrative tracking
    print("\n================== Execution Report ==================")
    print(f"✅ Successfully processed : {success_cnt} video(s)")
    print(f"⏩ Skipped (already done) : {skipped_cnt} video(s)")
    print(f"❌ Failed / Corrupted     : {failed_cnt} video(s)")
    print(f"🖼️  Total images generated  : {total_frames_saved} frame(s)")
    print("======================================================")


if __name__ == "__main__":
    main()