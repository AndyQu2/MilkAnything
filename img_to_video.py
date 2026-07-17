import os
import cv2
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_single_video(video_path, output_root, target_fps=None, frame_stride=None, overwrite=False):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        first_frame_name = f"{video_name}_frame_000000.jpg"
        first_frame_path = os.path.join(output_root, first_frame_name)
        if not overwrite and os.path.exists(first_frame_path):
            return {"video": video_name, "status": "skipped", "saved_count": 0}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"video": video_name, "status": "corrupted", "saved_count": 0}

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            original_fps = 25.0

        if target_fps is not None:
            stride = max(1, int(original_fps / target_fps))
        elif frame_stride is not None:
            stride = frame_stride
        else:
            stride = 1

        saved_count = 0
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            img_name = f"{video_name}_frame_{frame_idx:06d}.jpg"
            img_path = os.path.join(output_root, img_name)

            cv2.imwrite(img_path, frame)

            saved_count += 1
            frame_idx += stride

        cap.release()
        return {"video": video_name, "status": "success", "saved_count": saved_count}

    except Exception as e:
        return {"video": os.path.basename(video_path), "status": f"failed (Error: {str(e)})", "saved_count": 0}


def video_path_generator(video_dir):
    valid_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.webm', '.ts')
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                yield os.path.join(root, file)


def main():
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
        help="Number of concurrent processes to run (defaults to min(8, cpu_cores))."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite and re-process videos even if their frames already exist in the output folder."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_fps = args.fps
    frame_stride = args.stride
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

    worker_func = partial(
        extract_single_video,
        output_root=args.output_dir,
        target_fps=target_fps,
        frame_stride=frame_stride,
        overwrite=args.overwrite
    )

    video_stream = video_path_generator(args.input_dir)

    success_cnt = 0
    skipped_cnt = 0
    failed_cnt = 0
    total_frames_saved = 0

    with Pool(processes=args.jobs) as pool:
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

    print("\n================== Execution Report ==================")
    print(f"✅ Successfully processed : {success_cnt} video(s)")
    print(f"⏩ Skipped (already done) : {skipped_cnt} video(s)")
    print(f"❌ Failed / Corrupted     : {failed_cnt} video(s)")
    print(f"🖼️  Total images generated  : {total_frames_saved} frame(s)")
    print("======================================================")


if __name__ == "__main__":
    main()