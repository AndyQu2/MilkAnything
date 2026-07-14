import os
import cv2


def extract_frames_by_fps(video_path, output_dir, target_fps):
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file '{video_path}'. Please check the file path.")
        return

    # 2. Retrieve video metadata
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if orig_fps <= 0 or total_frames <= 0:
        print("❌ Error: Failed to retrieve video FPS or frame count metadata.")
        cap.release()
        return

    duration = total_frames / orig_fps

    print(f"\n🎬 Video metadata loaded:")
    print(f"   - Original FPS: {orig_fps:.2f}")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Duration: {duration:.2f} seconds")

    # 3. Handle target FPS safety limits
    if target_fps > orig_fps:
        print(f"⚠️ Warning: Target FPS ({target_fps}) is higher than original FPS ({orig_fps:.2f}).")
        print("   Downgrading to save all original frames instead.")
        target_fps = orig_fps

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Frame mapping algorithm to prevent float cumulative drift
    total_target_frames = int(duration * target_fps)
    target_indices = set(
        int(i * (orig_fps / target_fps)) for i in range(total_target_frames)
    )

    print(f"🚀 Extraction started. Saving {len(target_indices)} frames in total...")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream

        # Save frame if current index is in the pre-calculated target set
        if frame_idx in target_indices:
            output_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_filename, frame)
            saved_count += 1

            # Print progress bar
            if saved_count % 10 == 0 or saved_count == len(target_indices):
                print(f"   Progress: Saved {saved_count}/{len(target_indices)} frames...", end="\r")

        frame_idx += 1

    cap.release()
    print(f"\n🎉 Extraction completed! Saved {saved_count} frames to directory: '{output_dir}'")


if __name__ == "__main__":
    # Interactive input block
    video_file = input("1. Enter video file path (drag & drop here): ").strip('"').strip("'")
    out_directory = input("2. Enter output directory name (Default: output_frames): ").strip()
    if not out_directory:
        out_directory = "output_frames"

    try:
        user_fps = float(input("3. Enter your target FPS to save (e.g., 5.0): "))
        if user_fps <= 0:
            print("❌ Error: FPS must be greater than 0.")
        else:
            extract_frames_by_fps(video_file, out_directory, user_fps)
    except ValueError:
        print("❌ Error: Invalid input format. FPS must be a number.")