# video_pose_infer.py

import os
import json
import shutil
import cv2
import ffmpeg
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from build_barbell_track import select_lifter_idx, build_barbell_track_from_keypoints


# ---------------------------------------------------------
# Extract frames
# ---------------------------------------------------------
def extract_frames(video_path, frames_dir, fps=None):
    os.makedirs(frames_dir, exist_ok=True)
    output_pattern = os.path.join(frames_dir, "frame_%06d.jpg").replace("\\", "/")

    stream = ffmpeg.input(video_path)
    if fps:
        stream = stream.output(output_pattern, qscale=2, r=fps)
    else:
        stream = stream.output(output_pattern, qscale=2)

    stream.run(quiet=True)

    print(f"[INFO] Frames extracted to: {frames_dir}")


# ---------------------------------------------------------
# Pose inference
# ---------------------------------------------------------
def run_pose_inference(frames_dir, output_dir, keep=False, model_path="yolov8n-pose.pt"):
    os.makedirs(output_dir, exist_ok=True)

    images_out = os.path.join(output_dir, "images")
    keypoints_out = os.path.join(output_dir, "keypoints")
    track_path = os.path.join(output_dir, "barbell_track.json")

    if keep:
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(keypoints_out, exist_ok=True)

    model = YOLO(model_path)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    track = []

    for frame_file in tqdm(frame_files, desc="Running pose inference", unit="frame"):
        frame_path = os.path.join(frames_dir, frame_file)
        results = model(frame_path, verbose=False)
        kp = results[0].keypoints.xy.tolist() if results[0].keypoints is not None else None
        h, w = results[0].orig_shape
        frame_size = (w, h)

        if keep:
            rendered = results[0].plot()
            cv2.imwrite(os.path.join(images_out, frame_file), rendered)

            with open(os.path.join(keypoints_out, frame_file.replace(".jpg", ".json")), "w") as f:
                json.dump({"frame": frame_file, "keypoints": kp, "frame_size": frame_size}, f, indent=2)

        else:
            if kp:
                idx = select_lifter_idx(kp, *frame_size) if len(kp) > 1 else 0
                left = kp[idx][9]
                right = kp[idx][10]
                center = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2]
            else:
                center = None

            track.append({
                "frame": frame_file,
                "center": center
            })

    if keep:
        build_barbell_track_from_keypoints(keypoints_out, track_path)
    else:
        with open(track_path, "w") as f:
            json.dump(track, f, indent=2)
        print(f"[INFO] Saved barbell track to {track_path}")


# ---------------------------------------------------------
# Rebuild video
# ---------------------------------------------------------
def frames_to_video(frames_dir, output_video_path, fps=30):
    input_pattern = os.path.join(frames_dir, "frame_%06d.jpg").replace("\\", "/")

    (
        ffmpeg
        .input(input_pattern, framerate=fps)
        .output(output_video_path, vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"[INFO] Reconstructed video saved to: {output_video_path}")


# ---------------------------------------------------------
# Delete directories
# ---------------------------------------------------------
def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"[INFO] Deleted: {path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Pose Video Pipeline")
    parser.add_argument("video", help="Path to the input video file (e.g. videos/deadlift.mov)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep extracted frames and keypoints (default: delete them)")
    args = parser.parse_args()

    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = "frames"
    output_dir = "pose_results"
    output_video = os.path.join(output_dir, f"{video_name}_output.mp4")

    print("[INFO] Extracting frames...")
    extract_frames(video_path, frames_dir)
    run_pose_inference(frames_dir, output_dir, keep=args.keep)

    images_dir = os.path.join(output_dir, "images")

    # Only rebuild video if images exist (i.e., --keep was used)
    if args.keep and os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        frames_to_video(images_dir, output_video, fps=60)
    else:
        print("[INFO] Skipping video reconstruction (no images saved)")

    if not args.keep:
        delete_dir(frames_dir)
        delete_dir(os.path.join(output_dir, "images"))
    else:
        print("[INFO] Keeping all intermediate files (--keep enabled)")
