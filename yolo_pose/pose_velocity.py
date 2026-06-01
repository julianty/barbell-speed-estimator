# pose_velocity.py

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import json
import shutil
import cv2
import argparse
from tqdm import tqdm
from ultralytics import YOLO
from build_barbell_track import build_barbell_track_from_keypoints
from utils import extract_frames, frames_to_video


# ---------------------------------------------------------
# Pose inference
# ---------------------------------------------------------
def run_pose_inference(frames_dir, output_dir, model_path="models/yolov8n-pose.pt"):
    os.makedirs(output_dir, exist_ok=True)

    images_out = os.path.join(output_dir, "images")
    keypoints_out = os.path.join(output_dir, "keypoints")
    track_path = os.path.join(output_dir, "barbell_track.json")

    if os.path.exists(images_out):
        shutil.rmtree(images_out)
    os.makedirs(images_out)

    if os.path.exists(keypoints_out):
        shutil.rmtree(keypoints_out)
    os.makedirs(keypoints_out)

    model = YOLO(model_path)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    for frame_file in tqdm(frame_files, desc="Running pose inference", unit="frame", ncols=80):
        frame_path = os.path.join(frames_dir, frame_file)
        results = model(frame_path, verbose=False)
        kp = results[0].keypoints.xy.tolist() if results[0].keypoints is not None else None
        h, w = results[0].orig_shape
        frame_size = (w, h)

        rendered = results[0].plot()
        cv2.imwrite(os.path.join(images_out, frame_file), rendered)

        with open(os.path.join(keypoints_out, frame_file.replace(".jpg", ".json")), "w") as f:
            json.dump({"frame": frame_file, "keypoints": kp, "frame_size": frame_size}, f, indent=2)

    build_barbell_track_from_keypoints(keypoints_out, track_path)


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
    parser.add_argument("--clean", action="store_true",
                        help="Delete rendered images after video reconstruction to save disk space")
    args = parser.parse_args()

    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = "frames"
    output_dir = "pose_results"
    output_video = os.path.join(output_dir, f"{video_name}_output.mp4")

    print("[INFO] Extracting frames...")
    extract_frames(video_path, frames_dir)
    run_pose_inference(frames_dir, output_dir)

    images_dir = os.path.join(output_dir, "images")
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    if not image_files:
        print("[ERROR] No rendered images found — skipping video reconstruction.")
    else:
        frames_to_video(images_dir, output_video, fps=60)
        delete_dir(frames_dir)

        if args.clean:
            delete_dir(images_dir)
        else:
            print("[INFO] Keeping rendered images in pose_results/images")
