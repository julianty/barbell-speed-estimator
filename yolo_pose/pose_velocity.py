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
from utils import extract_frames, frames_to_video, get_video_fps

# Maps CLI model name → single-letter suffix used in YOLO weight filenames (e.g. "small" → "yolo26s-pose.pt")
MODEL_CODES = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}


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

    model = YOLO(model_path)  # model_path set by caller, e.g. models/yolov8s-pose.pt
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    for frame_file in tqdm(frame_files, desc="Running pose inference", unit="frame", ncols=80):
        frame_path = os.path.join(frames_dir, frame_file)
        results = model(frame_path, verbose=False)
        kp    = results[0].keypoints.xy.tolist() if results[0].keypoints is not None else []
        boxes = results[0].boxes.xyxy.tolist()   if results[0].boxes is not None      else []
        h, w  = results[0].orig_shape

        rendered = results[0].plot()
        cv2.imwrite(os.path.join(images_out, frame_file), rendered)

        with open(os.path.join(keypoints_out, frame_file.replace(".jpg", ".json")), "w") as f:
            json.dump({"frame": frame_file, "keypoints": kp, "boxes": boxes, "frame_size": (w, h)}, f, indent=2)

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
    parser.add_argument("model", choices=list(MODEL_CODES), nargs="?", default="large",
                        help="YOLO-pose scale: nano | small | medium | large | xlarge (default: large)")
    parser.add_argument("--yolo-version", choices=["8", "11", "26"], default="26",
                        help="YOLO generation to use (default: 26)")
    parser.add_argument("--video", default=None,
                        help="Path to the input video file (opens file picker if omitted)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Downsample inference to this frame rate (default: use the source video's native FPS)")
    parser.add_argument("--clean", action="store_true",
                        help="Delete rendered images after video reconstruction to save disk space")
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be a positive number")

    if args.video is None:
        from utils import pick_video_file
        args.video = pick_video_file("Select a video for YOLO-pose analysis")
    print(f"[INFO] Video: {args.video}")

    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = "frames"
    # v8 uses the "yolov8" prefix; v11 and v26 use "yolo11" / "yolo26"
    prefix = f"yolov{args.yolo_version}" if args.yolo_version == "8" else f"yolo{args.yolo_version}"
    model_path = os.path.join("models", f"{prefix}{MODEL_CODES[args.model]}-pose.pt")

    source_fps = get_video_fps(video_path)
    print(f"[INFO] Detected source FPS: {source_fps:.3f}")

    if args.fps is not None and args.fps > source_fps:
        print(f"[WARN] --fps {args.fps} exceeds source FPS {source_fps:.3f}; "
              "ffmpeg would just duplicate frames, so this won't add real information.")

    effective_fps = args.fps if args.fps is not None else source_fps
    # Sweeping --fps on the same video/model would otherwise silently overwrite prior runs
    fps_suffix = f"_fps{args.fps:g}" if args.fps is not None else ""
    output_dir = os.path.join("pose_results", video_name, f"yolo{args.yolo_version}_{args.model}{fps_suffix}")
    output_video = os.path.join(output_dir, f"{video_name}_output.mp4")

    if args.fps is not None:
        print(f"[INFO] Downsampling inference to {effective_fps:.3f} FPS")

    # Step 1: dump frames to JPEG files in frames_dir (resampled to --fps if given)
    print("[INFO] Extracting frames...")
    extract_frames(video_path, frames_dir, fps=args.fps)
    # Step 2: run YOLOv8-pose on each frame → keypoint JSONs + barbell_track.json
    run_pose_inference(frames_dir, output_dir, model_path=model_path)

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"fps": effective_fps, "source_fps": source_fps, "video": video_path}, f, indent=2)
    print(f"[INFO] Wrote metadata to {meta_path}")

    images_dir = os.path.join(output_dir, "images")
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    if not image_files:
        print("[ERROR] No rendered images found — skipping video reconstruction.")
    else:
        frames_to_video(images_dir, output_video, fps=effective_fps)
        delete_dir(frames_dir)

        if args.clean:
            delete_dir(images_dir)
        else:
            print("[INFO] Keeping rendered images in", images_dir)
