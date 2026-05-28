from ultralytics import YOLO
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python pose_infer.py path/to/image.jpg")
        return

    image_path = sys.argv[1]

    # Load a YOLO pose model (v8n is small + fast)
    model = YOLO("yolov8n-pose.pt")

    # Run inference
    results = model.predict(
        source= image_path,
        project="results",
        save=True,
    )

    

if __name__ == "__main__":
    main()
