import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "CK.pt"       
CONF_THRESHOLD = 0.25       


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model_on_image.py path/to/image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("\n[INFO] Model's known classes:")
    for idx, name in model.names.items():
        print(f"   {idx}: {name}")

    print(f"\n[INFO] Running detection on: {image_path}")
    results = model(str(image_path), conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("\n[RESULT] No detections above the confidence threshold.")
    else:
        print(f"\n[RESULT] {len(boxes)} detection(s) found:")
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            label = model.names[cls_id]
            print(f"   - {label}: {conf * 100:.2f}%")

    # save an annotated image next to the original 
    annotated = results[0].plot()
    out_path = image_path.with_name(image_path.stem + "_detected.jpg")
    import cv2
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[INFO] Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()