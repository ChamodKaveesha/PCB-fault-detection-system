import base64
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO


DATABASE_URL = "https://pcb-fault-detection-system-default-rtdb.asia-southeast1.firebasedatabase.app"
MODEL_PATH = "CK.pt"
CONF_THRESHOLD = 0.25
POLL_INTERVAL_SECONDS = 3

# Defect classes to ignore even if the model detects them
EXCLUDED_CLASSES = {"short", "short_circuit", "open_circuit"}


def get_upload_request():
    """Fetch the current upload_request node from Firebase."""
    url = f"{DATABASE_URL}/upload_request.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def mark_processed():
    """Flag the current upload_request as processed so it isn't re-run."""
    url = f"{DATABASE_URL}/upload_request/processed.json"
    requests.put(url, json=True, timeout=10)


def push_status(fault, name, confidence, model_name):
    url = f"{DATABASE_URL}/status.json"
    requests.put(
        url,
        json={
            "fault": fault,
            "name": name,
            "confidence": confidence,
            "model": model_name,
        },
        timeout=10,
    )


def push_snapshot(image_b64):
    url = f"{DATABASE_URL}/snapshot.json"
    requests.put(
        url,
        json={"image": image_b64, "updated_at": int(time.time() * 1000)},
        timeout=10,
    )


def push_log(defect_name, confidence):
    url = f"{DATABASE_URL}/logs.json"
    requests.post(
        url,
        json={
            "timestamp": int(time.time() * 1000),
            "defect_name": defect_name,
            "confidence": confidence,
        },
        timeout=10,
    )


def process_image(image_b64, model):
    """Decode the uploaded image, run detection, and push results to Firebase."""
    img_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("[ERROR] Could not decode uploaded image.")
        return

    results = model(img, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    # Keep only the boxes whose class is NOT in the excluded list
    keep_indices = [
        i for i in range(len(boxes))
        if model.names[int(boxes.cls[i])].lower() not in EXCLUDED_CLASSES
    ]

    if not keep_indices:
        # No valid (non-excluded) detections - draw a plain annotated image with no boxes
        annotated = img.copy()
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            print("[ERROR] Could not encode annotated image.")
            return
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        push_status(fault=False, name="None", confidence=0, model_name=MODEL_PATH)
        push_snapshot(annotated_b64)
        print("[RESULT] No defects detected (excluded classes were ignored, if any).")
        return

    # Draw only the kept (non-excluded) boxes manually
    annotated = img.copy()
    for i in keep_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        cls_id = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        label = model.names[cls_id]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{label} {conf * 100:.1f}%",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        print("[ERROR] Could not encode annotated image.")
        return
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    # Report the highest-confidence kept detection as the headline result
    best_i = max(keep_indices, key=lambda i: float(boxes.conf[i]))
    cls_id = int(boxes.cls[best_i])
    conf_pct = round(float(boxes.conf[best_i]) * 100, 2)
    label = model.names[cls_id]

    push_status(fault=True, name=label, confidence=conf_pct, model_name=MODEL_PATH)
    push_snapshot(annotated_b64)
    push_log(defect_name=label, confidence=conf_pct)
    print(f"[RESULT] Detected: {label} ({conf_pct}%) - {len(keep_indices)} box(es) kept "
          f"({len(boxes) - len(keep_indices)} excluded)")


def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model classes:", model.names)
    print(f"[INFO] Ignoring classes: {EXCLUDED_CLASSES}")
    print("[INFO] Watching Firebase for uploaded images... (Ctrl+C to stop)")

    while True:
        try:
            data = get_upload_request()
            if data and data.get("image") and data.get("processed") is False:
                print("[INFO] New image received. Processing...")
                process_image(data["image"], model)
                mark_processed()
        except requests.RequestException as e:
            print(f"[WARN] Firebase request failed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error while processing: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()