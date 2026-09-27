import cv2
import time
import base64
import threading
import queue
import requests
from ultralytics import YOLO

FIREBASE_DB_URL = "https://pcb-fault-detection-system-default-rtdb.asia-southeast1.firebasedatabase.app/"  

CAMERA_INDEX = 0
FRAME_WIDTH = 1280      
FRAME_HEIGHT = 720

AUTO_FOCUS = False          
MANUAL_FOCUS_VALUE = 30     

SHOW_LOCAL_WINDOW = True     # cv2.imshow debug window, set False if running headless

ENABLE_ENHANCEMENT = True    # quality boost 
ENHANCE_SCALE = 1.0          # keep at 1.0 for optimal performance
SHARPEN_STRENGTH = 1.0

DETECTION_IMG_SIZE = 640     

SNAPSHOT_INTERVAL = 1.0      # seconds between dashboard preview image updates
STATUS_INTERVAL = 0.5        # seconds between status updates to Firebase
SNAPSHOT_MAX_WIDTH = 480     # downsized before upload to keep payload small

# Defect classes to completely ignore, even if the model detects them
EXCLUDED_CLASSES = {"short", "short_circuit", "open_circuit"}

MODEL_REGISTRY = {
    "pcb_defect": {"path": "CK.pt", "conf": 0.5},
}

REPAIR_INFO = {
    "missing_hole":    {"repairable": True, "action": "Drill/re-via the missing hole"},
    "spur":            {"repairable": False, "action": "Replace board - excess copper defect"},
    "spurious_copper": {"repairable": False, "action": "Replace board - unwanted copper deposit"},
    "mouse_bite":      {"repairable": False, "action": "Replace board - notched/damaged trace"},
}
LOW_CONFIDENCE_THRESHOLD = 0.60  # flag for manual review instead of auto deciding


def assess_repairability(defect_name, confidence_pct):
    """Returns (verdict, action). verdict: Repairable / Replace / Manual Inspection Needed / N/A."""
    if defect_name is None or defect_name == "None":
        return "N/A", "No defect detected"
    if confidence_pct / 100.0 < LOW_CONFIDENCE_THRESHOLD:
        return "Manual Inspection Needed", "Detection confidence too low to auto-decide"

    info = REPAIR_INFO.get(defect_name)
    if info is None:
        return "Manual Inspection Needed", "Unrecognized defect type"
    return ("Repairable" if info["repairable"] else "Replace"), info["action"]


was_fault_active = False
last_snapshot_time = 0.0

_firebase_queue = queue.Queue()


def _firebase_worker():
    while True:
        method, path, data = _firebase_queue.get()
        try:
            if method == "PUT":
                requests.put(f"{FIREBASE_DB_URL}/{path}.json", json=data, timeout=10)
            elif method == "POST":
                requests.post(f"{FIREBASE_DB_URL}/{path}.json", json=data, timeout=10)
        except requests.RequestException as e:
            print(f"[FIREBASE] {method} {path} failed:", e)
        _firebase_queue.task_done()


def fb_put(path, data):
    # Drop stale updates if queue is backing up
    while _firebase_queue.qsize() > 5:
        try:
            _firebase_queue.get_nowait()
            _firebase_queue.task_done()
        except queue.Empty:
            break
    _firebase_queue.put(("PUT", path, data))


def fb_push(path, data):
    while _firebase_queue.qsize() > 5:
        try:
            _firebase_queue.get_nowait()
            _firebase_queue.task_done()
        except queue.Empty:
            break
    _firebase_queue.put(("POST", path, data))


def push_status(name, confidence, model_name):
    fault = name != "None"
    verdict, action = assess_repairability(name, confidence)
    fb_put("status", {
        "name": name,
        "confidence": confidence,
        "model": model_name,
        "fault": fault,
        "repair_verdict": verdict,
        "repair_action": action,
        "updated_at": int(time.time() * 1000),  # ms epoch
    })
    return fault


def push_log_entry(model_name, defect_name, confidence):
    verdict, action = assess_repairability(defect_name, confidence)
    fb_push("logs", {
        "timestamp": int(time.time() * 1000),
        "model": model_name,
        "defect_name": defect_name,
        "confidence": confidence,
        "repair_verdict": verdict,
        "repair_action": action,
    })


def push_snapshot(frame):
    h, w = frame.shape[:2]
    if w > SNAPSHOT_MAX_WIDTH:
        scale = SNAPSHOT_MAX_WIDTH / w
        frame = cv2.resize(frame, (SNAPSHOT_MAX_WIDTH, int(h * scale)))

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        return
    b64 = base64.b64encode(buf).decode("utf-8")
    fb_put("snapshot", {"image": b64, "updated_at": int(time.time() * 1000)})


def enhance_frame(frame):
    if ENHANCE_SCALE != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * ENHANCE_SCALE), int(h * ENHANCE_SCALE)),
                           interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if SHARPEN_STRENGTH > 0:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        frame = cv2.addWeighted(frame, 1 + SHARPEN_STRENGTH, blurred, -SHARPEN_STRENGTH, 0)

    return frame


def load_models():
    loaded = {}
    for name, cfg in MODEL_REGISTRY.items():
        print(f"[INFO] Loading model '{name}' from {cfg['path']}")
        loaded[name] = YOLO(cfg["path"])
    return loaded


def draw_box(frame, box, label, conf):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"{label} {conf * 100:.1f}%",
        (x1, max(y1 - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )


def run_all_models(models, frame):
    best_name, best_conf, best_model = None, 0.0, None
    annotated = frame.copy()
    any_kept_box = False

    for model_name, model in models.items():
        conf_threshold = MODEL_REGISTRY[model_name]["conf"]

        results = model(frame, conf=conf_threshold, imgsz=DETECTION_IMG_SIZE, verbose=False)
        boxes = results[0].boxes

        kept = [b for b in boxes if model.names[int(b.cls[0])].lower() not in EXCLUDED_CLASSES]

        for box in kept:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            draw_box(annotated, box, label, conf)
            any_kept_box = True
            if conf > best_conf:
                best_conf = conf
                best_name = label
                best_model = model_name

    if not any_kept_box:
        annotated = frame

    return best_name, round(best_conf * 100, 1), best_model, annotated


def setup_camera(index: int) -> cv2.VideoCapture:
    # Use Media Foundation backend for Windows
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {index}")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if AUTO_FOCUS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.2)
        cap.set(cv2.CAP_PROP_FOCUS, MANUAL_FOCUS_VALUE)

    time.sleep(1.0)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Camera active resolution: {actual_w}x{actual_h} @ {actual_fps} FPS")

    return cap


def main():
    global was_fault_active, last_snapshot_time

    if "YOUR-PROJECT" in FIREBASE_DB_URL:
        print("[ERROR] Set FIREBASE_DB_URL at the top of this file before running.")
        return

    models = load_models()
    print(f"[INFO] Ignoring classes: {EXCLUDED_CLASSES}")

    threading.Thread(target=_firebase_worker, daemon=True).start()

    try:
        cap = setup_camera(CAMERA_INDEX)
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    print("[INFO] Pushing live data to Firebase:", FIREBASE_DB_URL)

    try:
        last_status_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if ENABLE_ENHANCEMENT:
                frame = enhance_frame(frame)

            name, conf, model_name, annotated_frame = run_all_models(models, frame)

            # 1. Handle Logging (only log on transition from OK to FAULT)
            current_fault = (name is not None)
            if current_fault and not was_fault_active:
                push_log_entry(model_name, name, conf)
            was_fault_active = current_fault

            # 2. Handle Status Uploads (Throttled)
            now = time.time()
            if now - last_status_time >= STATUS_INTERVAL:
                if name:
                    push_status(name, conf, model_name)
                    print(f"[DETECT] ({model_name}) {name} | {conf:.2f}%")
                else:
                    push_status("None", 0.0, None)
                last_status_time = now

            # 3. Handle Snapshot Uploads (Throttled)
            if now - last_snapshot_time >= SNAPSHOT_INTERVAL:
                push_snapshot(annotated_frame)
                last_snapshot_time = now

            if SHOW_LOCAL_WINDOW:
                cv2.imshow("PCB Fault Detection - Live", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("[INFO] Stopped with Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()