import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from collections import deque
from datetime import datetime
import math
import os

# ------------- USER SETTINGS -------------
MODEL_PATH = r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\model\model_best.pt"
# Put two camera base URLs here (change to your actual IPs/ports)
IPCAM_BASES = [
    "http://10.84.242.43:8080",   # camera 1
    "http://10.84.242.34:8080",   # camera 2 (example)
    #or use video links present on you desktop
    # r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\4K Road traffic video for object detection and tracking ",
    #  r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\Road traffic video for object detection"
]
OUTPUT_JSON_PATH = "traffic_data.json"    # real-time json output
JSON_UPDATE_INTERVAL = 3.0                # seconds between JSON updates (published smoothed)
SMOOTHING_ALPHA = 0.6                     # smoothing weight for published value (0..1)
TRAFFIC_LIMIT = 2                         # threshold for Heavy/Smooth
MAX_WINDOW_SECONDS = 20.0                 # lookback for "max" window
TOTAL_20S_REFRESH = 20.0                  # refresh interval for total-in-20s
# -----------------------------------------

# Region of interest & display positions (same for both windows)
slice_top, slice_bottom = 325, 635
label_pos = (15, 55)
intensity_pos = (15, 105)
max20_pos = (15, 155)
total20_pos = (15, 205)  # where "Total (past 20s)" will be shown

# Font & style
font_style = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
text_color = (255, 255, 255)   # white
box_color = (0, 0, 255)        # red background

# Candidate endpoints commonly exposed by mobile IP camera apps
COMMON_ENDPOINTS = [
    "",                 # base (sometimes works)
    "/video",
    "/video_feed",
    "/shot.jpg",
    "/?action=stream",
    "/stream",
]

# ---------- helpers ----------
def write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def try_open_ipcam(base_url, endpoints, wait_first_frame=2.0):
    """Try multiple endpoints and return an opened cv2.VideoCapture and the first frame (or (None, None))."""
    for ep in endpoints:
        url = base_url.rstrip("/") + ep
        print(f"Trying stream: {url}")
        cap = cv2.VideoCapture(url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        start = time.time()
        while time.time() - start < wait_first_frame:
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, frame
        cap.release()
    return None, None

# ---------- Simple centroid tracker (unchanged) ----------
class CentroidTracker:
    def __init__(self, max_disappeared_seconds=1.0, max_distance=60):
        self.next_id = 1
        self.objects = {}
        self.last_seen = {}
        self.max_disappeared_seconds = max_disappeared_seconds
        self.max_distance = max_distance

    def register(self, centroid, ts):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = centroid
        self.last_seen[oid] = ts
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        # keep last_seen for unique-in-last-20s counting

    def update(self, input_centroids, ts):
        if len(input_centroids) == 0:
            to_remove = []
            for oid, last in list(self.last_seen.items()):
                if ts - last > self.max_disappeared_seconds:
                    to_remove.append(oid)
            for oid in to_remove:
                self.deregister(oid)
            return dict(self.objects)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c, ts)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            self.objects[oid] = input_centroids[c]
            self.last_seen[oid] = ts
            assigned_rows.add(r)
            assigned_cols.add(c)

        for j in range(len(input_centroids)):
            if j not in assigned_cols:
                self.register(input_centroids[j], ts)

        for i in range(len(object_centroids)):
            if i not in assigned_rows:
                oid = object_ids[i]
                if ts - self.last_seen.get(oid, 0) > self.max_disappeared_seconds:
                    self.deregister(oid)

        return dict(self.objects)

    def count_unique_since(self, since_ts):
        cnt = sum(1 for oid, last in self.last_seen.items() if last >= since_ts)
        return cnt

# ---------- load model ----------
print("Loading YOLO model...")
traffic_model = YOLO(MODEL_PATH)
print("Model loaded.")

# ---------- open IP cams ----------
caps = []
first_frames = []
for i, base in enumerate(IPCAM_BASES):
    cap, ff = try_open_ipcam(base, COMMON_ENDPOINTS)
    if cap is None:
        print(f"⚠️ Could not open IP camera {i+1}. Tried endpoints for: {base}")
    else:
        print(f"✅ Opened IP camera {i+1} stream successfully: {base}")
    caps.append(cap)
    first_frames.append(ff)

# If no cameras opened, exit
if all(c is None for c in caps):
    print("No IP cameras could be opened. Exiting.")
    exit()

# ---------- prepare writers and per-camera state ----------
frame_sizes = []
writers = []
for idx, ff in enumerate(first_frames):
    if ff is not None:
        h, w = ff.shape[:2]
        frame_sizes.append((w, h))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writers.append(cv2.VideoWriter(f"traffic_output_cam{idx+1}.avi", fourcc, 20.0, (w, h)))
    else:
        frame_sizes.append(None)
        writers.append(None)

# Per-camera buffers and trackers
camera_state = {}
for idx in range(len(IPCAM_BASES)):
    camera_state[idx] = {
        "counts_buffer": [],          # raw counts for median smoothing
        "deque_20s": deque(),         # (timestamp, raw_count)
        "tracker": CentroidTracker(max_disappeared_seconds=1.0, max_distance=60),
        "last_total20_refresh": time.time(),
        "total_unique_20s": 0,
        "last_published_count": None,
    }

# utility: process_frame uses global model
def process_frame(frame):
    """Run detection on ROI and return overlay image, raw vehicle count, and bbox centroids list."""
    roi_frame = frame.copy()
    roi_frame[:slice_top, :] = 0
    roi_frame[slice_bottom:, :] = 0

    detection = traffic_model.predict(roi_frame, imgsz=640, conf=0.4)
    overlay = detection[0].plot(line_width=1)

    # restore top/bottom so overlay isn't black there
    overlay[:slice_top, :] = frame[:slice_top, :]
    overlay[slice_bottom:, :] = frame[slice_bottom:, :]

    boxes = detection[0].boxes
    vehicle_count = len(boxes)

    centroids = []
    try:
        xyxy = boxes.xyxy.cpu().numpy()
    except Exception:
        xyxy = np.array([]).reshape(0,4)

    for b in xyxy:
        x1, y1, x2, y2 = b
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        centroids.append((cx, cy))

    return overlay, vehicle_count, centroids

# If we have first frames, process them once to initialize state and writers
for idx, ff in enumerate(first_frames):
    if ff is None:
        continue
    overlay, vehicle_count, centroids = process_frame(ff)
    now_ts = time.time()
    st = camera_state[idx]
    st["counts_buffer"].append(vehicle_count)
    st["deque_20s"].append((now_ts, vehicle_count))
    st["tracker"].update(centroids, now_ts)
    if writers[idx] is not None:
        writers[idx].write(overlay)
    winname = f"Traffic Monitor - Camera {idx+1}"
    cv2.imshow(winname, overlay)

print("Starting main loop. Press 'q' in any display window to quit.")

window_start = time.time()

# ---------- main loop ----------
while True:
    any_frame_processed = False
    now_ts = time.time()

    # iterate each camera sequentially
    for idx, cap in enumerate(caps):
        if cap is None:
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            # small sleep to avoid busy loop; continue to next camera
            time.sleep(0.01)
            continue

        any_frame_processed = True
        overlay, vehicle_count, centroids = process_frame(frame)
        st = camera_state[idx]

        # update buffers
        st["counts_buffer"].append(vehicle_count)
        st["deque_20s"].append((now_ts, vehicle_count))

        # remove old entries from deque_20s
        cutoff = now_ts - MAX_WINDOW_SECONDS
        while st["deque_20s"] and st["deque_20s"][0][0] < cutoff:
            st["deque_20s"].popleft()

        # update tracker
        st["tracker"].update(centroids, now_ts)

        # compute max over last 20 seconds
        if st["deque_20s"]:
            max_20s = max(v for (_, v) in st["deque_20s"])
        else:
            max_20s = 0

        # compute immediate raw status
        status_immediate = "Heavy" if vehicle_count > TRAFFIC_LIMIT else "Smooth"

        # Refresh total_unique_20s every TOTAL_20S_REFRESH seconds
        if now_ts - st["last_total20_refresh"] >= TOTAL_20S_REFRESH:
            since = now_ts - MAX_WINDOW_SECONDS
            st["total_unique_20s"] = st["tracker"].count_unique_since(since)
            st["last_total20_refresh"] = now_ts

        # Draw overlays
        cv2.rectangle(
            overlay,
            (label_pos[0]-12, label_pos[1]-28),
            (label_pos[0] + 520, label_pos[1] + 12),
            box_color, -1
        )
        cv2.putText(
            overlay, f"Total Vehicles (raw): {vehicle_count}",
            label_pos, font_style, scale, text_color, 2, cv2.LINE_AA
        )

        cv2.rectangle(
            overlay,
            (intensity_pos[0]-12, intensity_pos[1]-28),
            (intensity_pos[0] + 520, intensity_pos[1] + 12),
            box_color, -1
        )
        cv2.putText(
            overlay, f"Traffic Intensity (raw): {status_immediate}",
            intensity_pos, font_style, scale, text_color, 2, cv2.LINE_AA
        )

        cv2.rectangle(
            overlay,
            (max20_pos[0]-12, max20_pos[1]-28),
            (max20_pos[0] + 520, max20_pos[1] + 12),
            box_color, -1
        )
        cv2.putText(
            overlay, f"Max (20s): {max_20s}",
            max20_pos, font_style, scale, text_color, 2, cv2.LINE_AA
        )

        cv2.rectangle(
            overlay,
            (total20_pos[0]-12, total20_pos[1]-28),
            (total20_pos[0] + 520, total20_pos[1] + 12),
            box_color, -1
        )
        cv2.putText(
            overlay, f"Total (past 20s): {st['total_unique_20s']}",
            total20_pos, font_style, scale, text_color, 2, cv2.LINE_AA
        )

        # show & save
        winname = f"Traffic Monitor - Camera {idx+1}"
        cv2.imshow(winname, overlay)
        if writers[idx] is not None:
            writers[idx].write(overlay)

    # JSON update every JSON_UPDATE_INTERVAL seconds (compute per-camera published counts and write combined JSON)
    if time.time() - window_start >= JSON_UPDATE_INTERVAL:
        combined_payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "cameras": {}
        }

        for idx in range(len(IPCAM_BASES)):
            st = camera_state[idx]
            # median_count from buffer for this camera
            if st["counts_buffer"]:
                median_count = int(round(np.median(st["counts_buffer"])))
            else:
                median_count = 0

            if st["last_published_count"] is None:
                published_count = median_count
            else:
                published_count = int(round((1 - SMOOTHING_ALPHA) * st["last_published_count"] + SMOOTHING_ALPHA * median_count))

            # compute current max_20s for camera
            if st["deque_20s"]:
                max_20s = max(v for (_, v) in st["deque_20s"])
            else:
                max_20s = 0

            status_published = "Heavy" if published_count > TRAFFIC_LIMIT else "Smooth"

            camera_payload = {
                "vehicle_count_published": published_count,
                "vehicle_count_median_raw": median_count,
                # note: raw_latest here may be stale if camera produced no frames in the last interval;
                # we don't maintain the last per-camera raw_latest separately, so we approximate using median buffer's last appended (or 0)
                "vehicle_count_raw_latest": st["counts_buffer"][-1] if st["counts_buffer"] else 0,
                "max_vehicle_count_20s": max_20s,
                "total_unique_vehicles_past_20s": st["total_unique_20s"],
                "status": status_published,
                "window_seconds": JSON_UPDATE_INTERVAL
            }

            combined_payload["cameras"][f"camera_{idx+1}"] = camera_payload

            # commit published count and clear buffer
            st["last_published_count"] = published_count
            st["counts_buffer"].clear()

        # write atomic json
        try:
            write_json_atomic(OUTPUT_JSON_PATH, combined_payload)
        except Exception as e:
            print("Error writing JSON:", e)

        window_start = time.time()

    # Quit on 'q' in any window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup
for cap in caps:
    if cap is not None:
        cap.release()
for w in writers:
    if w is not None:
        w.release()
cv2.destroyAllWindows()

