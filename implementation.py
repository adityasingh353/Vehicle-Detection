import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLOv8 model
#model_path = r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\model\best.pt"
model_path =r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\model\model_best.pt"
traffic_model = YOLO(model_path)

# Parameters
TRAFFIC_LIMIT = 10   # Threshold for heavy traffic
slice_top, slice_bottom = 325, 635  # Region of Interest (ROI)
label_pos = (15, 55)
intensity_pos = (15, 105)

# Font and style configs
font_style = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
text_color = (255, 255, 255)   # White text
box_color = (0, 0, 255)        # Red background

# Open video
#video_path = r"C:\\Users\\ADITYA SINGH\\Downloads\\Spark traffic video surveillance.mp4"#video source:https://www.youtube.com/watch?v=MNn9qKG2UFI
video_path=r"C:\Users\ADITYA SINGH\OneDrive\Documents\Project\4K Road traffic video for object detection and tracking - free download now!.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("⚠️ Could not load video file!")
    exit()

# Output writer
codec = cv2.VideoWriter_fourcc(*'XVID')
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter("traffic_output.avi", codec, 20.0, (frame_w, frame_h))

# Processing loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    roi_frame = frame.copy()
    roi_frame[:slice_top, :] = 0
    roi_frame[slice_bottom:, :] = 0

    # Run YOLO detection
    detection = traffic_model.predict(roi_frame, imgsz=640, conf=0.4)
    overlay = detection[0].plot(line_width=1)

    # Restore original regions
    overlay[:slice_top, :] = frame[:slice_top, :]
    overlay[slice_bottom:, :] = frame[slice_bottom:, :]

    # Extract detections
    boxes = detection[0].boxes
    vehicle_count = len(boxes)

    # Decide intensity
    status = "Heavy" if vehicle_count > TRAFFIC_LIMIT else "Smooth"

    # Draw count background + text
    cv2.rectangle(
        overlay, 
        (label_pos[0]-12, label_pos[1]-28), 
        (label_pos[0] + 470, label_pos[1] + 12), 
        box_color, -1
    )
    cv2.putText(
        overlay, f"Total Vehicles: {vehicle_count}", 
        label_pos, font_style, scale, text_color, 2, cv2.LINE_AA
    )

    # Draw intensity background + text
    cv2.rectangle(
        overlay, 
        (intensity_pos[0]-12, intensity_pos[1]-28), 
        (intensity_pos[0] + 470, intensity_pos[1] + 12), 
        box_color, -1
    )
    cv2.putText(
        overlay, f"Traffic Intensity: {status}", 
        intensity_pos, font_style, scale, text_color, 2, cv2.LINE_AA
    )

    # Show & Save
    cv2.imshow("Traffic Monitor", overlay)
    writer.write(overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
writer.release()
cv2.destroyAllWindows()
