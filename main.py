import cv2
import time
import os
import threading
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import playsound

# Load YOLOv8n model (this line ensures model is downloaded once)
YOLO("yolov8n.pt")
model = YOLO("yolov8n.pt")

# Global variables
roi_points = []
drawing_done = False
alarm_playing = False

# Create folder to save intruder frames
save_dir = "intruder_alerts"
os.makedirs(save_dir, exist_ok=True)

def select_roi(event, x, y, flags, param):
    global roi_points, drawing_done
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        print(f"Point {len(roi_points)}: {x}, {y}")
        if len(roi_points) == 4:
            drawing_done = True

def play_alarm_sound():
    global alarm_playing
    try:
        playsound.playsound("mixkit-facility-alarm-sound-999.wav")
    finally:
        alarm_playing = False

def main():
    global alarm_playing

    # Initialize camera
    cap = cv2.VideoCapture(1)  # Change to 1 if needed
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi)

    print("Click 4 points to define ROI...")

    while not drawing_done:
        ret, frame = cap.read()
        if not ret:
            continue
        temp_frame = frame.copy()
        for point in roi_points:
            cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
        cv2.imshow("Select ROI", temp_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select ROI")

    # Create ROI mask
    roi_contour = np.array(roi_points, dtype=np.int32)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_contour], 255)

    print("ROI selected. Starting detection...")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[0], verbose=False)
        intruder_detected = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if roi_mask[cy, cx] == 255:
                    intruder_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Intruder!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.polylines(frame, [roi_contour], isClosed=True, color=(0, 255, 0), thickness=2)

        if intruder_detected:
            print("🚨 Intruder Detected!")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"intruder_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

            if not alarm_playing:
                alarm_playing = True
                threading.Thread(target=play_alarm_sound, daemon=True).start()

        cv2.imshow("AI Security Feed", frame)

        elapsed = time.time() - start_time
        delay = max(1.0 - elapsed, 0)
        if cv2.waitKey(int(delay * 1000)) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
