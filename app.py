"""
-----------------------------------------------------
   OBJECT DETECTION & TRACKING — CodeAlpha Task 4         
   Built with YOLOv8 + OpenCV + ByteTrack                 
-----------------------------------------------------

HOW TO RUN:
    python app.py                    ← uses webcam
    python app.py --source video.mp4 ← uses a video file

CONTROLS (while window is open):
    Q  →  Quit
    S  →  Save screenshot
    P  →  Pause / Resume
"""

import cv2
import argparse
import time
import os
from datetime import datetime
from ultralytics import YOLO


# SETTINGS
MODEL_PATH      = "yolov8n.pt"   # nano model — fast & lightweight (auto-downloads)
CONFIDENCE      = 0.4            # minimum confidence threshold (0–1)
WINDOW_NAME     = "CodeAlpha | Object Detection & Tracking"
SCREENSHOT_DIR  = "screenshots"

# Aesthetic colour palette for bounding boxes (BGR format)
COLORS = [
    (255, 99,  71),   # tomato red
    (255, 165,  0),   # orange
    (50,  205, 50),   # lime green
    (30,  144, 255),  # dodger blue
    (186,  85, 211),  # medium orchid
    (255, 20, 147),   # deep pink
    (0,   206, 209),  # dark turquoise
    (255, 215,   0),  # gold
    (127, 255,   0),  # chartreuse
    (255, 127,  80),  # coral
]

def get_color(track_id):
    """Return a consistent colour for each tracking ID."""
    return COLORS[int(track_id) % len(COLORS)]

def draw_box(frame, x1, y1, x2, y2, label, color, conf):
    """Draw a clean bounding box with label on the frame."""
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)

    # Label text
    cv2.putText(frame, text, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def draw_track_id(frame, x1, y1, x2, y2, track_id, color):
    """Draw tracking ID badge in the corner of the box."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    tid_text = f"#{int(track_id)}"
    (tw, th), _ = cv2.getTextSize(tid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.circle(frame, (cx, cy), max(tw, th) // 2 + 8, color, -1)
    cv2.putText(frame, tid_text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_hud(frame, fps, count, paused):
    """Draw HUD overlay with FPS, object count, and status."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "CodeAlpha | Object Detection & Tracking",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 1, cv2.LINE_AA)

    # Object count
    count_text = f"Objects: {count}"
    cv2.putText(frame, count_text, (w - 340, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 180, 255), 1, cv2.LINE_AA)

    # PAUSED badge
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 80, 255), 3, cv2.LINE_AA)

    # Controls hint at bottom
    hint = "Q: Quit  |  S: Screenshot  |  P: Pause"
    cv2.putText(frame, hint, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

def main(source):
    # Load model 
    print("\n✦ Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("✦ Model loaded!\n")
    print("Controls:")
    print("  Q → Quit")
    print("  S → Save screenshot")
    print("  P → Pause / Resume\n")

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Could not open video source. Check your webcam or file path.")
        return

    # Screenshot folder
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # State
    paused    = False
    prev_time = time.time()
    fps       = 0
    frame_count = 0

    print("✦ Running... press Q to quit.\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("✦ End of video or no frame received.")
                break

            frame_count += 1

            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # YOLOv8 tracking
            # persist=True keeps track IDs consistent across frames
            results = model.track(
                frame,
                persist=True,
                conf=CONFIDENCE,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # Draw detections
            object_count = 0
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    # Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Class & confidence
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = model.names[cls_id]

                    # Track ID (may be None if tracking lost)
                    track_id = int(box.id[0]) if box.id is not None else 0
                    color    = get_color(track_id)

                    # Draw
                    draw_box(frame, x1, y1, x2, y2, label, color, conf)
                    draw_track_id(frame, x1, y1, x2, y2, track_id, color)
                    object_count += 1

            # HUD overlay
            draw_hud(frame, fps, object_count, paused)

        # Show frame
        cv2.imshow(WINDOW_NAME, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("✦ Quitting...")
            break
        elif key == ord('s') or key == ord('S'):
            filename = os.path.join(
                SCREENSHOT_DIR,
                f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(filename, frame)
            print(f"✦ Screenshot saved: {filename}")
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print(f"✦ {'Paused' if paused else 'Resumed'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✦ Done! Processed {frame_count} frames.")


# ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection & Tracking — CodeAlpha Task 4")
    parser.add_argument(
        "--source", default=0,
        help="Video source: 0 for webcam (default), or path to video file e.g. video.mp4"
    )
    args = parser.parse_args()

    # Convert to int if webcam index
    source = int(args.source) if str(args.source).isdigit() else args.source
    main(source)
