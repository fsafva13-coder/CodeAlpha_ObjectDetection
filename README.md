# 🎯 Object Detection & Tracking
### CodeAlpha AI Internship | Task 4

A real-time object detection and tracking system built with **YOLOv8** and **OpenCV**. Detects 80+ object classes from a webcam or video file, assigns persistent tracking IDs, and displays results with a clean HUD overlay.

---

## ✨ Features
- 🎯 Real-time detection of 80+ object classes using YOLOv8
- 🔢 Persistent object tracking with unique IDs (ByteTrack)
- 🌈 Color-coded bounding boxes per tracking ID
- 📊 Live FPS counter and object count HUD
- 📸 Screenshot capture (press S)
- ⏸️ Pause / Resume (press P)
- 📁 Works with webcam or any video file

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| `YOLOv8 (ultralytics)` | Pre-trained object detection model |
| `OpenCV` | Video capture, frame processing & display |
| `ByteTrack` | Object tracking algorithm (built into ultralytics) |
| `Python 3.8+` | Core language |

---

## 🧠 How It Works
1. Video frames are captured from webcam or file using OpenCV
2. Each frame is passed through YOLOv8n (nano) for fast object detection
3. ByteTrack assigns consistent tracking IDs across frames
4. Bounding boxes, labels, confidence scores, and track IDs are drawn
5. A HUD overlay shows FPS, object count, and controls

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/fsafva13-coder/CodeAlpha_ObjectDetection
cd CodeAlpha_ObjectDetection
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run with webcam
```bash
python app.py
```

### Step 4 — Or run with a video file
```bash
python app.py --source video.mp4
```

---

## ⌨️ Controls
| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save screenshot to `/screenshots` folder |
| `P` | Pause / Resume |

---

## 📁 Project Structure
```
CodeAlpha_ObjectDetection/
│
├── app.py              # Main detection & tracking script
├── requirements.txt    # Python dependencies
├── screenshots/        # Saved screenshots (auto-created)
└── README.md           # Project documentation
```

---

## 🖼️ Screenshots
> Tested on a street traffic video from Pixabay.com
![screenshot_20260311_144544](https://github.com/user-attachments/assets/3915b656-074f-4364-8864-2d502636ec7e)
![screenshot_20260311_144631](https://github.com/user-attachments/assets/ec3117ea-811c-42ed-863d-274cba46a6fd)
![screenshot_20260311_144707](https://github.com/user-attachments/assets/728591e4-efa2-4d31-ac37-1fe0aa520d2a)

📹 Full demo video available on LinkedIn

---

## 🔗 Project Links
📂 GitHub: [CodeAlpha_ObjectDetection](https://github.com/fsafva13-coder/CodeAlpha_ObjectDetection)

---

## 👤 Author
**Fathima Safva** — CodeAlpha AI Intern
GitHub: [@fsafva13-coder](https://github.com/fsafva13-coder)
LinkedIn: [Fathima Safva](https://linkedin.com/in/fathima-safva-578294315)

---

## 📄 License
This project is built as part of the **CodeAlpha AI Internship Program**.
Website: [www.codealpha.tech](http://www.codealpha.tech)
