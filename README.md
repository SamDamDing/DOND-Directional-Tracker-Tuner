# DOND Directional Tracker Tuner

Interactive PyQt tool for visualizing and tuning a custom tracker for the *Deal or No Deal* arcade game (or any 4×4 grid of fast-moving boxes).

It:

- Runs a YOLO model on a video of shuffling cases.
- Tracks each **logical Case ID** across time using a grid-based, direction-aware tracker.
- Lets you tweak tracking & grid parameters in real time.
- Overlays boxes, case IDs, slot indices, and tracker state on the video.
- Exports a full-resolution overlay video (perfect for demos / YouTube).

---

## Features

- **YOLO-based detections**
  - Uses a trained Ultralytics YOLO model (`yolov12n.pt`) to detect cases.
  - Filters detections to a configurable panel region.

- **Directional multi-target tracking**
  - 1 track per logical case ID (16 by default).
  - Greedy assignment with gating radius.
  - Velocity smoothing + dead reckoning when detections are missed.
  - Horizontal swap-assist heuristics to handle occlusions and close passes.
  - Final idle “snap to nearest slot” so tracks settle cleanly to grid slots.

- **Interactive GUI (PyQt5)**
  - Live frame slider + video preview.
  - Model params: `conf`, `iou`, `max_det`, `imgsz`.
  - Tracker params: gate radius, velocity smoothing, swap gap, miss tolerance, etc.
  - Editable 4×4 grid center coordinates (per-slot X/Y spins).
  - Row / column nudge tools.
  - “Build grid from detections” helper for quick calibration.

- **Video export**
  - Renders a clean overlay at original video resolution.
  - MP4 export via OpenCV (`mp4v` codec).

---

## Requirements

- Python 3.9+ (tested on 3.10 / 3.11)
- Packages:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `PyQt5`

Install (example):

```bash
pip install ultralytics opencv-python numpy PyQt5
```
<img width="2560" height="1400" alt="python_SykqiDaDue" src="https://github.com/user-attachments/assets/c4a34e47-3aec-4716-b12e-6c5d3065fdde" />
