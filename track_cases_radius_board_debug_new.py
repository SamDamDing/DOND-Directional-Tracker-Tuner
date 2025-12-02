"""
DOND Directional Tracker Tuner
==============================

Small PyQt GUI that:

1. Loads a YOLO model (Ultralytics) trained to detect the DOND cases.
2. Runs detections on a source video and caches per-frame detections.
3. Tracks each logical case ID across time using a grid-based,
   direction-aware tracker.
4. Lets you interactively tune tracking parameters and grid centers.
5. Displays overlays in a resizable video preview.
6. Exports a full-resolution overlay video for demonstrations / YouTube.

Usage
-----
- Adjust MODEL_PATH and VIDEO_IN below, or just edit them from code.
- Run:
    python dond_tracker_gui.py

Requirements
-----------
- Python 3.9+
- ultralytics
- opencv-python
- numpy
- PyQt5

Notes
-----
- This file is meant to be self-contained for demos and public release.
- The tracker is deliberately simple and interpretable; see the
  Directional Tracker section for details.
"""

import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5 import QtWidgets, QtCore, QtGui


# ----------------- CONFIG -----------------
MODEL_PATH = r"runs/detect/train_annotated/weights/best.pt"
VIDEO_IN = r"Assets/DOND_RAW_SHUFFLE_01_BW.mp4"

# Initial 4x4 grid of slot centers in *image* coordinates
GRID_CENTERS_INIT: List[Tuple[float, float]] = [
    (440.0, 205), (790, 205), (1150, 205), (1500, 205),
    (440.0, 435), (790, 435), (1150, 435), (1500, 435),
    (440.0, 660), (790, 660), (1150, 660), (1500, 660),
    (440.0, 890), (790, 890), (1150, 890), (1500, 890),
]

NUM_CASES = len(GRID_CENTERS_INIT)
NUM_ROWS = 4
NUM_COLS = 4
assert NUM_CASES == NUM_ROWS * NUM_COLS

# Slot index (0..15) -> logical Case ID (as shown in-game)
SLOT_TO_CASE_ID: Dict[int, int] = {
    0: 10,
    1: 16,
    2: 3,
    3: 11,
    4: 9,
    5: 2,
    6: 12,
    7: 7,
    8: 14,
    9: 4,
    10: 15,
    11: 6,
    12: 13,
    13: 8,
    14: 1,
    15: 5,
}

# Panel bounds (for detection filtering + clamping) based on initial grid
xs = [x for x, _ in GRID_CENTERS_INIT]
ys = [y for _, y in GRID_CENTERS_INIT]
PANEL_X_MIN = min(xs) - 120
PANEL_X_MAX = max(xs) + 120
PANEL_Y_MIN = min(ys) - 120
PANEL_Y_MAX = max(ys) + 120

# Canonical case size at calibration resolution (used to draw boxes)
BASE_IMG_W = 1920.0
BASE_IMG_H = 1080.0
BASE_CASE_W = 235.0
BASE_CASE_H = 185.0


# ----------------- UTILS -----------------
def color_for_case(case_id: int) -> Tuple[int, int, int]:
    """Simple deterministic color generator based on case_id."""
    return (
        (37 * (case_id % 7)) % 256,
        (17 * (case_id % 11)) % 256,
        (29 * (case_id % 13)) % 256,
    )


def clamp_to_panel(x: float, y: float) -> Tuple[float, float]:
    """Clamp (x, y) to the pre-defined panel bounds."""
    x = max(PANEL_X_MIN, min(PANEL_X_MAX, x))
    y = max(PANEL_Y_MIN, min(PANEL_Y_MAX, y))
    return x, y


def snap_box_to_case_size(
    cx: float, cy: float, w_img: int, h_img: int
) -> Tuple[int, int, int, int]:
    """
    Snap a bounding box around center (cx, cy) using a fixed case size
    scaled to the current image resolution.
    """
    scale_w = w_img / BASE_IMG_W
    scale_h = h_img / BASE_IMG_H
    case_w = BASE_CASE_W * scale_w
    case_h = BASE_CASE_H * scale_h

    x1 = cx - case_w / 2.0
    y1 = cy - case_h / 2.0
    x2 = cx + case_w / 2.0
    y2 = cy + case_h / 2.0

    x1 = max(0.0, min(float(w_img - 1), x1))
    y1 = max(0.0, min(float(h_img - 1), y1))
    x2 = max(0.0, min(float(w_img), x2))
    y2 = max(0.0, min(float(h_img), y2))

    return int(x1), int(y1), int(x2), int(y2)


def nearest_slot(
    cx: float, cy: float, centers: List[Tuple[float, float]]
) -> int:
    """Return index of slot center nearest to (cx, cy)."""
    best_idx = 0
    best_d2 = float("inf")
    for i, (gx, gy) in enumerate(centers):
        dx = cx - gx
        dy = cy - gy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


def _normalize(dx: float, dy: float) -> Tuple[float, float, float]:
    """Return (ux, uy, mag) where (ux, uy) is normalized (dx, dy)."""
    mag = (dx * dx + dy * dy) ** 0.5
    if mag < 1e-6:
        return 0.0, 0.0, 0.0
    return dx / mag, dy / mag, mag


def get_video_fps(path: str, default: float = 60.0) -> float:
    """Read FPS from a video file; fall back to default on failure."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


# ----------------- DIRECTIONAL TRACKER -----------------
@dataclass
class DirectionParams:
    """
    Parameters controlling the directional tracker behavior.

    Note:
    - move_start_thresh is currently not used in the main logic but kept
      in the UI so you can experiment / extend easily.
    """
    move_start_thresh: float = 15.0
    move_stop_speed: float = 5.0
    gate_radius: float = 160.0
    dir_cosine_weight: float = 55.0
    vel_alpha: float = 0.65
    max_miss: int = 15
    slot_block_frames: int = 8

    enable_slot_block: bool = True
    enable_horizontal_swap_assist: bool = True
    hswap_max_gap_x: float = 120.0  # max X distance for swap assist


@dataclass
class TrajTrack:
    """Single track state for one logical case ID."""
    case_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    miss: int = 0
    phase: str = "idle"  # "idle" or "moving"
    move_dir_x: float = 0.0
    move_dir_y: float = 0.0
    move_speed: float = 0.0

    slot_idx: int = 0
    slot_block: Optional[int] = None
    slot_block_left: int = 0


def run_directional_tracking(
    frames_data: List[Dict[str, Any]],
    params: DirectionParams,
    grid_centers: List[Tuple[float, float]],
) -> List[List[Dict[str, Any]]]:
    """
    Track each case across all frames_data.

    frames_data: list of dicts with keys:
        - "img": np.ndarray HxWx3 (BGR)
        - "detections": list of dicts with "cx", "cy", "bbox"
    """
    n_frames = len(frames_data)
    if n_frames == 0:
        return []

    gate_r2 = params.gate_radius * params.gate_radius

    # Initialize one track per slot at t=0
    tracks: List[TrajTrack] = []
    for slot_idx in range(len(grid_centers)):
        gx, gy = grid_centers[slot_idx]
        case_id = SLOT_TO_CASE_ID.get(slot_idx, slot_idx)
        tracks.append(
            TrajTrack(
                case_id=case_id,
                x=float(gx),
                y=float(gy),
                slot_idx=slot_idx,
            )
        )

    all_states: List[List[Dict[str, Any]]] = []

    for frame_idx in range(n_frames):
        dets = frames_data[frame_idx]["detections"]
        dcx = (
            np.array([d["cx"] for d in dets], dtype=np.float32)
            if dets else np.zeros((0,), np.float32)
        )
        dcy = (
            np.array([d["cy"] for d in dets], dtype=np.float32)
            if dets else np.zeros((0,), np.float32)
        )

        # Predicted positions from last frame (dead-reckoning)
        preds = [(tr.x + tr.vx, tr.y + tr.vy) for tr in tracks]

        # Decay slot-block timers
        if params.enable_slot_block:
            for tr in tracks:
                if tr.slot_block_left > 0:
                    tr.slot_block_left -= 1
                    if tr.slot_block_left == 0:
                        tr.slot_block = None
        else:
            for tr in tracks:
                tr.slot_block = None
                tr.slot_block_left = 0

        # Build candidate assignments: (cost, track_idx, det_idx, cand_slot)
        candidates = []
        for ti, tr in enumerate(tracks):
            px, py = preds[ti]
            for di in range(len(dets)):
                dx = float(dcx[di] - px)
                dy = float(dcy[di] - py)
                d2 = dx * dx + dy * dy
                if d2 > gate_r2:
                    continue

                cand_slot = nearest_slot(
                    float(dcx[di]), float(dcy[di]), grid_centers
                )

                if (
                    params.enable_slot_block
                    and tr.slot_block is not None
                    and cand_slot == tr.slot_block
                ):
                    continue

                cost = d2

                # Bias cost if we're in a strong movement phase
                if tr.phase == "moving":
                    ux, uy, _ = _normalize(dx, dy)
                    cos_th = ux * tr.move_dir_x + uy * tr.move_dir_y
                    if cos_th > 0.0:
                        cost -= params.dir_cosine_weight * cos_th

                candidates.append((cost, ti, di, cand_slot))

        candidates.sort(key=lambda x: x[0])
        assigned_track = [False] * len(tracks)
        assigned_det = [False] * len(dets)

        # Greedy assignment
        for cost, ti, di, cand_slot in candidates:
            if assigned_track[ti] or assigned_det[di]:
                continue
            assigned_track[ti] = True
            assigned_det[di] = True

            tr = tracks[ti]
            old_x, old_y = tr.x, tr.y
            old_slot = tr.slot_idx

            new_x = float(dcx[di])
            new_y = float(dcy[di])
            new_x, new_y = clamp_to_panel(new_x, new_y)

            dx = new_x - old_x
            dy = new_y - old_y
            ux, uy, move_mag = _normalize(dx, dy)

            # Phase / direction
            if move_mag < params.move_stop_speed:
                tr.phase = "idle"
                tr.move_dir_x = tr.move_dir_y = 0.0
                tr.move_speed = 0.0
            else:
                tr.phase = "moving"
                if tr.move_dir_x == 0.0 and tr.move_dir_y == 0.0:
                    tr.move_dir_x, tr.move_dir_y = ux, uy
                    tr.move_speed = move_mag
                else:
                    cos_th = ux * tr.move_dir_x + uy * tr.move_dir_y
                    if cos_th < 0.0:
                        tr.move_dir_x, tr.move_dir_y = ux, uy
                        tr.move_speed = move_mag
                    else:
                        # Slightly smooth direction and speed
                        mx = 0.7 * tr.move_dir_x + 0.3 * ux
                        my = 0.7 * tr.move_dir_y + 0.3 * uy
                        ndx, ndy, _ = _normalize(mx, my)
                        tr.move_dir_x, tr.move_dir_y = ndx, ndy
                        tr.move_speed = 0.7 * tr.move_speed + 0.3 * move_mag

            # Commit pos/vel
            tr.x = new_x
            tr.y = new_y
            tr.vx = params.vel_alpha * tr.vx + (1.0 - params.vel_alpha) * dx
            tr.vy = params.vel_alpha * tr.vy + (1.0 - params.vel_alpha) * dy

            tr.miss = 0
            tr.age += 1

            # Slot commit
            new_slot = cand_slot
            if new_slot != old_slot:
                if params.enable_slot_block:
                    tr.slot_block = old_slot
                    tr.slot_block_left = params.slot_block_frames
                tr.slot_idx = new_slot

        # Unmatched tracks: dead-reckoning step
        for ti, tr in enumerate(tracks):
            if assigned_track[ti]:
                continue

            tr.miss += 1
            tr.age += 1
            tr.x += tr.vx
            tr.y += tr.vy
            tr.x, tr.y = clamp_to_panel(tr.x, tr.y)

            if tr.miss > params.max_miss:
                # Damp velocity if we lost this for too long
                tr.vx *= 0.3
                tr.vy *= 0.3
                tr.phase = "idle"
                tr.move_dir_x = tr.move_dir_y = 0.0
                tr.move_speed = 0.0

        # ---------- HORIZONTAL SWAP ASSIST ----------
        if params.enable_horizontal_swap_assist:
            # Stage 1: geometry-driven "better distance" swap
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    ti_tr = tracks[i]
                    tj_tr = tracks[j]

                    si = ti_tr.slot_idx
                    sj = tj_tr.slot_idx
                    if si == sj:
                        continue

                    ri, ci = divmod(si, NUM_COLS)
                    rj, cj = divmod(sj, NUM_COLS)

                    # only adjacent horizontally in same row
                    if not (ri == rj and abs(ci - cj) == 1):
                        continue
                    if ti_tr.phase != "moving" or tj_tr.phase != "moving":
                        continue

                    if ci < cj:
                        left_tr, right_tr = ti_tr, tj_tr
                        left_slot_idx, right_slot_idx = si, sj
                    else:
                        left_tr, right_tr = tj_tr, ti_tr
                        left_slot_idx, right_slot_idx = sj, si

                    gx_l, gy_l = grid_centers[left_slot_idx]
                    gx_r, gy_r = grid_centers[right_slot_idx]

                    lx, ly = left_tr.x, left_tr.y
                    rx, ry = right_tr.x, right_tr.y

                    # must be approximately same row in Y
                    if abs(ly - ry) > 60:
                        continue

                    gap_x = abs(lx - rx)
                    if gap_x > params.hswap_max_gap_x:
                        continue

                    def d2(px, py, gx, gy):
                        dx_ = px - gx
                        dy_ = py - gy
                        return dx_ * dx_ + dy_ * dy_

                    d_left_self = d2(lx, ly, gx_l, gy_l)
                    d_left_other = d2(lx, ly, gx_r, gy_r)
                    d_right_self = d2(rx, ry, gx_r, gy_r)
                    d_right_other = d2(rx, ry, gx_l, gy_l)

                    before = d_left_self + d_right_self
                    after = d_left_other + d_right_other

                    # Only swap if total squared distance to centers improves
                    if after >= 0.8 * before:
                        continue

                    si_old, sj_old = ti_tr.slot_idx, tj_tr.slot_idx
                    ti_tr.slot_idx, tj_tr.slot_idx = sj_old, si_old

                    if params.enable_slot_block:
                        soft_block = max(1, params.slot_block_frames // 2)
                        ti_tr.slot_block = si_old
                        tj_tr.slot_block = sj_old
                        ti_tr.slot_block_left = soft_block
                        tj_tr.slot_block_left = soft_block

            # Stage 2: nearest-slot symmetry fallback
            nearest_slots = [
                nearest_slot(tr.x, tr.y, grid_centers) for tr in tracks
            ]

            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    ti_tr = tracks[i]
                    tj_tr = tracks[j]

                    si = ti_tr.slot_idx
                    sj = tj_tr.slot_idx
                    if si == sj:
                        continue

                    ri, ci = divmod(si, NUM_COLS)
                    rj, cj = divmod(sj, NUM_COLS)

                    if not (ri == rj and abs(ci - cj) == 1):
                        continue

                    ni = nearest_slots[i]
                    nj = nearest_slots[j]

                    if not (ni == sj and nj == si):
                        continue

                    gx_i, gy_i = grid_centers[si]
                    gx_j, gy_j = grid_centers[sj]

                    def d2(px, py, gx, gy):
                        dx_ = px - gx
                        dy_ = py - gy
                        return dx_ * dx_ + dy_ * dy_

                    dist_i_self = d2(ti_tr.x, ti_tr.y, gx_i, gy_i)
                    dist_j_self = d2(tj_tr.x, tj_tr.y, gx_j, gy_j)

                    # If both are already extremely close, skip
                    if dist_i_self < 25.0 and dist_j_self < 25.0:
                        continue

                    si_old, sj_old = ti_tr.slot_idx, tj_tr.slot_idx
                    ti_tr.slot_idx, tj_tr.slot_idx = sj_old, si_old

                    if params.enable_slot_block:
                        soft_block = max(1, params.slot_block_frames // 2)
                        ti_tr.slot_block = si_old
                        tj_tr.slot_block = sj_old
                        ti_tr.slot_block_left = soft_block
                        tj_tr.slot_block_left = soft_block

            # Stage 3: simple X-order crossing fallback
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    ti_tr = tracks[i]
                    tj_tr = tracks[j]

                    si = ti_tr.slot_idx
                    sj = tj_tr.slot_idx
                    if si == sj:
                        continue

                    ri, ci = divmod(si, NUM_COLS)
                    rj, cj = divmod(sj, NUM_COLS)

                    if not (ri == rj and abs(ci - cj) == 1):
                        continue

                    if ci < cj:
                        left_tr, right_tr = ti_tr, tj_tr
                    else:
                        left_tr, right_tr = tj_tr, ti_tr

                    # If the "left" track is now to the right in X, swap
                    if left_tr.x > right_tr.x:
                        si_old, sj_old = ti_tr.slot_idx, tj_tr.slot_idx
                        ti_tr.slot_idx, tj_tr.slot_idx = sj_old, si_old

                        if params.enable_slot_block:
                            soft_block = max(1, params.slot_block_frames // 2)
                            ti_tr.slot_block = si_old
                            tj_tr.slot_block = sj_old
                            ti_tr.slot_block_left = soft_block
                            tj_tr.slot_block_left = soft_block

        # ---------- FINAL IDLE SNAP TO NEAREST SLOT ----------
        for tr in tracks:
            if tr.phase != "idle" or tr.miss != 0:
                continue

            cur_si = tr.slot_idx
            gx_cur, gy_cur = grid_centers[cur_si]
            d2_cur = (tr.x - gx_cur) ** 2 + (tr.y - gy_cur) ** 2

            nearest = nearest_slot(tr.x, tr.y, grid_centers)
            if nearest == cur_si:
                continue

            gx_n, gy_n = grid_centers[nearest]
            d2_new = (tr.x - gx_n) ** 2 + (tr.y - gy_n) ** 2

            # Require significant improvement to re-assign
            if d2_new < d2_cur * 0.8:
                tr.slot_idx = nearest
                tr.slot_block = None
                tr.slot_block_left = 0

        # Snapshot per-frame state
        frame_state: List[Dict[str, Any]] = []
        for tr in tracks:
            speed = (tr.vx * tr.vx + tr.vy * tr.vy) ** 0.5
            frame_state.append(
                {
                    "case_id": tr.case_id,
                    "cx": tr.x,
                    "cy": tr.y,
                    "phase": tr.phase,
                    "miss": tr.miss,
                    "speed": speed,
                    "slot_idx": tr.slot_idx,
                }
            )
        all_states.append(frame_state)

    return all_states


# ----------------- DRAWING HELPERS -----------------
def draw_tracking_overlay(
    frame: np.ndarray,
    frame_state: List[Dict[str, Any]],
    grid_centers: List[Tuple[float, float]],
    draw_grid: bool = True,
) -> np.ndarray:
    """
    Draw tracking overlay for one frame:
    - case boxes and labels for each tracked case
    - optional grid centers
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Cases
    for cs in frame_state:
        cid = cs["case_id"]
        cx = float(cs["cx"])
        cy = float(cs["cy"])
        phase = cs["phase"]
        miss = cs["miss"]
        speed = cs["speed"]
        slot_idx = cs.get("slot_idx", -1)

        x1i, y1i, x2i, y2i = snap_box_to_case_size(cx, cy, w, h)
        color = color_for_case(cid)
        label = f"Case {cid} S{slot_idx} [{phase}] v={speed:.1f} m={miss}"

        cv2.circle(out, (int(cx), int(cy)), 3, color, -1)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            out,
            (x1i, y1i - th - 4),
            (x1i + tw, y1i),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1i, y1i - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if draw_grid:
        for idx, (gx, gy) in enumerate(grid_centers):
            cv2.circle(out, (int(gx), int(gy)), 3, (255, 255, 255), -1)
            cv2.putText(
                out,
                f"S{idx}",
                (int(gx) - 10, int(gy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return out


# ----------------- GUI -----------------
class TrackerGUI(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DOND Directional Tracker Tuner")

        # YOLO model + parameters
        self.model = YOLO(MODEL_PATH)
        self.model_conf = 0.12
        self.model_iou = 0.45
        self.model_maxdet = 64
        self.model_imgsz = 640

        # Data cache
        self.frames_data: List[Dict[str, Any]] = []
        self.tracking_results: List[List[Dict[str, Any]]] = []
        self.current_params = DirectionParams()

        # Grid centers (editable)
        self.grid_centers: List[Tuple[float, float]] = [
            (float(x), float(y)) for (x, y) in GRID_CENTERS_INIT
        ]
        self.slot_x_spins: List[QtWidgets.QDoubleSpinBox] = []
        self.slot_y_spins: List[QtWidgets.QDoubleSpinBox] = []
        self._updating_spins_from_grid = False

        self._build_ui()
        self._load_detections()
        self._run_tracking()

    # ---------- UI BUILD ----------
    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left: video preview + frame slider
        left_layout = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel("Video will appear here")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        left_layout.addWidget(self.video_label)

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        left_layout.addWidget(self.frame_slider)

        main_layout.addLayout(left_layout, stretch=3)

        # Right: controls (scrollable)
        right_layout = QtWidgets.QVBoxLayout()

        # ----- Model params -----
        model_group = QtWidgets.QGroupBox("Model params (YOLO)")
        model_layout = QtWidgets.QVBoxLayout(model_group)

        def add_model_spin(
            label: str,
            minimum: float,
            maximum: float,
            step: float,
            value: float,
            decimals: int = 0,
        ) -> QtWidgets.QDoubleSpinBox:
            hl = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(minimum, maximum)
            sb.setSingleStep(step)
            sb.setValue(value)
            sb.setDecimals(decimals)
            sb.setMaximumWidth(90)
            hl.addWidget(lbl)
            hl.addWidget(sb)
            model_layout.addLayout(hl)
            return sb

        self.spin_conf = add_model_spin(
            "conf", 0.0, 1.0, 0.01, self.model_conf, 2
        )
        self.spin_iou = add_model_spin(
            "iou", 0.0, 1.0, 0.01, self.model_iou, 2
        )

        self.spin_maxdet = QtWidgets.QSpinBox()
        self.spin_maxdet.setRange(1, 512)
        self.spin_maxdet.setValue(self.model_maxdet)
        hl_md = QtWidgets.QHBoxLayout()
        hl_md.addWidget(QtWidgets.QLabel("max_det"))
        hl_md.addWidget(self.spin_maxdet)
        model_layout.addLayout(hl_md)

        self.spin_imgsz = QtWidgets.QSpinBox()
        self.spin_imgsz.setRange(256, 1920)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(self.model_imgsz)
        hl_is = QtWidgets.QHBoxLayout()
        hl_is.addWidget(QtWidgets.QLabel("imgsz"))
        hl_is.addWidget(self.spin_imgsz)
        model_layout.addLayout(hl_is)

        self.btn_reload_dets = QtWidgets.QPushButton(
            "Re-run detections with model params"
        )
        self.btn_reload_dets.clicked.connect(
            self.on_reload_detections_clicked
        )
        model_layout.addWidget(self.btn_reload_dets)

        right_layout.addWidget(model_group)

        # ----- Tracker params -----
        params_group = QtWidgets.QGroupBox("Tracker params")
        params_layout = QtWidgets.QVBoxLayout(params_group)

        def add_param_spin(
            label: str,
            minimum: float,
            maximum: float,
            step: float,
            value: float,
            decimals: int = 0,
        ) -> QtWidgets.QDoubleSpinBox:
            hl = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(minimum, maximum)
            sb.setSingleStep(step)
            sb.setValue(value)
            sb.setDecimals(decimals)
            sb.setMaximumWidth(90)
            hl.addWidget(lbl)
            hl.addWidget(sb)
            params_layout.addLayout(hl)
            return sb

        self.spin_move_start = add_param_spin(
            "MOVE_START",
            1.0,
            80.0,
            1.0,
            self.current_params.move_start_thresh,
            1,
        )
        self.spin_move_stop = add_param_spin(
            "MOVE_STOP",
            0.0,
            60.0,
            1.0,
            self.current_params.move_stop_speed,
            1,
        )
        self.spin_gate_radius = add_param_spin(
            "GATE_RADIUS",
            10.0,
            400.0,
            5.0,
            self.current_params.gate_radius,
            1,
        )
        self.spin_dir_weight = add_param_spin(
            "DIR_WEIGHT",
            0.0,
            200.0,
            5.0,
            self.current_params.dir_cosine_weight,
            1,
        )
        self.spin_vel_alpha = add_param_spin(
            "VEL_ALPHA",
            0.0,
            1.0,
            0.05,
            self.current_params.vel_alpha,
            2,
        )

        self.spin_max_miss = QtWidgets.QSpinBox()
        self.spin_max_miss.setRange(0, 120)
        self.spin_max_miss.setValue(self.current_params.max_miss)
        hl_mm = QtWidgets.QHBoxLayout()
        hl_mm.addWidget(QtWidgets.QLabel("MAX_MISS"))
        hl_mm.addWidget(self.spin_max_miss)
        params_layout.addLayout(hl_mm)

        self.spin_slot_block = QtWidgets.QSpinBox()
        self.spin_slot_block.setRange(0, 60)
        self.spin_slot_block.setValue(
            self.current_params.slot_block_frames
        )
        hl_sb = QtWidgets.QHBoxLayout()
        hl_sb.addWidget(QtWidgets.QLabel("BLOCK_FRAMES"))
        hl_sb.addWidget(self.spin_slot_block)
        params_layout.addLayout(hl_sb)

        self.spin_hswap_gap_x = add_param_spin(
            "HSWAP_GAP_X",
            10.0,
            400.0,
            5.0,
            self.current_params.hswap_max_gap_x,
            1,
        )

        self.chk_slot_block = QtWidgets.QCheckBox("Enable slot back-blocking")
        self.chk_slot_block.setChecked(True)
        params_layout.addWidget(self.chk_slot_block)

        self.chk_hswap = QtWidgets.QCheckBox(
            "Enable horizontal swap assist"
        )
        self.chk_hswap.setChecked(True)
        params_layout.addWidget(self.chk_hswap)

        right_layout.addWidget(params_group)

        # ----- Grid centers -----
        grid_group = QtWidgets.QGroupBox("Grid centers (editable)")
        grid_layout = QtWidgets.QVBoxLayout(grid_group)

        for idx, (gx, gy) in enumerate(self.grid_centers):
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(f"S{idx}")
            lbl.setFixedWidth(24)
            spin_x = QtWidgets.QDoubleSpinBox()
            spin_y = QtWidgets.QDoubleSpinBox()
            for sb in (spin_x, spin_y):
                sb.setRange(0, 3000)
                sb.setDecimals(1)
                sb.setSingleStep(1.0)
                sb.setMaximumWidth(80)
            spin_x.setValue(gx)
            spin_y.setValue(gy)

            self.slot_x_spins.append(spin_x)
            self.slot_y_spins.append(spin_y)

            spin_x.valueChanged.connect(
                self._make_slot_spin_callback(idx, True)
            )
            spin_y.valueChanged.connect(
                self._make_slot_spin_callback(idx, False)
            )

            row.addWidget(lbl)
            row.addWidget(QtWidgets.QLabel("X"))
            row.addWidget(spin_x)
            row.addWidget(QtWidgets.QLabel("Y"))
            row.addWidget(spin_y)
            grid_layout.addLayout(row)

        # Grid tools (row/col nudges, auto build)
        tools_group = QtWidgets.QGroupBox("Grid tools")
        tools_layout = QtWidgets.QVBoxLayout(tools_group)

        # Row nudge
        row_layout = QtWidgets.QHBoxLayout()
        self.spin_row_idx = QtWidgets.QSpinBox()
        self.spin_row_idx.setRange(0, NUM_ROWS - 1)
        self.spin_row_idx.setPrefix("Row ")
        self.spin_row_idx.setValue(0)

        self.spin_row_delta = QtWidgets.QDoubleSpinBox()
        self.spin_row_delta.setRange(-200.0, 200.0)
        self.spin_row_delta.setDecimals(1)
        self.spin_row_delta.setSingleStep(1.0)
        self.spin_row_delta.setValue(2.0)

        btn_row_up = QtWidgets.QPushButton("Row up")
        btn_row_down = QtWidgets.QPushButton("Row down")
        btn_row_up.clicked.connect(self.on_row_up)
        btn_row_down.clicked.connect(self.on_row_down)

        row_layout.addWidget(self.spin_row_idx)
        row_layout.addWidget(QtWidgets.QLabel("ΔY"))
        row_layout.addWidget(self.spin_row_delta)
        row_layout.addWidget(btn_row_up)
        row_layout.addWidget(btn_row_down)
        tools_layout.addLayout(row_layout)

        # Column nudge
        col_layout = QtWidgets.QHBoxLayout()
        self.spin_col_idx = QtWidgets.QSpinBox()
        self.spin_col_idx.setRange(0, NUM_COLS - 1)
        self.spin_col_idx.setPrefix("Col ")
        self.spin_col_idx.setValue(0)

        self.spin_col_delta = QtWidgets.QDoubleSpinBox()
        self.spin_col_delta.setRange(-200.0, 200.0)
        self.spin_col_delta.setDecimals(1)
        self.spin_col_delta.setSingleStep(1.0)
        self.spin_col_delta.setValue(2.0)

        btn_col_left = QtWidgets.QPushButton("Col left")
        btn_col_right = QtWidgets.QPushButton("Col right")
        btn_col_left.clicked.connect(self.on_col_left)
        btn_col_right.clicked.connect(self.on_col_right)

        col_layout.addWidget(self.spin_col_idx)
        col_layout.addWidget(QtWidgets.QLabel("ΔX"))
        col_layout.addWidget(self.spin_col_delta)
        col_layout.addWidget(btn_col_left)
        col_layout.addWidget(btn_col_right)
        tools_layout.addLayout(col_layout)

        # Build grid from detections of current frame
        btn_auto_grid = QtWidgets.QPushButton(
            "Build grid from detections (current frame)"
        )
        btn_auto_grid.clicked.connect(self.on_build_grid_from_detections)
        tools_layout.addWidget(btn_auto_grid)

        tools_layout.addStretch()
        grid_layout.addWidget(tools_group)

        # Reset grid to defaults
        btn_reset_grid = QtWidgets.QPushButton("Reset grid centers")
        btn_reset_grid.clicked.connect(self.on_reset_grid)
        grid_layout.addWidget(btn_reset_grid)

        grid_layout.addStretch()
        right_layout.addWidget(grid_group)

        # Run tracking
        self.run_btn = QtWidgets.QPushButton(
            "Run tracking with current parameters"
        )
        self.run_btn.clicked.connect(self.on_run_clicked)
        right_layout.addWidget(self.run_btn)

        # Export overlay video
        self.btn_export = QtWidgets.QPushButton("Export overlay video")
        self.btn_export.clicked.connect(self.on_export_video)
        right_layout.addWidget(self.btn_export)

        right_layout.addStretch()

        # Put right layout in a scroll area
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll.setWidget(right_widget)

        main_layout.addWidget(scroll, stretch=1)

    # ---------- GRID CALLBACKS ----------
    def _make_slot_spin_callback(self, idx: int, is_x: bool):
        def cb(val: float) -> None:
            if self._updating_spins_from_grid:
                return
            x, y = self.grid_centers[idx]
            if is_x:
                x = float(val)
            else:
                y = float(val)
            self.grid_centers[idx] = (x, y)
            self._run_tracking()

        return cb

    def _sync_spins_from_grid(self) -> None:
        self._updating_spins_from_grid = True
        for idx, (gx, gy) in enumerate(self.grid_centers):
            self.slot_x_spins[idx].setValue(gx)
            self.slot_y_spins[idx].setValue(gy)
        self._updating_spins_from_grid = False

    def on_row_up(self) -> None:
        dy = -float(self.spin_row_delta.value())
        self._nudge_row(dy)

    def on_row_down(self) -> None:
        dy = float(self.spin_row_delta.value())
        self._nudge_row(dy)

    def _nudge_row(self, dy: float) -> None:
        row = int(self.spin_row_idx.value())
        if not (0 <= row < NUM_ROWS):
            return
        for c in range(NUM_COLS):
            idx = row * NUM_COLS + c
            x, y = self.grid_centers[idx]
            self.grid_centers[idx] = (x, y + dy)
        self._sync_spins_from_grid()
        self._run_tracking()

    def on_col_left(self) -> None:
        dx = -float(self.spin_col_delta.value())
        self._nudge_col(dx)

    def on_col_right(self) -> None:
        dx = float(self.spin_col_delta.value())
        self._nudge_col(dx)

    def _nudge_col(self, dx: float) -> None:
        col = int(self.spin_col_idx.value())
        if not (0 <= col < NUM_COLS):
            return
        for r in range(NUM_ROWS):
            idx = r * NUM_COLS + col
            x, y = self.grid_centers[idx]
            self.grid_centers[idx] = (x + dx, y)
        self._sync_spins_from_grid()
        self._run_tracking()

    def on_build_grid_from_detections(self) -> None:
        """Fill grid_centers from sorted detections of current frame."""
        if not self.frames_data:
            return
        frame_idx = self.frame_slider.value()
        if frame_idx < 0 or frame_idx >= len(self.frames_data):
            return

        dets = self.frames_data[frame_idx]["detections"]
        if not dets:
            print("No detections on current frame for grid build.")
            return

        centers = [(d["cx"], d["cy"]) for d in dets]
        centers.sort(key=lambda c: (c[1], c[0]))  # sort by y then x

        new_gc = list(self.grid_centers)
        for i in range(min(len(centers), NUM_CASES)):
            new_gc[i] = (float(centers[i][0]), float(centers[i][1]))
        self.grid_centers = new_gc

        self._sync_spins_from_grid()
        self._run_tracking()

    # ---------- DATA LOADING ----------
    def _load_detections(self) -> None:
        """Run YOLO on the video and cache detections per frame."""
        print("Running YOLO to cache detections...")
        self.frames_data.clear()

        results_gen = self.model(
            source=VIDEO_IN,
            stream=True,
            conf=float(self.model_conf),
            iou=float(self.model_iou),
            max_det=int(self.model_maxdet),
            imgsz=int(self.model_imgsz),
            verbose=True,
            device=0,  # adjust if needed (e.g. "cpu")
        )

        first_shape = None

        for r in results_gen:
            frame = r.orig_img.copy()
            if first_shape is None:
                h, w = frame.shape[:2]
                first_shape = (h, w)
                self.video_label.setMinimumSize(w // 2, h // 2)

            boxes = r.boxes
            detections = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                for bb in xyxy:
                    x1, y1, x2, y2 = bb
                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)
                    if not (
                        PANEL_X_MIN <= cx <= PANEL_X_MAX
                        and PANEL_Y_MIN <= cy <= PANEL_Y_MAX
                    ):
                        continue
                    detections.append(
                        {
                            "cx": float(cx),
                            "cy": float(cy),
                            "bbox": bb.astype(np.float32),
                        }
                    )
            self.frames_data.append(
                {
                    "img": frame,
                    "detections": detections,
                }
            )

        n = len(self.frames_data)
        print(f"Cached detections for {n} frames.")
        if n > 0:
            self.frame_slider.setMaximum(n - 1)
            self.frame_slider.setValue(0)

    # ---------- MODEL PARAM CALLBACK ----------
    def on_reload_detections_clicked(self) -> None:
        """Update model params from UI and re-run detections."""
        self.model_conf = float(self.spin_conf.value())
        self.model_iou = float(self.spin_iou.value())
        self.model_maxdet = int(self.spin_maxdet.value())
        self.model_imgsz = int(self.spin_imgsz.value())

        self._load_detections()
        self._run_tracking()

    # ---------- TRACKING ----------
    def _read_params_from_ui(self) -> None:
        """Sync DirectionParams from the UI controls."""
        p = DirectionParams()
        p.move_start_thresh = float(self.spin_move_start.value())
        p.move_stop_speed = float(self.spin_move_stop.value())
        p.gate_radius = float(self.spin_gate_radius.value())
        p.dir_cosine_weight = float(self.spin_dir_weight.value())
        p.vel_alpha = float(self.spin_vel_alpha.value())
        p.max_miss = int(self.spin_max_miss.value())
        p.slot_block_frames = int(self.spin_slot_block.value())
        p.hswap_max_gap_x = float(self.spin_hswap_gap_x.value())
        p.enable_slot_block = self.chk_slot_block.isChecked()
        p.enable_horizontal_swap_assist = self.chk_hswap.isChecked()
        self.current_params = p

    def _run_tracking(self) -> None:
        """Run tracking with current params and update preview."""
        if not self.frames_data:
            return
        print("Running directional tracking...")
        self.tracking_results = run_directional_tracking(
            self.frames_data,
            self.current_params,
            self.grid_centers,
        )
        print("Tracking finished.")
        self.on_frame_changed(self.frame_slider.value())

    # ---------- UI CALLBACKS ----------
    def on_run_clicked(self) -> None:
        self._read_params_from_ui()
        self._run_tracking()

    def on_reset_grid(self) -> None:
        self.grid_centers = [
            (float(x), float(y)) for (x, y) in GRID_CENTERS_INIT
        ]
        self._sync_spins_from_grid()
        self._run_tracking()

    def on_frame_changed(self, frame_idx: int) -> None:
        if not self.frames_data:
            return
        if frame_idx < 0 or frame_idx >= len(self.frames_data):
            return

        frame = self.frames_data[frame_idx]["img"]
        if (
            self.tracking_results
            and frame_idx < len(self.tracking_results)
        ):
            frame_state = self.tracking_results[frame_idx]
        else:
            frame_state = []

        overlay = draw_tracking_overlay(
            frame, frame_state, self.grid_centers, draw_grid=True
        )

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)

        # Scale to current label size, preserving aspect
        scaled = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def on_export_video(self) -> None:
        """Export full-resolution overlay video to MP4."""
        if not self.frames_data:
            print("No frames loaded, cannot export.")
            return
        if not self.tracking_results:
            self._read_params_from_ui()
            self._run_tracking()
            if not self.tracking_results:
                print("Tracking empty, cannot export.")
                return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export overlay video",
            "tracking.mp4",
            "MP4 Files (*.mp4);;All Files (*)",
        )
        if not filename:
            return

        fps = get_video_fps(VIDEO_IN, 30.0)
        h, w = self.frames_data[0]["img"].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("Failed to open VideoWriter for:", filename)
            return

        n_frames = min(
            len(self.frames_data), len(self.tracking_results)
        )
        print(
            f"Exporting {n_frames} frames to {filename} at {fps:.2f} fps..."
        )

        for frame_idx in range(n_frames):
            frame = self.frames_data[frame_idx]["img"]
            frame_state = self.tracking_results[frame_idx]
            frame_out = draw_tracking_overlay(
                frame, frame_state, self.grid_centers, draw_grid=True
            )
            writer.write(frame_out)

        writer.release()
        print("Export finished.")


# ----------------- MAIN -----------------
def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = TrackerGUI()
    w.resize(1400, 800)
    w.setMinimumSize(800, 600)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
