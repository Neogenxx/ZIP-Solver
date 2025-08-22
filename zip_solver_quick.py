# zip_solver_quick.py
# Quick end-to-end Zip puzzle auto-solver (single file).
# Requirements: python 3.9+, opencv-python, numpy, pytesseract, mss
# Install: pip install opencv-python numpy pytesseract mss
# Also install the Tesseract binary and ensure it's in PATH.

import time
from collections import deque
import cv2
import numpy as np
import pytesseract
from mss import mss

# ---------- Configuration ----------
# If Tesseract is not on PATH, set explicit path here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESS_CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
CAPTURE_MONITOR = 1   # monitor index for mss().monitors[1] (primary)
LOOP_DELAY = 0.2      # seconds between loop iterations
MIN_CELL_SIZE = 12    # minimum width/height in pixels for OCR attempt
BARRIER_DARK_THRESHOLD = 0.45  # fraction of very-dark pixels -> barrier

# ---------- Screen capture helpers ----------
class ScreenCapture:
    def __init__(self, monitor=CAPTURE_MONITOR):
        self.sct = mss()
        self.monitor = monitor

    def grab_full(self):
        img = np.array(self.sct.grab(self.sct.monitors[self.monitor]))
        return img[:, :, :3]  # BGR-like (actually BGRA->RGB order from mss; OpenCV uses BGR but this generally works)

    def grab_region(self, region):
        img = np.array(self.sct.grab(region))
        return img[:, :, :3]

# ---------- ROI selection ----------
def select_roi_manual(frame):
    # OpenCV selectROI expects RGB; frame from mss is BGRA-like array but works in practice.
    r = cv2.selectROI("Select Zip ROI (press ENTER/SPACE to confirm)", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Zip ROI (press ENTER/SPACE to confirm)")
    x, y, w, h = map(int, r)
    return {'top': y, 'left': x, 'width': w, 'height': h}

# ---------- Grid detection & splitting ----------
def detect_grid_and_cells(roi_bgr, approx_min_cell=MIN_CELL_SIZE):
    """
    Heuristic detection: attempt to find horizontal/vertical line bands, otherwise fall back to equal cell split.
    Returns: (warped_image, rows, cols, cell_h, cell_w, cells_list)
    cells_list is a rows×cols list of sub-images (BGR).
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology to find strong horizontal & vertical strokes (grid lines)
    horizontal = th.copy()
    vertical = th.copy()
    cols = horizontal.shape[1]
    horizontal_size = max(1, cols // 30)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, hor_kernel)
    horizontal = cv2.dilate(horizontal, hor_kernel)

    rows_img = vertical.shape[0]
    vertical_size = max(1, rows_img // 30)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, ver_kernel)
    vertical = cv2.dilate(vertical, ver_kernel)

    h_proj = np.sum(horizontal == 255, axis=1)
    v_proj = np.sum(vertical == 255, axis=0)

    def find_peaks(proj):
        peaks = []
        i = 0
        L = len(proj)
        while i < L:
            if proj[i] > 0:
                j = i
                while j < L and proj[j] > 0:
                    j += 1
                peaks.append((i, j))
                i = j
            else:
                i += 1
        return peaks

    h_peaks = find_peaks(h_proj)
    v_peaks = find_peaks(v_proj)

    H, W = roi_bgr.shape[:2]
    if len(h_peaks) < 2 or len(v_peaks) < 2:
        # fallback: equal grid split heuristic
        est_rows = max(3, H // approx_min_cell)
        est_cols = max(3, W // approx_min_cell)
        cell_h = H / est_rows
        cell_w = W / est_cols
        rows = est_rows
        cols = est_cols
    else:
        h_centers = [ (s+e)//2 for s,e in h_peaks ]
        v_centers = [ (s+e)//2 for s,e in v_peaks ]
        rows = max(1, len(h_centers)-1)
        cols = max(1, len(v_centers)-1)
        cell_h = H / rows
        cell_w = W / cols

    # split cells
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y1 = int(r * cell_h)
            y2 = int(min(H, (r+1) * cell_h))
            x1 = int(c * cell_w)
            x2 = int(min(W, (c+1) * cell_w))
            if y2 - y1 <= 0 or x2 - x1 <= 0:
                # degenerate: create tiny transparent cell
                row_cells.append(roi_bgr[y1:y1+1, x1:x1+1])
            else:
                row_cells.append(roi_bgr[y1:y2, x1:x2])
        cells.append(row_cells)

    return roi_bgr, rows, cols, cell_h, cell_w, cells

# ---------- OCR helper ----------
def read_cell_number(cell_bgr):
    """Return (intval or None, confidence_est). Uses Tesseract with psm 10 and whitelist digits."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    if h < MIN_CELL_SIZE or w < MIN_CELL_SIZE:
        return None, 0
    # Resize to help OCR
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If background darker than text invert
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
    # crop central area to avoid grid lines
    pad = int(0.12 * min(th.shape))
    if pad > 0:
        crop = th[pad:-pad, pad:-pad]
        if crop.size == 0:
            crop = th
    else:
        crop = th
    txt = pytesseract.image_to_string(crop, config=TESS_CONFIG)
    digits = ''.join(ch for ch in txt if ch.isdigit())
    if digits:
        try:
            return int(digits), 90
        except ValueError:
            return None, 0
    return None, 0

def read_grid(cells):
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    grid_vals = [[None]*cols for _ in range(rows)]
    conf = [[0]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            val,cf = read_cell_number(cells[r][c])
            grid_vals[r][c] = val
            conf[r][c] = cf
    return grid_vals, conf

# ---------- Barrier detection ----------
def is_barrier(cell_bgr, dark_thresh=40, ratio_thresh=BARRIER_DARK_THRESHOLD):
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    pct_dark = float(np.mean(gray < dark_thresh))
    return pct_dark >= ratio_thresh

def blocked_cells_map(cells):
    blocked = set()
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            if is_barrier(cells[r][c]):
                blocked.add((r,c))
    return blocked

# ---------- Pathfinding (BFS) ----------
def bfs_shortest(start, goal, blocked, rows, cols):
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    while q:
        cur = q.popleft()
        for dr,dc in dirs:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nxt not in blocked and nxt not in parent:
                parent[nxt] = cur
                if nxt == goal:
                    # reconstruct path
                    path = [goal]
                    p = cur
                    while p is not None:
                        path.append(p)
                        p = parent[p]
                    path.reverse()
                    return path
                q.append(nxt)
    return None

def solve_grid(grid_vals, blocked):
    # map value to (r,c)
    pos = {}
    rows = len(grid_vals)
    cols = len(grid_vals[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            v = grid_vals[r][c]
            if v is not None:
                pos[v] = (r,c)
    if not pos:
        return None, "No numbers found"
    seq = sorted(pos.keys())
    full_path = []
    for i in range(len(seq)-1):
        a = pos[seq[i]]
        b = pos[seq[i+1]]
        seg = bfs_shortest(a, b, blocked, rows, cols)
        if seg is None:
            return None, f"No path between {seq[i]} and {seq[i+1]}"
        # append avoiding duplicate join
        if full_path and full_path[-1] == seg[0]:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)
    return full_path, None

# ---------- Visualization ----------
def compute_centers(rows, cols, cell_h, cell_w):
    centers = [[None]*cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            centers[r][c] = (cx, cy)
    return centers

def draw_path_on_image(warped, path, rows, cols, cell_h, cell_w):
    out = warped.copy()
    centers = compute_centers(rows, cols, cell_h, cell_w)
    if not path:
        return out
    pts = [centers[r][c] for (r,c) in path]
    N = len(pts)
    for i in range(len(pts)-1):
        t = i / max(1, N-2)
        b = 0
        g = int(255 * (1 - t))
        r = int(255 * t)
        cv2.line(out, pts[i], pts[i+1], (b,g,r), thickness=6, lineType=cv2.LINE_AA)
    for p in pts:
        cv2.circle(out, p, 4, (255,255,255), -1)
    return out

# ---------- Main loop ----------
def main_loop():
    cap = ScreenCapture()
    frame = cap.grab_full()
    roi = select_roi_manual(frame)
    print("ROI selected:", roi)
    while True:
        start = time.time()
        img = cap.grab_region(roi)
        warped, rows, cols, cell_h, cell_w, cells = detect_grid_and_cells(img)
        grid_vals, conf_map = read_grid(cells)
        blocked = blocked_cells_map(cells)

        path, err = solve_grid(grid_vals, blocked)
        if err:
            preview = warped.copy()
            cv2.putText(preview, err, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            preview = draw_path_on_image(warped, path, rows, cols, cell_h, cell_w)
            cv2.putText(preview, f"Steps: {len(path)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Zip Solver — Preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elapsed = time.time() - start
        if elapsed < LOOP_DELAY:
            time.sleep(LOOP_DELAY - elapsed)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
