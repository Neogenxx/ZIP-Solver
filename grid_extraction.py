import cv2
import numpy as np
import pytesseract
import os
from typing import Dict, List, Tuple

# Set this if Tesseract isn't on PATH (Windows default shown)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def _ensure_dirs():
    os.makedirs("debug", exist_ok=True)
    os.makedirs("digits_debug", exist_ok=True)

def _binarize(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Invert so black walls/lines become white; easier to count coverage
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return gray, bw

def _find_grid_lines(bw, grid_size: int) -> Tuple[List[int], List[int]]:
    """
    Estimate exact grid line positions (xs, ys) by snapping each expected line
    to the strongest vertical/horizontal response near its equally-spaced guess.
    """
    h, w = bw.shape
    guess_xs = [int(round(i * w / grid_size)) for i in range(grid_size + 1)]
    guess_ys = [int(round(i * h / grid_size)) for i in range(grid_size + 1)]

    # integrated sums
    col_sum = np.sum(bw // 255, axis=0)  # white count per column
    row_sum = np.sum(bw // 255, axis=1)  # white count per row

    def snap(guesses, arr, window):
        coords = []
        for g in guesses:
            a = max(0, g - window)
            b = min(len(arr) - 1, g + window)
            seg = arr[a:b + 1]
            off = int(np.argmax(seg))
            coords.append(a + off)
        return coords

    # window ~ 8% of cell size
    win_x = max(4, int(0.08 * w / grid_size))
    win_y = max(4, int(0.08 * h / grid_size))
    xs = snap(guess_xs, col_sum, win_x)
    ys = snap(guess_ys, row_sum, win_y)
    return xs, ys

def _detect_walls(bw, xs, ys) -> Tuple[List[List[bool]], List[List[bool]]]:
    """
    Decide where thick black walls exist between adjacent cells.
    Returns:
      walls_h[r][c]: wall between (r,c) and (r+1,c)  -> size (rows-1) x cols
      walls_v[r][c]: wall between (r,c) and (r,c+1)  -> size rows x (cols-1)
    """
    rows = len(ys) - 1
    cols = len(xs) - 1
    walls_h = [[False] * cols for _ in range(rows - 1)]
    walls_v = [[False] * (cols - 1) for _ in range(rows)]

    for r in range(rows - 1):
        yb = ys[r + 1]  # border y between r and r+1
        for c in range(cols):
            x1, x2 = xs[c], xs[c + 1]
            # central horizontal band across the border; height small
            pad_x = int(0.18 * (x2 - x1))
            band = bw[max(0, yb - 3): yb + 3, x1 + pad_x: x2 - pad_x]
            cover = band.mean() / 255.0
            # thin grid line -> low cover; thick wall -> high cover
            walls_h[r][c] = cover > 0.25

    for r in range(rows):
        for c in range(cols - 1):
            xb = xs[c + 1]  # border x between c and c+1
            y1, y2 = ys[r], ys[r + 1]
            pad_y = int(0.18 * (y2 - y1))
            band = bw[y1 + pad_y: y2 - pad_y, max(0, xb - 3): xb + 3]
            cover = band.mean() / 255.0
            walls_v[r][c] = cover > 0.25

    return walls_h, walls_v

def _ocr_digit_from_circle(gray, x, y, r, tag: str) -> int:
    # Crop a square ROI around the circle
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = x + r, y + r
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0

    # Threshold and remove circle ring: keep bright digit only
    roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Largest bright blob (digit), remove small noise
    cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(roi)
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(mask, [biggest], -1, 255, -1)
        roi = cv2.bitwise_and(roi, mask)

    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join("digits_debug", f"{tag}.png"), roi)

    config = "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789"
    txt = pytesseract.image_to_string(roi, config=config).strip()
    return int(txt) if txt.isdigit() else 0

def _detect_digits(img_bgr, gray, xs, ys) -> Dict[int, Tuple[int, int]]:
    """
    Find the black circular markers and OCR the digit inside.
    Map each recognized digit -> (col,row) in grid cell coordinates.
    """
    rows = len(ys) - 1
    cols = len(xs) - 1
    cell_w = int(np.median(np.diff(np.array(xs))))
    cell_h = int(np.median(np.diff(np.array(ys))))
    circle_min_r = int(0.22 * min(cell_w, cell_h))
    circle_max_r = int(0.50 * min(cell_w, cell_h))

    # HoughCircles on slightly blurred gray
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=int(0.8 * min(cell_w, cell_h)),
        param1=120, param2=25,
        minRadius=circle_min_r, maxRadius=circle_max_r
    )

    digits: Dict[int, Tuple[int, int]] = {}
    if circles is None:
        return digits

    circles = np.round(circles[0]).astype(int)
    for i, (cx, cy, r) in enumerate(circles):
        val = _ocr_digit_from_circle(gray, cx, cy, r, tag=f"circle_{i}")
        if not (1 <= val <= 9):
            continue

        # Map center -> cell (col,row)
        col = max(0, min(cols - 1, int(np.searchsorted(xs, cx) - 1)))
        row = max(0, min(rows - 1, int(np.searchsorted(ys, cy) - 1)))
        # Keep highest confidence per value: prefer nearer to cell center
        if val not in digits:
            digits[val] = (col, row)
        else:
            prev = digits[val]
            pcx = (xs[prev[0]] + xs[prev[0] + 1]) // 2
            pcy = (ys[prev[1]] + ys[prev[1] + 1]) // 2
            d_old = (pcx - cx) ** 2 + (pcy - cy) ** 2
            ccx = (xs[col] + xs[col + 1]) // 2
            ccy = (ys[row] + ys[row + 1]) // 2
            d_new = (ccx - cx) ** 2 + (ccy - cy) ** 2
            if d_new < d_old:
                digits[val] = (col, row)

    return digits

def extract_puzzle(image_path: str, grid_size: int, debug: bool = False):
    """
    Returns:
      {
        "image": BGR image,
        "xs": [x0..xN], "ys": [y0..yN],
        "walls_h": [[...]], "walls_v": [[...]],
        "digits": {digit: (col,row)}
      }
    """
    _ensure_dirs()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray, bw = _binarize(img)
    xs, ys = _find_grid_lines(bw, grid_size)
    walls_h, walls_v = _detect_walls(bw, xs, ys)
    digits = _detect_digits(img, gray, xs, ys)

    if debug:
        dbg = img.copy()
        # draw grid lines
        for x in xs:
            cv2.line(dbg, (x, ys[0]), (x, ys[-1]), (200, 200, 0), 1)
        for y in ys:
            cv2.line(dbg, (xs[0], y), (xs[-1], y), (200, 200, 0), 1)

        # draw walls
        rows = len(ys) - 1
        cols = len(xs) - 1
        for r in range(rows - 1):
            for c in range(cols):
                if walls_h[r][c]:
                    x1, x2 = xs[c], xs[c + 1]
                    yb = ys[r + 1]
                    cv2.line(dbg, (x1 + 6, yb), (x2 - 6, yb), (0, 0, 255), 3)
        for r in range(rows):
            for c in range(cols - 1):
                if walls_v[r][c]:
                    y1, y2 = ys[r], ys[r + 1]
                    xb = xs[c + 1]
                    cv2.line(dbg, (xb, y1 + 6), (xb, y2 - 6), (255, 0, 0), 3)

        # draw digits
        for d, (cx, cy) in digits.items():
            px = (xs[cx] + xs[cx + 1]) // 2
            py = (ys[cy] + ys[cy + 1]) // 2
            cv2.circle(dbg, (px, py), 10, (0, 255, 0), -1)
            cv2.putText(dbg, str(d), (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

        cv2.imwrite("debug/grid_walls_digits.png", dbg)
        cv2.imwrite("debug/binary.png", bw)

    return {
        "image": img,
        "xs": xs,
        "ys": ys,
        "walls_h": walls_h,
        "walls_v": walls_v,
        "digits": digits
    }
