import cv2
import os

class DigitExtractor:
    def __init__(self, frame):
        self.frame = frame
        os.makedirs("digits", exist_ok=True)

    def extract_digits(self):
        # Placeholder ROIs — replace with automatic grid detection
        rois = [
            (200, 150, 40, 40),
            (250, 200, 40, 40),
            (300, 50, 40, 40),
            (280, 400, 40, 40)
        ]
        digits, positions = [], []
        for i, (x, y, w, h) in enumerate(rois, 1):
            digit_img = self.frame[y:y+h, x:x+w]
            cv2.imwrite(f"digits/digit_{i}.png", digit_img)
            digits.append(digit_img)
            positions.append((x, y))
        return digits, positions
