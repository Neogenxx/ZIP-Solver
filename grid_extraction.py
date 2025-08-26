import cv2
import pytesseract
import os

# Update to your Tesseract installation path if needed
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_grid_and_digits(image_path, debug=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Preprocessing to handle black circles + white digits ---
    # Step 1: Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Invert -> digits become black, background white
    inverted = cv2.bitwise_not(binary)

    # Step 3: Remove thick walls using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = {}
    cells = []

    # Create output folder for debug images
    if debug and not os.path.exists("digits_debug"):
        os.makedirs("digits_debug")

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter: skip tiny noise and very large blocks
        if 20 < w < 120 and 20 < h < 120:
            roi = cleaned[y:y+h, x:x+w]

            # OCR with digit-only config
            text = pytesseract.image_to_string(
                roi,
                config="--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
            ).strip()

            if text.isdigit():
                digits[(x, y)] = int(text)

            cells.append((x, y, w, h))

            if debug:
                cv2.imwrite(f"digits_debug/cell_{i}_{text}.png", roi)

    return digits, cells, img
