import cv2
import numpy as np

# === CONFIG ===
CHESSBOARD_SIZE = (11, 11)  # inner corners: 9 cols x 6 rows (standard)

image_path = 'photos/DSCF3020.JPG'  # CHANGE THIS
# === MAIN FUNCTION ===
def detect_chessboard_auto(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found!")
        return

    # Work on a copy
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Smart Preprocessing ===
    # # 1. Increase contrast
    # enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

    # # 2. Reduce noise
    # blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # # 3. Local contrast (CLAHE)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # clahe_img = clahe.apply(blurred)

    # # 4. Adaptive threshold
    # binary = cv2.adaptiveThreshold(
    #     clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY, 15, -5
    # )

    binary = gray

    # === Find Chessboard ===
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(
        binary, CHESSBOARD_SIZE, None,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_EXHAUSTIVE
    )

    # Refine corners if found
    if found:
        cv2.cornerSubPix(binary, corners, (11,11), (-1,-1), criteria)

    # === Display Result ===
    display = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if found:
        # cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        status = "CHESSBOARD FOUND!"
        color = (0, 255, 0)
        print(f"Success: {len(corners)} corners detected.")
    else:
        status = "CHESSBOARD NOT FOUND"
        color = (0, 0, 255)
        print("Failed to detect chessboard.")

    # Overlay status
    cv2.putText(display, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    cv2.imshow('Chessboard Auto-Detector', display)
    print("Press 's' to save | 'q' to quit")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = 'auto_detected_chessboard.png'
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")

    cv2.destroyAllWindows()

# === RUN ===
if __name__ == "__main__":
    detect_chessboard_auto(image_path)