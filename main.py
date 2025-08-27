from screen_capture import ScreenCapture
from digit_extractor import DigitExtractor
from solver import PuzzleSolver
from overlay import Overlay

def main():
    # Step 1: Load puzzle (from image for now)
    sc = ScreenCapture("assets/puzzle.png")
    frame = sc.get_frame()

    # Step 2: Extract digits
    extractor = DigitExtractor(frame)
    digits, positions = extractor.extract_digits()

    # Step 3: Solve puzzle
    solver = PuzzleSolver(digits, positions)
    path = solver.solve()

    # Step 4: Overlay solution
    overlay = Overlay(frame, path)
    solved_frame = overlay.draw_solution()

    print("Pipeline complete. Solution overlay ready.")

if __name__ == "__main__":
    main()
