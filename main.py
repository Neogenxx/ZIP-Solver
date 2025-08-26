from grid_extraction import extract_grid_and_digits
from solver import build_graph, solve_zip
import cv2

def main():
    # Extract digits
    digits, cells, img = extract_grid_and_digits("sample_puzzle.png", debug=True)

    print("Detected digits:", digits)

    # Build graph + solve
    G = build_graph(digits)
    solution = solve_zip(G)

    print("Solution path:", [G.nodes[n]['value'] for n in solution])

    # Draw solution path
    for i in range(len(solution) - 1):
        x1, y1 = solution[i]
        x2, y2 = solution[i+1]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Solution", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
