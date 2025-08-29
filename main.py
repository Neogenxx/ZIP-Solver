from grid_extraction import extract_puzzle
from solver import build_graph, solve_sequence
from overlay import draw_solution

# --- CONFIG ---
IMAGE_PATH = "assets/puzzle.png"
GRID_SIZE = 5           # Zip puzzle here is 5x5
DENY_REUSE_CELLS = True # prevent self-crossing by disallowing reused cells

def main():
    # Step 1–3: grid lines, walls, digits
    data = extract_puzzle(
        IMAGE_PATH,
        grid_size=GRID_SIZE,
        debug=True  # writes helpful images into ./debug and ./digits_debug
    )

    img          = data["image"]
    xs, ys       = data["xs"], data["ys"]            # grid line coordinates
    walls_h      = data["walls_h"]                   # (rows-1) x cols
    walls_v      = data["walls_v"]                   # rows x (cols-1)
    digits_map   = data["digits"]                    # {value: (cx, cy)} in cell coords

    if not digits_map:
        print("No digits found — check puzzle image or OCR settings.")
        return

    # Step 4: build wall-aware graph + solve sequence 1..N
    graph = build_graph(GRID_SIZE, walls_h, walls_v)
    path_cells = solve_sequence(graph, digits_map, GRID_SIZE, deny_reuse=DENY_REUSE_CELLS)
    if not path_cells:
        print("No valid path found.")
        return

    # Step 5: overlay
    out = draw_solution(img, xs, ys, path_cells, out_path="solution.png")
    print("Solved. Saved overlay -> solution.png")

if __name__ == "__main__":
    main()
