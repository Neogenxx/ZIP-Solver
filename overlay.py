import cv2

class Overlay:
    def __init__(self, frame, path):
        self.frame = frame
        self.path = path

    def draw_solution(self):
        for i in range(len(self.path) - 1):
            cv2.line(self.frame, self.path[i], self.path[i+1], (0, 0, 255), 2)
        cv2.imwrite("solution.png", self.frame)
        print("Saved overlay to solution.png")
        return self.frame
