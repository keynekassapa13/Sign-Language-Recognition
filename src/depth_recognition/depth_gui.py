from depth_recognition.depth_hand_recognition import DepthHandRecogniser
from typing import Tuple
import cv2 as cv
import numpy as np


class DepthRecogniserGUI:

    def __init__(self, depth_recogniser: DepthHandRecogniser, frame_size: Tuple[int, int]):
        self.recogniser = depth_recogniser
        self.output = None
        self.width = frame_size[0]
        self.height = frame_size[1]

    def __prepare_frames(self):
        # Process Depth Image
        depth_colourmap = cv.applyColorMap(cv.convertScaleAbs(self.recogniser.depth_img, alpha=0.03), cv.COLORMAP_JET)
        # Process Distance
        distance = self.recogniser.depth_frame.get_distance(self.width // 2, self.height // 2)
        distance_str = f"Dist: {round(distance, 3)}m"
        cv.putText(self.recogniser.colour_img, distance_str,
                   (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        # Final images to output
        self.output = np.vstack((self.recogniser.colour_img, depth_colourmap))

    def draw_frames(self):
        self.__prepare_frames()
        cv.namedWindow("Real Sense", cv.WINDOW_AUTOSIZE)
        cv.imshow("Real Sense", self.output)

