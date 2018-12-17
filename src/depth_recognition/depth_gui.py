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

    def get_frames_to_draw(self):
        depth_image = np.asanyarray(self.recogniser.depth_frame.get_data())
        depth_colourmap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        colour_image = np.asanyarray(self.recogniser.colour_frame.get_data())
        distance = self.recogniser.get_distance(self.width, self.height)

        cv.putText(colour_image, str(distance) + " m", (0, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        self.output = np.vstack((colour_image, depth_colourmap))

    def draw_frames(self):
        cv.namedWindow("Real Sense", cv.WINDOW_AUTOSIZE)
        cv.imshow("Real Sense", self.output)

