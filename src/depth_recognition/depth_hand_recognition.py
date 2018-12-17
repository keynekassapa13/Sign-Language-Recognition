import cv2 as cv
import pyrealsense2 as rs
import numpy as np


class DepthHandRecogniser:

    def __init__(self, pipeline):
        self.stream = pipeline
        self.depth_frame = None
        self.colour_frame = None

    def get_frames(self) -> bool:
        """Gets a depth and colour frame from realsense stream pipeline.

        Returns:
            True if successfully retrieved both depth and colour frames, else False.
        """
        frames = self.stream.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        self.colour_frame = frames.get_color_frame()
        if not self.depth_frame or self.colour_frame:
            return False
        return True

    def get_distance(self, width, height) -> float:
        x = x / 2
        y = y / 2
        return self.depth_frame.get_distance(x, y)
