from settings import logger_settings
import cv2 as cv
import numpy as np


DEPTH_LOG = logger_settings.setup_custom_logger("Depth Recogniser")


class DepthHandRecogniser:

    def __init__(self, pipeline):
        self.stream = pipeline
        self.depth_frame = None
        self.depth_img = None
        self.colour_frame = None
        self.colour_img = None

    def get_frames(self) -> bool:
        """Gets a depth and colour frame from realsense stream pipeline.

        Returns:
            True if successfully retrieved both depth and colour frames, else False.
        """
        frames = self.stream.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        self.colour_frame = frames.get_color_frame()
        if not self.depth_frame or not self.colour_frame:
            DEPTH_LOG.debug(f"Dropped frames. Depth: {self.depth_frame}, Colour: {self.colour_frame}")
            return False
        # Frames available to process
        self.__process_rs_frames()
        return True

    def __process_rs_frames(self):
        # Process Depth Image
        self.depth_img = np.asanyarray(self.depth_frame.get_data())
        self.depth_img = cv.flip(self.depth_img, 1)
        # Process Colour Image
        self.colour_img = np.asanyarray(self.colour_frame.get_data())
        self.colour_img = cv.flip(self.colour_img, 1)

