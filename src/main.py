import cv2 as cv
import imutils
import math
import numpy as np
from typing import List, Tuple

from recognition import *
from settings import logger_settings
from settings.recognition_settings import *

"""
TODO: A real GUI

For now capture_camera_loop() just runs the camera capture frame-by-frame.

Keys:
-----
    q       - exit

"""

CAMERA_LOG = logger_settings.setup_custom_logger("CAMERA")


# TODO: James is going to clean this up!!
def capture_camera_loop():
    camera = cv.VideoCapture(0)
    CAMERA_LOG.info(f"Camera {camera} capture started.")

    # Camera Size Adjustment
    if camera.isOpened():
        width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)

        right_rectangle_points = [
            (math.floor(0.65 * width * 2), 0),
            (math.floor(1 * width * 2), 350)
        ]
        left_rectangle_points = [
            (math.floor(0 * width * 2), 0),
            (math.floor(0.35 * width * 2), 350)
        ]

        CAMERA_LOG.debug(f"Camera Width: {width} Height: {height}")
        CAMERA_LOG.debug(f"Left Rectangle Points {left_rectangle_points}")
        CAMERA_LOG.debug(f"Right Rectangle Points {right_rectangle_points}")

    # ------------------ Skin Extraction version ------------------------ #
    hand_recognition_skin_extract(camera, width,
                                  right_rectangle_points, left_rectangle_points)
    # ------------------ Background subtraction version ----------------- #
    # run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    # Quit
    camera.release()
    CAMERA_LOG.info(f"Camera {camera} released.")


def hand_recognition_skin_extract(camera,
                                  width: float,
                                  right_rectangle_points: List[Tuple[int, int]],
                                  left_rectangle_points: List[Tuple[int, int]]):
    """
    Runs skin extraction methodology.

    Shows a window for left hand, right hand, and actual video capture (2x width)

    End loop with 'q' keyboard kye.

    Args:
        camera: Camera currently capturing frames, should be started prior to
                running this.
        width: Width of the camera.
        right_rectangle_points: Position of left hand frame.
        left_rectangle_points: Position of left hand frame.
    """
    # Read first frame
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, int(width * 2))
    right_frame_class = VideoEnhancement(frame, HSV_LOWER2, HSV_UPPER2, right_rectangle_points)
    left_frame_class = VideoEnhancement(frame, HSV_LOWER2, HSV_UPPER2, left_rectangle_points)

    while True:
        # Capture current frame from active camera.
        return_val, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, int(width * 2))

        # ------------------ Right Hand ------------------

        right_frame_class.set_frame(frame)
        right_frame_class.skinExtraction()
        right_frame_class.contours2()

        # ------------------ Left Hand ------------------

        left_frame_class.set_frame(frame)
        left_frame_class.skinExtraction()
        left_frame_class.contours(1000)

        # ------------------ Hand Recognition ------------------

        hand_recognition_frame = HandRecognition(
            left_frame_class.frame,
            right_frame_class.frame,
            frame,
            left_rectangle_points,
            right_rectangle_points
        )

        # ------------------ Output Camera ------------------

        cv.imshow('Left Hand', hand_recognition_frame.left_frame)
        cv.imshow('Right Hand', hand_recognition_frame.right_frame)
        cv.imshow('Original', hand_recognition_frame.original)

        # ------------------ Break loop ------------------

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
