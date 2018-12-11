import cv2 as cv
import imutils
import math
import numpy as np
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

# right_rectangle_points = [(700, 0), (1200, 350)]
# left_rectangle_points = [(0, 0), (400, 350)]

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

    # Read first frame
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, int(width * 2))
    right_frame_class = VideoEnhancement(frame, HSV_LOWER2, HSV_UPPER2, right_rectangle_points)
    left_frame_class = VideoEnhancement(frame, HSV_LOWER2, HSV_UPPER2, left_rectangle_points)

    while True:
        # Capture frames
        return_val, frame = camera.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, int(width * 2))

        # ------------------ Right Hand ------------------
        # right_frame_class = VideoEnhancement(
        #     frame,
        #     lower,
        #     upper,
        #     right_rectangle_points
        # )

        right_frame_class.set_frame(frame)
        right_frame_class.skinExtraction()
        right_frame_class.contours2()

        # ------------------ Left Hand ------------------

        # left_frame_class = VideoEnhancement(
        #     frame,
        #     lower,
        #     upper,
        #     left_rectangle_points
        # )

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

    # ------------------ Background subtraction version -----------------
    # run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    # Quit
    camera.release()
    CAMERA_LOG.info(f"Camera {camera} released.")


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
