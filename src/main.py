import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
TODO: A real GUI

For now capture_camera_loop() just runs the camera capture frame-by-frame.
"""

# ------------------ Files ------------------
from recognition import *
from settings import logger_settings

CAMERA_LOG = logger_settings.setup_custom_logger("MAIN")

# ------------------ Variables ------------------
# lower = np.array([0,133,77], dtype="uint8")
# upper = np.array([255,173,127], dtype="uint8")

lower = np.array([0,140,0], dtype="uint8")
upper = np.array([255,173,127], dtype="uint8")

right_rectangle_points = [(700,0), (1200,350)]
left_rectangle_points = [(0,0), (400,350)]


def capture_camera_loop():
    camera = cv.VideoCapture(0)
    CAMERA_LOG.info(f"Camera capture started with {camera}")

    while True:
        # Capture frames
        return_val, frame = camera.read()

        frame = cv2.flip(frame, 1)

        # Colour spaces for camera output
        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ------------------ Right Hand ------------------
        right_frame_class = VideoEnhancement(
            frame,
            lower,
            upper,
            right_rectangle_points
        )
        right_frame_class.skinExtraction()
        right_frame_class.contours(1000)

        # ------------------ Left Hand ------------------

        left_frame_class = VideoEnhancement(
            frame,
            lower,
            upper,
            left_rectangle_points
        )
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

        # Break loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Quit
    logger_settings.log_time()
    camera.release()


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
