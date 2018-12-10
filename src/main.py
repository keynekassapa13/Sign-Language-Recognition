import cv2 as cv
from settings import logger_settings
from recognition.hand_segment_bgsub import HandSegmentation, run_hand_segmentation

"""
TODO: A real GUI

For now capture_camera_loop() just runs the camera capture frame-by-frame.
"""

MAIN_LOG = logger_settings.setup_custom_logger("MAIN")


def capture_camera_loop():
    camera = cv.VideoCapture(0)
    MAIN_LOG.info(f"Camera {camera} capture started.")

    run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    # Quit
    camera.release()
    MAIN_LOG.info(f"Camera {camera} released.")


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
