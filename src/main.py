import cv2 as cv
from settings import logger_settings

"""
TODO: A real GUI

For now capture_camera_loop() just runs the camera capture frame-by-frame.
"""

MAIN_LOG = logger_settings.setup_custom_logger("MAIN")


def capture_camera_loop():
    camera = cv.VideoCapture(0)
    MAIN_LOG.info(f"Camera capture started with {camera}")

    while True:
        # Capture frames
        return_val, frame = camera.read()

        # Colour spaces for camera output
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Output Camera
        cv.imshow('Capture', frame)
        # cv.imshow('Capture', frame_gray)

        # Break loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Quit
    camera.release()


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
