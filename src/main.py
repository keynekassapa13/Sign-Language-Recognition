import cv2 as cv
from settings import logger_settings
from recognition.hand_segment_bgsub import HandSegmentation

"""
TODO: A real GUI

For now capture_camera_loop() just runs the camera capture frame-by-frame.
"""

MAIN_LOG = logger_settings.setup_custom_logger("MAIN")


def capture_camera_loop():
    camera = cv.VideoCapture(0)
    MAIN_LOG.info(f"Camera {camera} capture started.")
    hand_segment = HandSegmentation(camera)
    num_frames = 0

    MAIN_LOG.info("Started background averaging.")
    while True:
        # Capture frames
        return_val, frame = hand_segment.camera.read(0)
        frame = cv.flip(frame, 1)
        cloned_frame = frame.copy()

        ########################################################################
        # Average background Subtraction, comment out if necessary for other attempts.
        ########################################################################
        # Output Segmentation Camera
        gray_frame = hand_segment.apply_gray_scale(cloned_frame)
        KNN_frame = hand_segment.apply_bg_subtract(cloned_frame, 'KNN')
        MOG2_frame = hand_segment.apply_bg_subtract(cloned_frame, 'MOG2')

        # Background subtract before hand segmentation.
        if num_frames < 60:
            # Only run one at a time!
            hand_segment.average_background(gray_frame)
            # hand_segment.average_background(KNN_frame)
            # hand_segment.average_background(MOG2_frame)
        elif num_frames == 60:
            MAIN_LOG.info("Finished background averaging.")
        else:
            # Segment with appropriate frame
            hand = hand_segment.segment_hand(gray_frame)
            # hand = hand_segment.segment_hand(KNN_frame)
            # hand = hand_segment.segment_hand(MOG2_frame)
            cv.imshow('Hand No Subtraction', hand)
        num_frames += 1

        ########################################################################
        # Average background Subtraction ends
        ########################################################################

        # Output Standard Camera
        cv.imshow('Capture', frame)

        # Break loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Quit
    camera.release()
    MAIN_LOG.info(f"Camera {camera} released.")


if __name__ == '__main__':
    capture_camera_loop()
    cv.destroyAllWindows()
