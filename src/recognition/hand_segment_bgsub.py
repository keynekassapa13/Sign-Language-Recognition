import cv2 as cv
import imutils
from typing import Tuple, List
from settings import logger_settings


HAND_SEG_LOG = logger_settings.setup_custom_logger("HAND_SEG")

"""
This technique is potentially not as efficient as skin extraction but I have 
left this here for future reference, in case we figure out a better method of 
using background and motion.
"""


class HandRecogniserOutput:

    def __init__(self, left_frame, right_frame, main_frame, left_rect_dims: List[Tuple[int, int]], right_rect_dims: List[Tuple[int, int]]):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.main_frame = main_frame
        self.left_rectangle = left_rect_dims
        self.right_rectangle = right_rect_dims
        self.hand_segment = HandSegmentation(self)

    def run_hand_segmentation(self):
        return None

    def run_counter(self):
        return None

class HandSegmentation:
    """Segments the hand from the background using background subraction."""

    def __init__(self, output):
        self.output = output
        self.bg_subtractors = {
            'KNN': cv.createBackgroundSubtractorKNN(),
            'MOG2': cv.createBackgroundSubtractorMOG2()
        }
        self.bg_avg = None

    def apply_bg_subtract(self, frame, subtract: str = 'MOG2'):
        return self.bg_subtractors[subtract].apply(frame)

    def apply_gray_scale(self, frame, ksize: Tuple[int, int]):
        """
        Converts frame to grayscale, and applies a Gaussian Blur to smooth out
        noise.
        Args:
            frame: image frame captured from camera.
            ksize: Tuple(int, int) containing Gaussian Blur kernal. Always
                    positive and odd.

        Returns:
            The converted image frame.
        """
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray, ksize, 0)
        return frame_gray

    def average_background(self, frame, weight: float = 0.5) -> None:
        """
        Function creates an average of the background based on current frame and
        previous frames.
        Args:
            frame: image frame captured from camera.
            weight: float representing the weight applied to average function.
                    Lower weight is less sensitive to movement. (0 < a < 1.0)
        """
        if self.bg_avg is None:
            # Init a background
            self.bg_avg = frame.copy().astype("float")
        else:
            # Compute the weighted average, accumulate, and update background with result.
            cv.accumulateWeighted(frame, self.bg_avg, weight)

    def segment_hand(self, frame, threshold: int = 25):
        # Find absolute diff between background and current frame.
        diff = cv.absdiff(self.bg_avg.astype("uint8"), frame)
        # Threshold diffed image for foreground
        threshold_frame = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        threshold_frame = cv.erode(threshold_frame, None, iterations=2)
        threshold_frame = cv.dilate(threshold_frame, None, iterations=2)
        return threshold_frame

    def contour_hand(self, threshold_frame):
        _, contours, _ = cv.findContours(threshold_frame.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        return max(contours, key=cv.contourArea)


def run_hand_segmentation(camera, roi: Tuple[int, int, int, int], alpha: float = 0.5):
    hand_segmentor = HandSegmentation()
    # Frames to run background averaging before segmentation
    bg_avg_frames = 30
    num_frames = 0
    # Region of Interest (avoids contouring entire frame)
    top, right, bottom, left = roi
    # Alpha weighting for averaging function
    avg_weight = alpha

    HAND_SEG_LOG.info(f"Running hand segmentation for {bg_avg_frames} frames.")
    HAND_SEG_LOG.info(f"Region of Interest {roi}")
    HAND_SEG_LOG.info(f"Alpha for background averaging: {avg_weight}")
    while True:
        ret, frame = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv.flip(frame, 1)
        frame_clone = frame.copy()
        frame_roi = frame[top:bottom, right:left]
        grayscale_frame = hand_segmentor.apply_gray_scale(frame_roi, (5, 5))

        if num_frames < bg_avg_frames:
            hand_segmentor.average_background(grayscale_frame, avg_weight)
        else:
            thresholded_hand = hand_segmentor.segment_hand(grayscale_frame, 10)
            contour_hand = hand_segmentor.contour_hand(thresholded_hand)
            if thresholded_hand is not None:
                cv.drawContours(frame_clone, [contour_hand + (right, top)], -1, (0, 0, 255))
                cv.imshow("Thresholded Hand", thresholded_hand)

        num_frames += 1
        cv.rectangle(frame_clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.imshow("Video Feed", frame_clone)

        # Wait for key before quitting
        keypress = cv.waitKey(1) & 0xFF
        if keypress == ord('q'):
            HAND_SEG_LOG.debug("User hit 'q' to quit.")
            break

