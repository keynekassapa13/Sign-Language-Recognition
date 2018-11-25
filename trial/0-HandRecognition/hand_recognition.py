import cv2 as cv
import imutils
from typing import Tuple

"""
This experimental attempt is  based on the tutorial from: https://gogul09.github.io/software/hand-gesture-recognition-p1
Author of tutorial: Gogul

Note - excuse the mess of a class, this is just a rough implementation.
"""

# ROI of top, right, bottom, left
REGIONS_OF_INTEREST = (100, 200, 400, 500)

class HandRecognition:
    """Hand Recognition handler."""

    def __init__(self, weight: float, roi: Tuple[int, int, int, int]):
        self.camera = None
        self.roi_top = roi[0]
        self.roi_right = roi[1]
        self.roi_bottom = roi[2]
        self.roi_left = roi[3]
        self.bg_model = None
        self.image = None
        self.running_avg_weight = weight

    def bg_running_avg(self):
        if self.bg_model is None:
            # Init a background
            self.bg_model = self.image.copy().astype("float")
        else:
            # Compute the weighted average, accumulate, and update background with result.
            cv.accumulateWeighted(self.image, self.bg_model, self.running_avg_weight)

    def segment_hand_region(self, threshold=25):
        # Find absolute diff between background and current frame.
        diff = cv.absdiff(self.bg_model.astype("uint8"), self.image)

        # Threshold diffed image for foreground
        thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]

        # Get contours of hand in thresholded image
        (_, cnts, _) = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts) == 0:
            return
        else:
            # Get maximum contour representing the hand
            segmented = max(cnts, key=cv.contourArea)
            return thresholded, segmented

    def run_hand_recognition(self):
        self.camera = cv.VideoCapture(0)
        num_frames = 0

        while(True):
            # Current Frame
            grabbed, frame = self.camera.read()

            # Resize  and flip frame first
            frame = imutils.resize(frame, width=700)
            frame = cv.flip(frame, 1)

            # Clone frame, and get regions of interest and dimensions
            cloned_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]
            roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]

            # Grayscale and blur roi
            gray_frame = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            gray_frame = cv.GaussianBlur(gray_frame, (7, 7), 0)

            # Set image to our adjusted grayscaled frame
            self.image = gray_frame

            # Calculate running average over 30 frames, then segment.
            if num_frames < 30:
                self.bg_running_avg()
            else:
                # Segment hand
                hand = self.segment_hand_region()
                if hand:
                    thresholded, segment = hand
                    cv.drawContours(cloned_frame, [segment + (self.roi_right, self.roi_top)], -1, (0, 0, 255))
                    cv.imshow("Thresholded", thresholded)

            # Draw segmented hand
            cv.rectangle(cloned_frame, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0, 255, 0), 2)
            num_frames += 1

            # Display frame with segmented hand
            cv.imshow("Video Feed", cloned_frame)

            # if the user pressed "q", then stop looping
            keypress = cv.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break


if __name__ == '__main__':
    hand_recogniser = HandRecognition(0.5, REGIONS_OF_INTEREST)
    hand_recogniser.run_hand_recognition()
    hand_recogniser.camera.release()
    cv.destroyAllWindows()
