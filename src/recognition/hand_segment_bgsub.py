import cv2 as cv


class HandSegmentation:
    """Segments the hand from the background using background subraction."""

    def __init__(self, camera):
        self.camera = camera
        self.bg_subtractors = {
            'KNN': cv.createBackgroundSubtractorKNN(),
            'MOG2': cv.createBackgroundSubtractorMOG2()
        }
        self.bg_model = None

    def apply_bg_subtract(self, frame, subtract: str = 'MOG2'):
        return self.bg_subtractors[subtract].apply(frame)

    def apply_gray_scale(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray, (7, 7), 0)
        return frame_gray

    def average_background(self, frame):
        if self.bg_model is None:
            # Init a background
            self.bg_model = frame.copy().astype("float")
        else:
            # Compute the weighted average, accumulate, and update background with result.
            cv.accumulateWeighted(frame, self.bg_model, 0.5)

    def segment_hand(self, frame, threshold=25):
        # Find absolute diff between background and current frame.
        diff = cv.absdiff(self.bg_model.astype("uint8"), frame)
        # Threshold diffed image for foreground
        threshold_frame = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)[1]
        return threshold_frame
