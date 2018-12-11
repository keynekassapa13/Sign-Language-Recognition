import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from settings import logger_settings

CAMERA_LOG = logger_settings.setup_custom_logger("MAIN")

class VideoEnhancement:
    """
    Video Enhancement:
    -----------------

    Skin Color Extraction
    Make contours based on the extraction
    Do convexHull and convexity defects based on the contours
    """

    def __init__(self, frame, lower = 0, upper = 0, rectangle = []):
        self.frame = frame
        self.original = frame
        self.lower = lower
        self.upper = upper
        self.mask = None
        self.rectangle = rectangle
        self.__rectangle()
        self.__turnToYCrCb()

    def __rectangle(self):
        self.frame = self.frame[
            self.rectangle[0][1]:self.rectangle[1][1],
            self.rectangle[0][0]:self.rectangle[1][0]
        ]

    def __turnToYCrCb(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)

    def backgroundSubstraction(self):
        return 0

    def skinExtraction(self):
        self.frame = cv2.inRange(self.frame, self.lower, self.upper)

    def contours(self, areaNum):
        ret, thresh = cv2.threshold(self.frame, 50, 255, cv2.THRESH_BINARY)
        # _, contours, _ = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im2, contours, hierarchy = cv2.findContours(self.frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            cv2.drawContours(self.frame, [cnt], -1, (0, 255, 0), 3)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            cv2.circle(self.frame, (cx, cy), 30, (0,0,255), -1)

            hull1 = cv2.convexHull(cnt)

            cv2.drawContours(self.frame, [hull1], -1, (255, 0, 0),  1, 8)

            hull2 = cv2.convexHull(cnt, returnPoints=False)

            try:
                defects = cv2.convexityDefects(cnt, hull2)
            except Exception as e:
                defects = None
                print(e)

            counter = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    if (d < 20000):
                        continue

                    if (far[1] >= (cy+40)):
                        continue

                    cv2.line(self.frame, end, far, (0, 100, 0), 2, 8)
                    counter += 1


        # hull = []
        #
        # for i in range(len(contours)):
        #     hull.append(cv2.convexHull(contours[i], False))
        #
        # drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        # for i, c in enumerate(contours):
        #     area = cv2.contourArea(c)
        #     if area > areaNum:
        #         cv2.drawContours(self.frame, contours, i, (0, 255, 0), 3)
        #         cv2.drawContours(self.frame, hull, i, (255, 0, 0),  1, 8)


    def frameFiltering(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.frame = cv2.erode(self.frame, kernel, iterations = 2)
        self.frame = cv2.dilate(self.frame, kernel, iterations = 2)
        self.frame = cv2.GaussianBlur(self.frame, (3, 3), 0)
        self.frame = cv2.bitwise_and(self.original, self.original, mask = self.frame)
