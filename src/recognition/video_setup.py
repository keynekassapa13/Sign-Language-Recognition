import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
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

    def __init__(self, frame, lower=0, upper=0, rectangle=[]):
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
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2YCR_CB)

    def backgroundSubstraction(self):
        return 0

    def skinExtraction(self):
        self.frame = cv.inRange(self.frame, self.lower, self.upper)

    def contours(self, areaNum):

        """
        Contours:
        ---------

        Choose the maximum area from the contour
        """

        ret, thresh = cv.threshold(self.frame, 50, 255, cv.THRESH_BINARY)
        # _, contours, _ = cv.findContours(
        #     self.frame,
        #     cv.RETR_EXTERNAL,
        #     cv.CHAIN_APPROX_SIMPLE
        # )
        im2, contours, hierarchy = cv.findContours(
            self.frame,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE
        )

        if contours:
            areas = [cv.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            cv.drawContours(self.frame, [cnt], -1, (0, 255, 0), 3)

            fingers = self.__convexity(cnt)
            self.__printText(str(fingers))

    def __convexity(self, cnt):

        """
        Convexity:
        ---------

        cv.moments to choose the middle point
        cv.convexhull to convex polygon surrounded by all convex vertices
        cv.convexitydefects find convexity defects of a contour
        """

        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(self.frame, (cx, cy), 30, (0, 0, 255), -1)
        else:
            cx, cy = 0, 0

        hull = cv.convexHull(cnt)
        cv.drawContours(self.frame, [hull], -1, (255, 0, 0),  1, 8)
        hull = cv.convexHull(cnt, returnPoints=False)

        try:
            defects = cv.convexityDefects(cnt, hull)
        except Exception as e:
            defects = None
            print(e)

        counter = 0
        if defects is not None:
            for i in range(defects.shape[0]):

                """
                Defects:
                --------

                s = CvPoint* start         --> point of the contour where the defect begins
                e = CvPoint* end           --> point of the contour where the defect ends
                d = CvPoint* depth_point   --> the farthest from the convex hull point within the defect
                f = float depth            --> approximate distance to farthest point
                """
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                if (d < 10000):
                    continue

                if (far[1] >= (cy + 40)):
                    continue

                cv.line(self.frame, end, far, (0, 100, 0), 2, 8)
                counter += 1

        return counter

        # hull = []
        #
        # for i in range(len(contours)):
        #     hull.append(cv.convexHull(contours[i], False))
        #
        # drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3),snp.uint8)
        # for i, c in enumerate(contours):
        #     area = cv.contourArea(c)
        #     if area > areaNum:
        #         cv.drawContours(self.frame, contours, i, (0, 255, 0), 3)
        #         cv.drawContours(self.frame, hull, i, (255, 0, 0),  1, 8)

    def __printText(self, text):
        try:
            cv.putText(
                self.frame,
                str(text),
                (0, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )
            return 1
        except Exception as e:
            print(e)

    def frameFiltering(self):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        self.frame = cv.erode(self.frame, kernel, iterations=2)
        self.frame = cv.dilate(self.frame, kernel, iterations=2)
        self.frame = cv.GaussianBlur(self.frame, (3, 3), 0)
        self.frame = cv.bitwise_and(
            self.original,
            self.original,
            mask=self.frame
        )
