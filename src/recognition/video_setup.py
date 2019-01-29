import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import math
from settings import logger_settings

VIDEO_E = logger_settings.setup_custom_logger("VIDEO_E")


class VideoEnhancement:
    """
    Video Enhancement:
    -----------------

    Skin Color Extraction
    Make contours based on the extraction
    Do convexHull and convexity defects based on the contours
    """

    def __init__(self, frame, lower=0, upper=0, rectangle=[]):
        self.frame = None
        self.original = None
        self.bw = None

        self.lower = lower
        self.upper = upper
        self.mask = None
        self.rectangle = rectangle

        self.background = None
        self.max_contours = None

        self.final_count = 0

        self.set_frame(frame)


    def set_frame(self, frame):
        self.original = frame
        self.original = self.original[
            self.rectangle[0][1]:self.rectangle[1][1],
            self.rectangle[0][0]:self.rectangle[1][0]
        ]
        self.frame = self.original
        self.frame = cv.cvtColor(self.frame, cv.COLOR_BGR2YCR_CB)


    def skin_extraction(self):
        self.bw = cv.inRange(self.frame, self.lower, self.upper)
        self.frame = cv.bitwise_and(
            self.original,
            self.original,
            mask=self.bw
        )
        self.original = self.frame


    def contours(self, area_num):

        """
        Contours:
        ---------

        Choose the maximum area from the contour
        """
        # _, contours, _ = cv.findContours(
        #     self.frame,
        #     cv.RETR_EXTERNAL,
        #     cv.CHAIN_APPROX_SIMPLE
        # )

        # ret, self.frame = cv.threshold(
        #     self.frame,
        #     50,
        #     255,
        #     cv.THRESH_BINARY + cv.THRESH_OTSU
        # )
        # im2, contours, hierarchy = cv.findContours(
        #     self.frame,
        #     cv.RETR_TREE,
        #     cv.CHAIN_APPROX_SIMPLE
        # )

        _, contours, _ = cv.findContours(
            self.bw,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE
        )

        if contours:
            areas = [cv.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            temp_area = cv.contourArea(cnt)
            if (temp_area < area_num):
                self.final_count = 0
                return

            cv.drawContours(self.frame, [cnt], -1, (255, 255, 255), 2)

            self.final_count = self.__convexity(cnt)


    def __convexity(self, cnt):

        """
        Convexity:
        ---------

        cv.moments to choose the middle point
        cv.convexhull to convex polygon surrounded by all convex vertices
        cv.convexitydefects find convexity defects of a contour
        """

        # if contours:
        #     max_contours = max(contours, key=cv.contourArea)
        #     cv.drawContours(self.frame, max_contours, -1, (0, 255, 0), 3)
        #
        #     hull = []
        #     for i in range(len(contours)):
        #         hull.append(cv.convexHull(contours[i], False))
        #     for i in range(len(contours)):
        #         cv.drawContours(self.frame, hull, i, (255, 0, 0), 3, 8)
        #
        #     hull2 = cv.convexHull(max_contours, returnPoints=False)
        #     defects = cv.convexityDefects(max_contours, hull2)
        #
        #     if defects is not None:
        #         for i in range(defects.shape[0]):
        #             s, e, f, d = defects[i, 0]
        #             start = tuple(max_contours[s][0])
        #             end = tuple(max_contours[e][0])
        #             far = tuple(max_contours[f][0])
        #             cv.line(self.frame, start, end, (0, 255, 0), 2)
        #             cv.circle(self.frame, far, 5, (0, 0, 255), -1)

        # hull = []
        #
        # for i in range(len(contours)):
        #     hull.append(cv.convexHull(contours[i], False))
        #
        # drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3),snp.uint8)
        # for i, c in enumerate(contours):
        #     area = cv.contourArea(c)
        #     if area > area_num:
        #         cv.drawContours(self.frame, contours, i, (0, 255, 0), 3)
        #         cv.drawContours(self.frame, hull, i, (255, 0, 0),  1, 8)

        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(self.frame, (cx, cy), 35, (255, 0, 0), -1)
        else:
            cx, cy = 0, 0

        hull = cv.convexHull(cnt)
        cv.drawContours(self.frame, [hull], -1, (255, 0, 255),  2, 8)
        hull = cv.convexHull(cnt, returnPoints=False)

        try:
            defects = cv.convexityDefects(cnt, hull)
        except Exception as e:
            defects = None
            print(e)

        counter = 1
        if defects is not None:
            for i in range(defects.shape[0]):

                """
                Defects:
                --------

                s = CvPoint* start         --> point of the contour
                                            where the defect begins
                e = CvPoint* end           --> point of the contour
                                            where the defect ends
                d = CvPoint* depth_point   --> the farthest from the convex
                                            hull point within the defect
                f = float depth            --> approximate distance
                                            to farthest point
                """
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])

                if (d < 10000):
                    continue

                if (far[1] >= (cy + 40)):
                    continue

                # From https://github.com/patilnabhi/
                # tbotnav/blob/master/scripts/fingers_recog.py

                if self.__angle_rad(
                    np.subtract(start, far),
                    np.subtract(end, far)
                ) < self.__deg2rad(80):
                    counter += 1
                    cv.circle(self.frame, far, 5, [0, 255, 0], -1)
                else:
                    cv.circle(self.frame, far, 5, [0, 0, 255], -1)

        return counter


    def __angle_rad(self, v1, v2):
        return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))


    def __deg2rad(self, angle_deg):
        return angle_deg/180.0*np.pi


    def __image_filtering(self, iters=2):
        """Experiment attempt at filtering pass on the skin extraction step."""
        # Threshold Step, examines intensity of object and background,
        # tries to focus only on the foreground.
        # However this may not be as useful as hoped  as
        # it seems to work best with grayscale images.
        ret, self.frame = cv.threshold(
            self.frame,
            50,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))

        # The following two filtering techniques are incompatible
        # with gradient morph.
        # self.frame = cv.erode(self.frame, kernel, iterations=iters)
        # self.frame = cv.dilate(self.frame, kernel, iterations=iters)

        # Gaussian Blur will try to smooth the 'holes' in the image
        # which is helpful for the morphology step.
        self.frame = cv.GaussianBlur(self.frame, (3, 3), 0)
        self.frame = cv.morphologyEx(self.frame, cv.MORPH_GRADIENT, kernel)
