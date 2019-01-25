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

        self.final_count = None

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
            mask = self.bw
        )
        self.original = self.frame

    def contours(self, area_num):

        """
        Contours:
        ---------

        Choose the maximum area from the contour
        """

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
                return

            cv.drawContours(self.frame, [cnt], -1, (255, 255, 255), 2)

            self.final_count = self.__convexity(cnt)
            self.__printText(str(self.final_count))

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

                s = CvPoint* start         --> point of the contour where the defect begins
                e = CvPoint* end           --> point of the contour where the defect ends
                d = CvPoint* depth_point   --> the farthest from the convex hull point within the defect
                f = float depth            --> approximate distance to farthest point
                """
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])    # tip
                end = tuple(cnt[e][0])      # tip
                far = tuple(cnt[f][0])      # angle

                if (d < 10000):
                    continue

                if (far[1] >= (cy + 40)):
                    continue

                # From https://github.com/patilnabhi/tbotnav/blob/master/scripts/fingers_recog.py

                if self.__angle_rad(np.subtract(start, far), np.subtract(end, far)) < self.__deg2rad(80):
                    counter += 1
                    cv.circle(self.frame, far, 5, [0, 255, 0], -1)
                else:
                    cv.circle(self.frame, far, 5, [0, 0, 255], -1)

        return counter

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

    def __angle_rad(self, v1, v2):
	       return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    def __deg2rad(self, angle_deg):
    	return angle_deg/180.0*np.pi
