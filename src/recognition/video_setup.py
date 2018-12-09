import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

class VideoEnhancement:

    def __init__(self, frame, lower = 0, upper = 0, rectangle = []):
        self.frame = frame
        self.original = frame
        self.lower = lower
        self.upper = upper
        self.mask = None
        self.rectangle = rectangle
        self.__rectangle()

    def __rectangle(self):
        self.frame = self.frame[
            self.rectangle[0][1]:self.rectangle[1][1],
            self.rectangle[0][0]:self.rectangle[1][0]
        ]

    def turnToYCrCb(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)

    def skinExtraction(self):
        self.frame = cv2.inRange(self.frame, self.lower, self.upper)

    def contours(self, areaNum):
        ret, thresh = cv2.threshold(self.frame, 50, 255, cv2.THRESH_BINARY)
        # _, contours, _ = cv2.findContours(self.frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im2, contours, hierarchy = cv2.findContours(self.frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []

        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))

        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > areaNum:
                cv2.drawContours(self.frame, contours, i, (0, 255, 0), 3)
                cv2.drawContours(self.frame, hull, i, (255, 0, 0),  1, 8)

    def frameFiltering(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self.frame = cv2.erode(self.frame, kernel, iterations = 2)
        self.frame = cv2.dilate(self.frame, kernel, iterations = 2)
        self.frame = cv2.GaussianBlur(self.frame, (3, 3), 0)
        self.frame = cv2.bitwise_and(self.original, self.original, mask = self.frame)
