import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

class VideoEnhancement:

    def __init__(self, frame, lower = 0, upper = 0):
        self.frame = frame
        self.original = frame
        self.lower = lower
        self.upper = upper
        self.mask = None
        self.__flip()
        self.__rectangle()

    def __flip(self):
        self.frame = cv2.flip(self.frame, 1)
        self.original = cv2.flip(self.original, 1)

    def __rectangle(self):
        self.frame = self.frame[0:350, 700:1200]

    def turnToYCrCb(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCR_CB)

    def skinExtraction(self):
        self.frame = cv2.inRange(self.frame, self.lower, self.upper)
