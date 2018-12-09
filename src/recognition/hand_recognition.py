import matplotlib.pyplot as plt
import numpy as np
import cv2

class HandRecognition:

    def __init__(self, frame, original):
        self.frame = frame
        self.original = original
        self.__rectangle()
        self.makePoints()

    def makePoints(self):
        # TO DO
        return None

    def __rectangle(self):
        self.original = cv2.rectangle(self.original, (700,0),(1200,350),(0,255,0),3)
