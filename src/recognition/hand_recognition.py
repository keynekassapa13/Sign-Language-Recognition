import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class HandRecognition:

    def __init__(
        self,
        left_frame,
        right_frame,
        original,
        left_rectangle,
        right_rectangle,
        rectangle_points
    ):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.original = original

        self.all_rectangles = None

        self.left_rectangle = left_rectangle
        self.right_rectangle = right_rectangle
        self.rectangle_points = rectangle_points
        self.__rectangle()

    def __rectangle(self):
        self.original = cv.rectangle(
            self.original,
            (self.left_rectangle[0][0],self.left_rectangle[0][1]),
            (self.left_rectangle[1][0],self.left_rectangle[1][1]),
            (0,255,0),
            3
        )

        self.original = cv.rectangle(
            self.original,
            (self.right_rectangle[0][0],self.right_rectangle[0][1]),
            (self.right_rectangle[1][0],self.right_rectangle[1][1]),
            (0,255,0),
            3
        )

        self.all_rectangles = self.original[
            self.rectangle_points[0][1]:self.rectangle_points[1][1],
            self.rectangle_points[0][0]:self.rectangle_points[1][0]
        ]
