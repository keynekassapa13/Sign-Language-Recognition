import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


class HandRecognition:

    def __init__(
        self,
        left_frame,
        right_frame,
        original,
        rectangle_points
    ):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.original = original
        self.show_original = original

        self.all_rectangles = None
        self.ml_predictions = {}

        self.left_rectangle = left_frame.rectangle
        self.right_rectangle = right_frame.rectangle
        self.rectangle_points = rectangle_points
        self.__rectangle()
        self.printResult()

    def printResult(self):
        self.__printText(
            self.left_frame.frame,
            str(self.left_frame.final_count)
        )
        self.__printText(
            self.right_frame.frame,
            str(self.right_frame.final_count)
        )

    def classification(self):
        return

    def __rectangle(self):
        self.all_rectangles = self.original[
            self.rectangle_points[0][1]: self.rectangle_points[1][1],
            self.rectangle_points[0][0]: self.rectangle_points[1][0]
        ]
        self.show_original = cv.rectangle(
            self.show_original,
            (self.left_rectangle[0][0], self.left_rectangle[0][1]),
            (self.left_rectangle[1][0], self.left_rectangle[1][1]),
            (0, 255, 0),
            3
        )

        self.show_original = cv.rectangle(
            self.show_original,
            (self.right_rectangle[0][0], self.right_rectangle[0][1]),
            (self.right_rectangle[1][0], self.right_rectangle[1][1]),
            (0, 255, 0),
            3
        )

    def __printText(self, frame, text):
        try:
            cv.putText(
                frame,
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
