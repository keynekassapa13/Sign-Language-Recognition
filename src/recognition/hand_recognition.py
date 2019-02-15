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

    def print_result(self):
        self.__printText(
            self.left_frame.frame,
            str(self.left_frame.final_count)
        )
        self.__printText(
            self.right_frame.frame,
            str(self.right_frame.final_count)
        )

    def set_ml_predictions(self, ml_predictions):
        self.ml_predictions = ml_predictions

    def classification(self):
        if self.ml_predictions == {}:
            return False

        left_count = self.left_frame.final_count
        right_count = self.right_frame.final_count

        top_1 = list(self.ml_predictions[0].keys())[0]
        top_2 = list(self.ml_predictions[1].keys())[0]
        top_3 = list(self.ml_predictions[2].keys())[0]

        final_prediction = ""

        if (left_count == 3 or left_count == 2 or left_count == 4) and\
                (right_count == None or right_count == 1 or right_count == 2) and\
                (top_1 == "alphabet_a" or top_2 == "alphabet_a" or top_3 == "alphabet_a"):
            final_prediction = "A"
        elif (left_count == 3 or left_count == 2 or left_count == 4) and\
                (right_count == 3 or right_count == 2 or right_count == 4) and\
                (top_1 == "alphabet_b" or top_2 == "alphabet_b" or top_3 == "alphabet_b"):
            final_prediction = "B"
        elif (left_count == None) and\
                (right_count == 1 or right_count == 2) and\
                (top_1 == "alphabet_c" or top_2 == "alphabet_c" or top_3 == "alphabet_c"):
            final_prediction = "C"

        if final_prediction != "":
            self.final_prediction = final_prediction
            return final_prediction

        if (left_count == None) and\
                (right_count == 1) and\
                (top_1 == "number_1" or top_2 == "number_1" or top_3 == "number_1"):
            final_prediction = "1"
        elif (left_count == None) and\
                (right_count == 2) and\
                (top_1 == "number_2" or top_2 == "number_2" or top_3 == "number_2"):
            final_prediction = "2"
        elif (left_count == None) and\
                (right_count == 2) and\
                (top_1 == "number_2" or top_2 == "number_2" or top_3 == "number_2"):
            final_prediction = "2"
        elif (left_count == None) and\
                (right_count == 3) and\
                (top_1 == "number_3" or top_2 == "number_3" or top_3 == "number_3"):
            final_prediction = "3"
        elif (left_count == None) and\
                (right_count == 4) and\
                (top_1 == "number_4" or top_2 == "number_4" or top_3 == "number_4"):
            final_prediction = "4"
        elif (left_count == None) and\
                (right_count == 5) and\
                (top_1 == "number_5" or top_2 == "number_5" or top_3 == "number_5"):
            final_prediction = "5"
        elif (left_count == None) and\
                (right_count == 1 or right_count == None) and\
                (top_1 == "number_6" or top_2 == "number_6" or top_3 == "number_6"):
            final_prediction = "6"
        elif (left_count == None) and\
                (right_count == 1 or right_count == 2) and\
                (top_1 == "number7" or top_2 == "number_7" or top_3 == "number_7"):
            final_prediction = "7"

        if final_prediction == "":
            final_prediction = "None detected"

        self.final_prediction = final_prediction
        return final_prediction

    def print_predictions(self):
        self.__printText(
            self.show_original,
            str(self.final_prediction)
        )

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
