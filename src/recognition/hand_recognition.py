import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class HandRecognition:

    def __init__(
        self,
        left_frame,
        right_frame,
        original
    ):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.original = original
