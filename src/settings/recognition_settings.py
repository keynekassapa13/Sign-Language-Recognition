import numpy as np


"""
HSV Variables:
----------

lower - the lowest HSV skin color
upper - the highest HSV skin color

Please mind that lower and upper color threshold need to be adjusted with
surrounding situation
"""

HSV_LOWER1 = np.array([0, 133, 77], dtype="uint8")
HSV_UPPER1 = np.array([255, 173, 127], dtype="uint8")

HSV_LOWER2 = np.array([0, 140, 0], dtype="uint8")
HSV_UPPER2 = np.array([255, 173, 127], dtype="uint8")
