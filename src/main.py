import tensorflow as tf
import sys
import cv2 as cv
import imutils
import math
import numpy as np
import os
from typing import List, Tuple

from depth_recognition import *
from recognition import *
from settings import logger_settings
from settings.recognition_settings import *

"""
TODO: A real GUI

For now setup_camera() just runs the camera capture frame-by-frame.

Keys:
-----
    q       - exit

"""

CAMERA_LOG = logger_settings.setup_custom_logger("CAMERA")
CAMERA_LOG.debug(f"Tensorflow {tf.__version__}")

label_lines = [line.rstrip() for line in tf.gfile.GFile("mllib/tf_files/retrained_labels.txt")]

with tf.gfile.GFile("mllib/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(
        graph_def,
        name=''
    )

def setup_camera(
    resize=1
):
    """
    Function setup_camera:
    -----------------------------
    TODO
    """
    camera = cv.VideoCapture(0)
    CAMERA_LOG.info(f"Camera {camera} capture started.")

    # Camera Size Adjustment
    if camera.isOpened():
        width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)

        right_rectangle_points = [
            (math.floor(0.5 * width * resize), 100),
            (math.floor(0.9 * width * resize), 350)
        ]
        left_rectangle_points = [
            (math.floor(0.1 * width * resize), 100),
            (math.floor(0.5 * width * resize), 350)
        ]

        rectangle_points = [
            (math.floor(0.1 * width * resize), 100),
            (math.floor(0.9 * width * resize), 350)
        ]

        CAMERA_LOG.debug(f"Camera Width: {width} Height: {height}")
        CAMERA_LOG.debug(f"Left Rectangle Points {left_rectangle_points}")
        CAMERA_LOG.debug(f"Right Rectangle Points {right_rectangle_points}")

        return (
            camera,
            width,
            left_rectangle_points,
            right_rectangle_points,
            rectangle_points
        )
    return False


def hand_recognition(
    camera,
    resize,
    width: float,
    right_rectangle_points: List[Tuple[int, int]],
    left_rectangle_points: List[Tuple[int, int]],
    rectangle_points: List[Tuple[int, int]]
):
    """
    Function hand_recognition:
    -----------------------------
    Runs skin extraction methodology.
    Shows a window for left hand, right hand, and actual video capture
    End loop with 'q' keyboard kye.

    Args:
        camera: Camera currently capturing frames, should be started prior to
                running this.
        width: Width of the camera.
        right_rectangle_points: Position of left hand frame.
        left_rectangle_points: Position of left hand frame.
    """
    # Capture first background
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        _, frame = camera.read()
        frame = cv.flip(frame, 1)
        if resize:
            frame = imutils.resize(frame, int(width * resize))

        right_frame_class = VideoEnhancement(
            frame,
            HSV_LOWER2,
            HSV_UPPER2,
            right_rectangle_points
        )

        left_frame_class = VideoEnhancement(
            frame,
            HSV_LOWER2,
            HSV_UPPER2,
            left_rectangle_points
        )

        while True:
            # Capture current frame from active camera.
            _, frame = camera.read()
            frame = cv.flip(frame, 1)
            if resize:
                frame = imutils.resize(frame, int(width * resize))

            right_frame_class.set_frame(frame)
            right_frame_class.skin_extraction()
            right_frame_class.contours(CONTOURS_AREA_THRESH)

            left_frame_class.set_frame(frame)
            left_frame_class.skin_extraction()
            left_frame_class.contours(CONTOURS_AREA_THRESH)

            hand_recognition_frame = HandRecognition(
                left_frame_class,
                right_frame_class,
                frame,
                rectangle_points
            )

            # predictions = sess.run(softmax_tensor, {'Cast:0': hand_recognition_frame.all_rectangles})
            #
            # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            #
            # for node_id in top_k:
            #     human_string = label_lines[node_id]
            #     score = predictions[0][node_id]
            #     print('%s (score = %.5f)' % (human_string, score))
            #
            # print('------------------ \n\n')

            cv.imshow('Left Hand', hand_recognition_frame.left_frame.frame)
            cv.imshow('Right Hand', hand_recognition_frame.right_frame.frame)
            cv.imshow('Original', hand_recognition_frame.original)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Quit
        camera.release()
        CAMERA_LOG.info(f"Camera {camera} released.")


if __name__ == '__main__':
    camera, width, left_rectangle_points, right_rectangle_points, rectangle_points = setup_camera(RESIZE)
    hand_recognition(
        camera,
        RESIZE,
        width,
        right_rectangle_points,
        left_rectangle_points,
        rectangle_points
    )
    run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    cv.destroyAllWindows()
