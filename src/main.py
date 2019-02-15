import tensorflow as tf
import sys
import cv2 as cv
# import pyrealsense2 as rs
import imutils
import math
import numpy as np
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

label_lines = [
    line.rstrip()
    for line in tf.gfile.GFile("mllib/tf_files/retrained_labels.txt")
]

with tf.gfile.GFile("mllib/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(
        graph_def,
        name=''
    )

###############################################################################
# Standard Camera Functionality
###############################################################################


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

            predictions = sess.run(
                softmax_tensor,
                {'Cast:0': hand_recognition_frame.all_rectangles}
            )
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            i = 0
            ml_predictions = []

            for node_id in top_k:
                if i == 3:
                    break
                i += 1
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                ml_predictions.append({human_string: score})

            hand_recognition_frame.set_ml_predictions(ml_predictions)
            final_prediction = hand_recognition_frame.classification()
            CAMERA_LOG.debug(f"Final Prediction: {final_prediction}\n")
            hand_recognition_frame.print_result()
            hand_recognition_frame.print_predictions()

            cv.imshow('Left Hand', hand_recognition_frame.left_frame.frame)
            cv.imshow('Right Hand', hand_recognition_frame.right_frame.frame)
            cv.imshow('Rectangles', hand_recognition_frame.show_original)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Quit
        camera.release()
        CAMERA_LOG.info(f"Camera {camera} released.")


###############################################################################
# Realsense Depth Camera Functionality
###############################################################################
def setup_rs_pipeline(frame_size: Tuple[int, int], framerate: int = 30):
    """Configures the realsense streaming pipeline.

    Args:
        frame_size: Tuple(width, height) of the size of stream frame.
        framerate: fps of stream. Default = 30fps.

    Returns:
        Configured rs pipeline that has begun streaming.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth,
        frame_size[0],
        frame_size[1],
        rs.format.z16,
        framerate
    )
    config.enable_stream(
        rs.stream.color,
        frame_size[0],
        frame_size[1],
        rs.format.bgr8,
        framerate
    )
    try:
        pipeline.start(config)
    except Exception as e:
        CAMERA_LOG.warn(f"RS Pipeline failed to start: {e}")
        raise
    return pipeline


def hand_recognition_depth(pipeline, frame_size: Tuple[int, int]):
    """Runs hand recognition using depth camera.

    NOTE - Not complete yet, so for now just runs
    the test stream to verify camera runs.

    Args:
        pipeline: realsense stream pipeline pre-configured.

    """
    width, height = frame_size
    recogniser = DepthHandRecogniser(pipeline, width, height)
    depth_gui = DepthRecogniserGUI(recogniser, frame_size)

    while True:
        if not recogniser.get_frames():
            CAMERA_LOG.warn("Dropped frames.")
            continue
        recogniser.segment_hand()
        depth_gui.draw_frames()
        # frames = stream.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        # colour_frame = frames.get_color_frame()
        # if not depth_frame or not colour_frame:
        #     continue
        #
        # # Distance
        # distance = depth_frame.get_distance(320, 240)
        #
        # # Convert images to data points in np array
        # depth_image = np.asanyarray(depth_frame.get_data())
        # colour_image = np.asanyarray(colour_frame.get_data())
        #
        # cv.putText(
        # colour_image,
        # str(distance) + " m",
        # (0, 30),
        # cv.FONT_HERSHEY_SIMPLEX,
        # 1.0,
        # (255, 255, 255), 2, cv.LINE_AA)
        #
        # Apply colour map to depth image
        # (converts image to 8-bit per pixel first)
        # depth_colourmap = cv.applyColorMap(
        # cv.convertScaleAbs(depth_image, alpha=0.03),
        # cv.COLORMAP_JET)
        #
        # # Vertical Stack Image
        # images = np.vstack((colour_image, depth_colourmap))
        #
        # # Show cv window
        # cv.namedWindow("Real Sense", cv.WINDOW_AUTOSIZE)
        # cv.imshow("Real Sense", images)

        # Quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    recogniser.stream.stop()


if __name__ == '__main__':
    ###########################################################################
    # Standard + Skin Extraction
    ###########################################################################
    camera, width, lp, rp, rps = setup_camera(RESIZE)
    hand_recognition(
        camera,
        RESIZE,
        width,
        rp,
        lp,
        rps
    )

    run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    ###########################################################################
    # RealSense
    ###########################################################################
    # FRAME_SIZE = (640, 480)
    # rs_pipeline = setup_rs_pipeline(FRAME_SIZE, 30)
    # hand_recognition_depth(rs_pipeline, FRAME_SIZE)

    cv.destroyAllWindows()
