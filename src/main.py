import cv2 as cv
import pyrealsense2 as rs
import imutils
import math
import numpy as np
from typing import List, Tuple

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


################################################################################
# Standard Camera Functionality
################################################################################
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
            (math.floor(0.55 * width * resize), 0),
            (math.floor(1 * width * resize), 350)
        ]
        left_rectangle_points = [
            (math.floor(0 * width * resize), 0),
            (math.floor(0.45 * width * resize), 350)
        ]

        CAMERA_LOG.debug(f"Camera Width: {width} Height: {height}")
        CAMERA_LOG.debug(f"Left Rectangle Points {left_rectangle_points}")
        CAMERA_LOG.debug(f"Right Rectangle Points {right_rectangle_points}")

        return (camera, width, left_rectangle_points, right_rectangle_points)
    return False


def hand_recognition(
    camera,
    resize,
    width: float,
    right_rectangle_points: List[Tuple[int, int]],
    left_rectangle_points: List[Tuple[int, int]]
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
    ret, frame = camera.read()
    frame = cv.flip(frame, 1)
    if resize > 1:
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
        return_val, frame = camera.read()
        frame = cv.flip(frame, 1)
        if resize > 1:
            frame = imutils.resize(frame, int(width * resize))

        right_frame_class.set_frame(frame)
        right_frame_class.skin_extraction()
        right_frame_class.contours(1000)

        left_frame_class.set_frame(frame)
        left_frame_class.skin_extraction()
        left_frame_class.contours(1000)

        hand_recognition_frame = HandRecognition(
            left_frame_class.frame,
            right_frame_class.frame,
            frame,
            left_rectangle_points,
            right_rectangle_points
        )

        cv.imshow('Left Hand', hand_recognition_frame.left_frame)
        cv.imshow('Right Hand', hand_recognition_frame.right_frame)
        cv.imshow('Original', hand_recognition_frame.original)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Quit
    camera.release()
    CAMERA_LOG.info(f"Camera {camera} released.")


################################################################################
# Realsense Depth Camera Functionality
################################################################################
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
    config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, framerate)
    config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, framerate)
    try:
        pipeline.start(config)
    except Exception as e:
        CAMERA_LOG.warn(f"RS Pipeline failed to start: {e}")
        raise
    return pipeline


def hand_recognition_depth(pipeline):
    """Runs hand recognition using depth camera.

    NOTE - Not complete yet, so for now just runs the test stream to verify camera runs.

    Args:
        pipeline: realsense stream pipeline pre-configured.

    """
    stream = pipeline

    while True:
        frames = stream.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        colour_frame = frames.get_color_frame()
        if not depth_frame or not colour_frame:
            continue

        # Convert images to data points in np array
        depth_image = np.asanyarray(depth_frame.get_data())
        colour_image = np.asanyarray(colour_frame.get_data())

        # Apply colour map to depth image (converts image to 8-bit per pixel first)
        depth_colourmap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_RAINBOW)

        # Vertical Stack Image
        images = np.vstack((colour_image, depth_image, depth_colourmap))

        # Show cv window
        cv.namedWindow("Real Sense", cv.WINDOW_AUTOSIZE)
        cv.imshow("Real Sense", images)

        # Quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()


if __name__ == '__main__':
    ############################################################################
    # Standard + Skin Extraction
    ############################################################################
    # camera, width, left_rectangle_points, right_rectangle_points = setup_camera(RESIZE)
    # hand_recognition(
    #     camera,
    #     RESIZE,
    #     width,
    #     right_rectangle_points,
    #     left_rectangle_points
    # )

    # run_hand_segmentation(camera, (10, 100, 225, 350), 0.2)

    ############################################################################
    # RealSense
    ############################################################################
    rs_pipeline = setup_rs_pipeline((640, 480), 30)
    hand_recognition_depth(rs_pipeline)

    cv.destroyAllWindows()
