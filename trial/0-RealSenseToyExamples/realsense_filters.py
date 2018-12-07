import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()

    for x in range(10):
        frame = frames[x]
        frame = decimation.process(frame)
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = disparity_to_depth.process(frame)

    colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
    plt.imshow(colorized_depth)