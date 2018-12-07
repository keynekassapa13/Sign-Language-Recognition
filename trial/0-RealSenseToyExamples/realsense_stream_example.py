# First import the library
import pyrealsense2 as rs


# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

while True:
    # Create a pipeline object. This object configures the streaming camera and owns it's handle
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth: continue

    # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
    coverage = [0]*64
    for y in range(480):
        for x in range(640):
            dist = depth.get_distance(x, y)
            print(dist)
            if 0 < dist and dist < 1:
                cov_index = (int) (x/10)
                coverage[cov_index] += 1

        if y%20 is 19:
            line = ""
            for c in coverage:
                index = (int) (c/25)
                line += " .:nhBXWW"[index]
            coverage = [0]*64
            print(line)
