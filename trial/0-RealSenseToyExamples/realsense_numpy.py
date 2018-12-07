import numpy as np
depth = frames.get_depth_frame()
depth_data = depth.as_frame().get_data()
np_image = np.asanyarray(depth_data)