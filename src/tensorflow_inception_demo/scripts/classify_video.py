import argparse
import cv2 as cv
import math
import sys
import os

'''
Adapted fromm: https://towardsdatascience.com/deep-learning-with-tensorflow-part-4-face-classification-and-video-inputs-fa078f22c1e5
'''

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# TODO: Might want to touch up the script into a class, though not essential to functionality.
parser = argparse.ArgumentParser(description="Classify from video stream.")
parser.add_argument('--tf_files', help='Path to tf_files', required=True)
args = vars(parser.parse_args())

# Get Labels, should fine: "tf_files/retrained_labels.txt"
retrained_labels = os.path.join(args['tf_files'], 'retrained_labels.txt')
label_lines = [line.rstrip() for line
               in tf.gfile.GFile(retrained_labels)]

# Tensorflow Graphs, should find: "tf_files/retrained_graph.pb"
retrained_graph = os.path.join(args['tf_files'], 'retrained_graph.pb')
with tf.gfile.FastGFile(retrained_graph, 'rb') as f:
    graph_def = tf.GraphDef()  # The graph-graph_def is a saved copy of a TensorFlow graph
    graph_def.ParseFromString(f.read())  # Parse serialized protocol buffer data into variable
    _ = tf.import_graph_def(graph_def,
                            name='')  # import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

# Prediction begins here
with tf.Session() as sess:
    camera = cv.VideoCapture(0)
    frame_rate = camera.get(5)
    top_prediction = ""

    while True:
        ret, frame = camera.read()
        frame = cv.flip(frame, 1)
        frame_ID = camera.get(1)

        if frame_ID % math.floor(frame_rate) == 0:
            cv.imwrite(filename="./testimageframe.jpg", img=frame)

            # Extract Image Data Here
            image_data = tf.gfile.FastGFile("./testimageframe.jpg", 'rb').read()

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, \
                                   {'DecodeJpeg/contents:0': image_data})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            # Output to console and frame
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
            print('=' * 10)

            top_prediction = f"{label_lines[top_k[0]]} : Score = {predictions[0][top_k[0]]}"

        # Show frame with prediction
        cv.putText(
            frame,
            top_prediction,
            (0, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA
        )
        cv.imshow("image", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()
