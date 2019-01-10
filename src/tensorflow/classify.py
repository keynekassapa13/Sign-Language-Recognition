import tensorflow as tf
import sys
import os
import cv2
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
video_path = sys.argv[1]

label_lines = [line.rstrip() for line
               in tf.gfile.GFile("tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(
        graph_def,
        name=''
    )

with tf.Session() as sess:
    video_capture = cv2.VideoCapture(video_path)
    # frameRate = video_capture.get(5)
    i = 0
    while True:
        frame = video_capture.read()[1]

        # current frame number
        frameId = video_capture.get(1)

        # if (frameId % math.floor(frameRate) == 0):
        if True:
            i = i + 1
            cv2.imwrite(filename="screens/" + str(i) + ".png", img=frame);
            image_data = tf.gfile.FastGFile("screens/" + str(i) + ".png", 'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(
                softmax_tensor,
                {'DecodeJpeg/contents:0': image_data}
            )

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))

            print("\n\n")
            cv2.imshow("image", frame)

        interrupt = cv2.waitKey(1)

        # Quit by pressing 'q'
        if interrupt & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
