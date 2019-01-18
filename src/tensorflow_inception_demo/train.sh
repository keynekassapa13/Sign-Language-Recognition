python3 scripts/retrain.py \
  --bottleneck_dir=/home/jcap/Projects/Summer1819/tf_files/gestures_100/bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=/home/jcap/Projects/Summer1819/tf_files/gestures_100/models \
  --summaries_dir=/home/jcap/Projects/Summer1819/tf_files/gestures_100/training_summaries \
  --output_graph=/home/jcap/Projects/Summer1819/tf_files/gestures_100/retrained_graph.pb \
  --output_labels=/home/jcap/Projects/Summer1819/tf_files/gestures_100/retrained_labels.txt \
  --image_dir=/home/jcap/Projects/Summer1819/inception_datasets/gestures_subset_small_100
