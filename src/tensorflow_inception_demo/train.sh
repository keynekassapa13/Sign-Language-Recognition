python3 scripts/retrain.py \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=4000 \
  --model_dir=tf_files/models \
  --summaries_dir=tf_files/training_summaries \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --image_dir=/home/jcap/Projects/Summer1819/inception_datasets/gestures_subset
