# Tensorflow with Inception for Image Classification
See the following as references:
* [Building an Image Classifier using Tensorflow](https://medium.com/datadriveninvestor/building-an-image-classifier-using-tensorflow-3ac9ccc92e7c)
* [Google codelabs: Tensorflow for Poets - Google Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)
* [Google ImageNet Classifier](https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py)
* [Towards Data Science: Deep Learning with Tensorflow: Face Classification and Video Inputs](https://towardsdatascience.com/deep-learning-with-tensorflow-part-4-face-classification-and-video-inputs-fa078f22c1e5)

## Dataset
Example flower dataset can be downloaded from: 
[Flower Dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)

Flower Dataset contains:
* 5 classes of flowers
* 700 images per type

Alternatively make your own dataset, with at least 100 images per class.

## Run this example
1. Put image dataset in ./{dataset_name}. E.g., ./flower_dataset
2. In train.sh set --image_dir={dataset_name}
    * If you want to run tensorboard, use: tensorboard --logdir ./tf_files/training_summaries (assuming you have an alias)
3. Run python3 scripts/classify.py path_to_test_image
