# Tensorflow with Inception for Image Classification
See the following as references:
* [Building an Image Classifier using Tensorflow](https://medium.com/datadriveninvestor/building-an-image-classifier-using-tensorflow-3ac9ccc92e7c)
* [Tensorflow for Poets - Google Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)

## Dataset
Example flower dataset can be downloaded from: 
[Flower Dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)

Flower Dataset contains:
* 5 classes of flowers
* 700 images per type

Alternatively make your own dataset, with at least 100 images per class.

## Run this example
1. Put image dataset in tf_files/{dataset_name}. E.g., tf_files/flower_dataset
2. In train.sh set --image_dir=tf_files/{dataset_name}
3. Run python3 -m ./scripts/classify.py path_to_test_image
