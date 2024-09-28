
Honey Bee Detection Model - v4 883base-x7aug
==============================

This dataset was exported via roboflow.com on November 13, 2023 at 12:44 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 4575 images.
Workers-drones-queens-pollenbees are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:

The following augmentation was applied to create 7 versions of each source image:

The following transformations were applied to the bounding boxes of each image:
* 50% probability of horizontal flip
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 10 pixels
* Salt and pepper noise was applied to 5 percent of pixels


