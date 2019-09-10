# Steel-Defect-Detection-CNN
Detection, localization and classification of surface defects on a steel sheet using CNN. Using libraries Keras and sklearn.

## The Problem
The production of flat steel isn't a very perfect process. It can lead to a number of different categories of defects on its surface. The code localizes and classifies the various defects by training and testing on images from high frequency cameras.


## Data source
https://www.kaggle.com/c/severstal-steel-defect-detection/data

## Utility Functions
https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

## Sample visualization
We can visualize a sample image and its masks using this part of code.

## Model Architecture
The model is inspired from https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data.
This is a bit different as it predicts all four masks at the same time rather than one by one.

## Loss
We use Dice loss is used as a measure of loss. In general, dice loss works better on images than on single pixels.
  
