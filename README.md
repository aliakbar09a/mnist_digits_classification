# Digit Classification using Logistic Regression and CNN 
This is a project focusing on the classification of digits from 0-9 using Logistic Regression and Convolutional Neural Network.
The trained model is used to predict digits drawn on the captured frames from the webcam using object tracking.

## Getting Started

### Built with
The Logistic Regression is implemented using numpy and softmax function is used for multiclass classification.
The Convolutional Neural Network uses Keras API with tensorflow as backend.

### Prerequisites
Install Conda to resolve all requirements of python related dependencies.

## Usage

### Training the models
To Train the models on your own, delete the weights folder and run digit_recognizer.py
```
python digit_recognizer.py
```
(If any of the files in weights folder is not present, the models will be trained again.)

### Testing using Camera
Run camera_pred.py (Use a green colored object to draw digit inside the red box).
```
python camera_pred.py
```
<img src= "https://github.com/aliakbar09a/mnist_digits_classification/blob/master/sample.gif">
Press *c* to clear the box.


