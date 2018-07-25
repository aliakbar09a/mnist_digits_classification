# Digit Classification using Logistic Regression and CNN 
This is a project focusing on the classification of digits from 0-9 using Logistic Regression and Convolutional Neural Network.
The trained model is used to predict digits drawn on the captured frames from the webcam using object tracking.

## Getting Started

### Built with
The Logistic Regression is implemented using numpy and softmax function is used for multiclass classification.
The Convolutional Neural Network uses Keras API with tensorflow as backend.

<img src= "https://github.com/aliakbar09a/mnist_digits_classification/blob/master/softmax.png">

### Pretrained weights
Training examples = 8000, Test examples = 1000

**Logistic Regression**

Train accuracy = 92.1%, 
Test accuracy = 91.3%,
learning rate = 0.1

**CNN Model**

Train accuracy = 98.1%,
Test accuracy = 96.2%,
no. of epochs = 8

(Feel free to train the models on your own.)
### Prerequisites
Install Conda to resolve all requirements of python related dependencies.

## Usage
### Files usage
- LRmodel.py : Logistic Regression implemented using numpy
- conv_network.py : CNN model implemented using Keras API
- digit_recognizer.py : File to either train or load saved weights
- camera_pred.py : Used to test the models using webcam

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

Press **c** to clear the box.


