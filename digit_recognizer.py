import input_data
import conv_network
import LRmodel
import numpy as np
from tensorflow.python import keras

num_classes = 10
img_rows, img_cols = 28, 28

# loading the MNIST dataset
mnist = input_data.read_data_sets("data/", one_hot=False)
train_size, test_size = 8000, 1000
train_data = mnist.train.next_batch(train_size)
test_data = mnist.test.next_batch(test_size)


def prep_data(data, train_size):
    '''
    :param data: data with both features and labels
    :param train_size: no. of examples in training data
    :return: X, y
    '''
    x = data[0]
    y = data[1]
    out_y = keras.utils.to_categorical(y, num_classes)
    out_x = x.reshape(train_size, img_rows, img_cols, 1)
    return out_x, out_y


# preparing training and test data
train_x, train_y = prep_data(train_data, train_size)
test_x, test_y = prep_data(test_data, test_size)

# training and testing LR model
weights, base = LRmodel.model(train_x, train_y, test_x, test_y,
                              iters=1500, alpha=0.1, print_cost=True)
# training and testing CNN model
model_conv, pre_train_y, pre_test_y = conv_network.model(train_x, train_y,
                                                         test_x, test_y, epoch=5)
