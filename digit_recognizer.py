import input_data
import conv_network
import LRmodel
import numpy as np
from tensorflow.python import keras
import os

num_classes = 10
img_rows, img_cols = 28, 28


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


# creating weights folder if deleted
if os.path.isdir('weights') == False:
    os.mkdir('weights')

# list of filename of the weights and saved model
weights_files = os.listdir('weights')
# files that should be there to load the models
correct_files = ['LR_params.npy', 'cnn_accuracy.npy', 'cnn_model.json', 'cnn_weights.h5']

if weights_files == correct_files:
    # loading the LR weights
    LR_params = np.load('weights/LR_params.npy')
    # loading the cnn model using json
    CNN_acc = np.load('weights/cnn_accuracy.npy')
    json_file = open('weights/cnn_model.json', 'r')
    model = json_file.read()
    json_file.close()
    model_conv = keras.models.model_from_json(model)
    # loading the cnn weights into the models
    model_conv.load_weights('weights/cnn_weights.h5')
    # printing LR saved models parameters
    print('Trained Logistic Regression')
    print("Logistic Regression Train accuracy : ", LR_params.item().get('train_accuracy'))
    print("Logistic Regression Test accuracy : {}%".format(LR_params.item().get('test_accuracy')))
    print("Training size : {}, alpha : {}, iterations : {}\n\n".format(LR_params.item().get('train_size'),
                                                                       LR_params.item().get('alpha'), LR_params.item().get('iters')))
    # printing CNN saved models parameters
    print('Trained Convolution Neural Network')
    print('Train accuracy : ', CNN_acc.item().get('train_accuracy'))
    print('Test accuracy : ', CNN_acc.item().get('test_accuracy'))
    print('No. of epochs used = ', CNN_acc.item().get('epoch'))
else:
    # loading the MNIST dataset
    mnist = input_data.read_data_sets("data/", one_hot=False)
    train_size, test_size = 8000, 1000
    train_data = mnist.train.next_batch(train_size)
    test_data = mnist.test.next_batch(test_size)

    # preparing training and test data
    train_x, train_y = prep_data(train_data, train_size)
    test_x, test_y = prep_data(test_data, test_size)

    # training and testing LR model
    LR_params = LRmodel.model(train_x, train_y, test_x, test_y,
                              iters=2000, alpha=0.1, print_cost=True)
    # training and testing CNN model
    model_conv, CNN_accuracy = conv_network.model(train_x, train_y,
                                                  test_x, test_y, epoch=8)
    np.save('weights/LR_params.npy', LR_params)
    np.save('weights/cnn_accuracy.npy', CNN_accuracy)
    # converting model to json
    json_model = model_conv.to_json()
    # saving the json model
    with open('weights/cnn_model.json', 'w') as json_file:
        json_file.write(json_model)
    # saving weights of the cnn models
    model_conv.save_weights('weights/cnn_weights.h5')
