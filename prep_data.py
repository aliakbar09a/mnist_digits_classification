from tensorflow.python import keras


num_classes = 10
def prep(raw, train_size, validation_size):
    x = raw[0]
    y = raw[1]
    out_y = keras.utils.to_categorical(y, num_classes)
