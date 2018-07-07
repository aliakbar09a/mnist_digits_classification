
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python import keras


num_classes = 10
img_rows, img_cols = 28, 28


def prep(data, batch_size):
    raw = data.train.next_batch(batch_size)
    x = raw[0]
    y = raw[1]
    out_y = keras.utils.to_categorical(y, num_classes)
    out_x = x.reshape(batch_size, img_rows, img_cols, 1)
    out_x = out_x/255
    return out_x, out_y


def convolution_model(train_x, train_y):
    conv_model = Sequential()
    # first layer with input shape (img_rows, img_cols, 1) and 12 filters
    conv_model.add(Conv2D(12, kernel_size=(3, 3), activation='relu',
                          input_shape=(img_rows, img_cols, 1)))
    # second layer with 12 filters
    conv_model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
    # third layer with 12 filers
    conv_model.add(Conv2D(12, kernel_size=(3, 3), activation='relu'))
    # flatten layer
    conv_model.add(Flatten())
    # adding a Dense layer
    conv_model.add(Dense(100, activation='relu'))
    # adding the final Dense layer with softmax
    conv_model.add(Dense(num_classes, activation='softmax'))

    # compile the model
    conv_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # fit the model
    conv_model.fit(train_x, train_y, batch_size=100, epochs=4,
                   validation_split=0.2)
    return conv_model
