#=============================================================================#
#                                                                             #
# MODIFIED: 05-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

#-----------------------------------------------------------------------------#
class SimpleNet:
    @staticmethod
    def build(width, height, depth, classes):

        # Initialize the model and channel order
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # First CONV(32) => RELU => POOL(2) layer set
        model.add(Conv2D(32, (3, 3), input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second CONV(32) => RELU => POOL(2) layer set
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third CONV(64) => RELU => POOL(2) layer set
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # FC(64) => RELU => DROP(0.5)
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
