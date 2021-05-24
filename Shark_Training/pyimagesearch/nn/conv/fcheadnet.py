#=============================================================================#
#                                                                             #
# MODIFIED: 13-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

#-----------------------------------------------------------------------------#
class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):

        # Build a FC layer on top of an existing network
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # Add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel
