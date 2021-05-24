#=============================================================================#
#                                                                             #
# MODIFIED: 08-Jan-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model

#-----------------------------------------------------------------------------#
class FCSimple:
    @staticmethod
    def build(inTensor, classes, D):

        # Build a simple network with one hidden layer
        flatten = Flatten(name="flatten")(inTensor)
        hidden = Dense(D, activation="relu")(flatten)
        dropout = Dropout(0.5)(hidden)
        outputs = Dense(classes, activation="softmax")(dropout)

        # Convert the network into a model
        model = Model(inputs=inTensor, outputs=outputs)

        return model
