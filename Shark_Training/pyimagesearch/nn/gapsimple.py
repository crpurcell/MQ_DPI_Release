#=============================================================================#
#                                                                             #
# MODIFIED: 25-Jan-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.layers import Dense
from keras.models import Model

#-----------------------------------------------------------------------------#
class GAPSimple:
    @staticmethod
    def build(inTensor, classes, D, dropout=0.2):

        # Build a simple network with one hidden dense layer
        if D>0:
            gap = GlobalAveragePooling2D(name="gap")(inTensor)
            hidden = Dense(D, activation="relu")(gap)
            dropout = Dropout(dropout)(hidden)
            outputs = Dense(classes, activation="softmax")(dropout)
        
        # Or build a simple network with only a GAP layer
        else:
            gap = GlobalAveragePooling2D(name="gap")(inTensor)
            outputs = Dense(classes, activation="softmax")(gap)
        
        # Convert the network into a model
        model = Model(inputs=inTensor, outputs=outputs)

        return model
