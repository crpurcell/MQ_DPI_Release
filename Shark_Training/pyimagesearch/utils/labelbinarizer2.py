#=============================================================================#
#                                                                             #
# MODIFIED: 05-Sep-2018 by C. Purcell                                         #
#                                                                             #
# https://stackoverflow.com/questions/31947140/                               #
#             sklearn-labelbinarizer-returns-vector-when-there-are-2-classes  #
#                                                                             #
#=============================================================================#
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class LabelBinarizer2(LabelBinarizer):
    """
    LabelBinarizer class that has a consistent result when there are only
    two classes of object.
    """

    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((1-Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 1], threshold)
        else:
            return super().inverse_transform(Y, threshold)
