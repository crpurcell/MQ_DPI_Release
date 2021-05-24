#=============================================================================#
#                                                                             #
# MODIFIED: 15-Jan-2019 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

#-----------------------------------------------------------------------------#
class RBGPreprocessor:
    """
    Class used to convert from BGR to RGB ordering of colours. The default
    colour ordering used by OpenCV (and the PyImageSearch HDF5 functions) is
    BGR. Pre-trained Keras networks expect RGB odering, so this preprocessor
    can be used to do the conversion.
    """

    def __init__(self):
        pass

    def preprocess(self, image):
        return image[..., ::-1]
