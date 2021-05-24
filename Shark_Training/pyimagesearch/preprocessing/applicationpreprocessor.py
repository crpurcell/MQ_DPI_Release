#=============================================================================#
#                                                                             #
# MODIFIED: 30-Dec-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

#-----------------------------------------------------------------------------#
class ApplicationPreprocessor:
    """
    Wrapper class to allow use of Keras application preprocessor with the
    HDF5 generator.
    """

    def __init__(self, preprocess_function):
        self.preprocess_function = preprocess_function

    def preprocess(self, image):
        return self.preprocess_function(image)

