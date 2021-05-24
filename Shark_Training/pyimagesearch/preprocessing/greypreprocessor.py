#=============================================================================#
#                                                                             #
# MODIFIED: 15-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import numpy as np

#-----------------------------------------------------------------------------#
class GreyPreprocessor:

    def __init__(self, mode="Luma"):
        if mode == "sRGB":
             self.coeff = [0.2126, 0.7152, 0.0722]
        else:
            self.coeff = [0.299, 0.587, 0.114]

    def preprocess(self, image):
        greyPlane = np.dot(image, self.coeff)
        return np.stack((greyPlane,)*3, axis=-1)
