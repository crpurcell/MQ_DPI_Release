#=============================================================================#
#                                                                             #
# MODIFIED: 06-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import cv2

#-----------------------------------------------------------------------------#
class SimplePreprocessor:
    """
    Class containing methods to transform an image prior to using with 
    machine learning algorithms.
    """
    
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width    # target image width
        self.height = height  # target image height
        self.inter = inter    # interpolarion, INTER_AREA=3
        
    def preprocess(self, image):
        # Resize the image to a fixed size, ignoring the native aspect ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.inter)
