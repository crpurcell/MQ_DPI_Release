#=============================================================================#
#                                                                             #
# MODIFIED: 06-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import os
import numpy as np
import cv2

#-----------------------------------------------------------------------------#
class SimpleDatasetLoader:
    """
    Class containing methods to load an image dataset and apply transformation
    functions prior to using with machine learning algorithms.
    """
    
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        # Loop over the input images, extracting class labels
        # Assuming /path/to/dataset/{class}/{image}.jpg
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # Run any pre-processor functions in-place
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # Append image and label to disk
            data.append(image)
            labels.append(label)

            # Show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))
