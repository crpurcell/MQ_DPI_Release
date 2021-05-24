#=============================================================================#
#                                                                             #
# MODIFIED: 19-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.utils import np_utils
import numpy as np
import h5py

#-----------------------------------------------------------------------------#
class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
                 binarize=True, classes=2, dataKey="images"):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.dataKey = dataKey

        # Open the HDF5 database for reading and query the number of entries
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0 

        # Loop infinitely (model will stop at desired number of epochs)
        while epochs < passes:

            # Loop over the HDF5 dataset in batches
            for i in np.arange(0, self.numImages, self.batchSize):

                # Reference a batch of images and labels in the HDF5 file
                images = self.db[self.dataKey][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # Vectorize the labels, if required
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)

                # Apply preprocessors to the images, if required
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    images = np.array(procImages)

                # Apply augmentation, if required
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,
                                        labels, batch_size=self.batchSize))

                # Yield a tuple of images and labels for the batch
                yield (images, labels)

            # Increment epoch counter
            epochs += 1

    def  close(self):
        # Close the HDF5 database
        self.db.close()
