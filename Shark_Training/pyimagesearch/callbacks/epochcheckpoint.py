#=============================================================================#
#                                                                             #
# MODIFIED: 12-Oct-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.callbacks import Callback
import os

#-----------------------------------------------------------------------------#
class EpochCheckpoint(Callback):
    """Class to save a HDF model file every N epochs."""

    def __init__(self, outputPath, every=5, startAt=0):

        # Call the parent constructor
        super(Callback, self).__init__()
        self.outputPath = outputPath
        self.every = every              # Save model ever N epochs
        self.intEpoch = startAt         # Starting epoch

    def on_epoch_end(self, epoch, logs={}):

        # Check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                                  "epoch_{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite=True)

        # Increment the internal epoch counter
        self.intEpoch += 1
