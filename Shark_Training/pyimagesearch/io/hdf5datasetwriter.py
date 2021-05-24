#=============================================================================#
#                                                                             #
# MODIFIED: 13-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import h5py
import os

#-----------------------------------------------------------------------------#
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):

        # Don't overwrite the existing datasets
        if os.path.exists(outputPath):
            raise ValueError("The output dataset already exists. Please "
                             "rename or delete:\n[{}]".format(outputPath))

        # Open the HDF5 database for writing and create two datasets:
        # ["images"] = binary data,   ["labels"] = integer labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # Initialize the memory buffer and local variables
        self.buffer = {"data": [], "labels": []}
        self.bufSize = bufSize        # Size of the buffer (memory used)
        self.idx = 0                  # Current index being accessed

    def add(self, rows, labels):
        """Add rows of features (images) to the buffer"""

        # Add rows to the data and labels lists
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # Flush the buffer if the memory limit is reached
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        """Flux the buffer to the HDF file"""

        # Write the buffers to disk
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]

        # Update current index and clear the buffer
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        """Create a dataset to store the class label strings"""

        dt = h5py.special_dtype(vlen=str) # 'vlen=unicode' for Py2.7
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),),
                                          dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        """Flush remaining buffer and close the HDF5 file"""
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
