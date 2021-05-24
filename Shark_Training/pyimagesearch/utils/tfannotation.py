#=============================================================================#
#                                                                             #
# MODIFIED: 13-May-2020 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

# These are TF utility functions to encode python objects
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature

#-----------------------------------------------------------------------------#
class TFAnnotation:
    """Class containing information about an image for TensorFlow"""

    def __init__(self):

        # Bounding boxes and labels (multiple per image
        self.xMins = []           # Bounding box lists
        self.xMaxs = []           #
        self.yMins = []           #
        self.yMaxs = []           #
        self.textLabels = []      # List of class labels
        self.classes = []         # List of classes
        self.difficult = []       # 'difficult' flag (set to 0 almost always)

        # The image and ancillary information
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self, isNeg=False):

        # Encode the attributes using their respective TensorFlow encoders
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.encoding.encode("utf8"))
        image = bytes_feature(self.image)
        xMins = float_list_feature(self.xMins)
        xMaxs = float_list_feature(self.xMaxs)
        yMins = float_list_feature(self.yMins)
        yMaxs = float_list_feature(self.yMaxs)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)

        # Construct a TensorFlow-compatible attribute dictionary
        # Negative images don't omit the boxes boxes
        if isNeg:
            data = {
                "image/height": h,
                "image/width": w,
                "image/filename": filename,
                'image/source_id': filename,
                "image/encoded": image,
                "image/format": encoding
            }
        else:
            data = {
                "image/height": h,
                "image/width": w,
                "image/filename": filename,
                'image/source_id': filename,
                "image/encoded": image,
                "image/format": encoding,
                "image/object/bbox/xmin": xMins,
                "image/object/bbox/xmax": xMaxs,
                "image/object/bbox/ymin": yMins,
                "image/object/bbox/ymax": yMaxs,
                "image/object/class/text": textLabels,
                "image/object/class/label": classes,
                "image/object/difficult": difficult
            }

        return data
