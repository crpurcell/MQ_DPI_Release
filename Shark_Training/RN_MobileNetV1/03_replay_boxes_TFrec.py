#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import imp
import numpy as np
from PIL import Image
import tensorflow as tf
tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from object_detection.utils import label_map_util
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inFile", required=True,
                help="path to record file")
ap.add_argument("-l", "--labelFile", default=None,
                help="path to labels file 'classes.pbtxt'")
ap.add_argument("-s", dest="scaleFac", type=float, default=3,
                help="shrink native resolution factor [3].")
args = vars(ap.parse_args())
def main():
    parsedDataset = parse_record(args["inFile"])
    categoryIdx = None
    if args["labelFile"]:
        labelMap = label_map_util.load_labelmap(args["labelFile"])
        numClasses = len(labelMap.item)
        categories = label_map_util.convert_label_map_to_categories(
            labelMap, max_num_classes=numClasses, use_display_name=True)
        categoryIdx = label_map_util.create_category_index(categories)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    print("\n\nPress any key to advance to next image  or <q> to quit.\n\n")
    for example in parsedDataset:
        imageDecoded = tf.image.decode_image(example['image/encoded']).numpy()
        height = example['image/height'].numpy()
        width = example['image/width'].numpy()
        filename = example['image/filename'].numpy()
        imgFormat = example['image/format'].numpy()
        x1norm =  tf.sparse_tensor_to_dense(
            example['image/object/bbox/xmin'], default_value=0).numpy()
        x2norm =  tf.sparse_tensor_to_dense(
            example['image/object/bbox/xmax'], default_value=0).numpy()
        y1norm =  tf.sparse_tensor_to_dense(
            example['image/object/bbox/ymin'], default_value=0).numpy()
        y2norm =  tf.sparse_tensor_to_dense(
            example['image/object/bbox/ymax'], default_value=0).numpy()
        labels =  tf.sparse_tensor_to_dense(
            example['image/object/class/label'], default_value=0).numpy()
        numBoxes = len(labels)
        widthScreen = int(width / args["scaleFac"])
        heightScreen = int(height / args["scaleFac"])
        cv2.resizeWindow('Frame', widthScreen, heightScreen)
        image = np.array(imageDecoded, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if numBoxes > 0:
            x1 = np.int64(x1norm * width)
            x2 = np.int64(x2norm * width)
            y1 = np.int64(y1norm * height)
            y2 = np.int64(y2norm * height)
            for i in range(numBoxes):
                bbox = (x1[i], y1[i], x2[i], y2[i])
                cv_bbox(image, bbox, color=(0, 255, 255), line_width=2)
                if args["labelFile"]:
                    labelStr = categoryIdx[labels[i]]["name"]
                    labelPos = (bbox[2] +2, bbox[1] - 10)
                    cv2.putText(image,
                                labelStr,
                                labelPos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 255), 2);
        cv2.imshow("Frame", image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
def cv_bbox(image, bbox, color = (255, 255, 255), line_width = 2):
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  color, line_width)
    return
def parse_record(recordFile):
    dataset = tf.data.TFRecordDataset(recordFile)
    featureDict = {
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/object/bbox/xmin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(tf.float32),
        "image/object/class/label": tf.VarLenFeature(tf.int64),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/format": tf.FixedLenFeature([], tf.string)
    }
    def _parse_function(example_proto):
        return tf.parse_single_example(example_proto, featureDict)
    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset
if __name__ == "__main__":
    main()
