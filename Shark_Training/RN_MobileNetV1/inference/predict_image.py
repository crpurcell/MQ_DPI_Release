#!/usr/bin/env python
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
                help="labels file")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())
model = tf.Graph()
with model.as_default():
    graphDef = tf.GraphDef()
    with tf.gfile.GFile(args["model"], "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")
labelMap = label_map_util.load_labelmap(args["labels"])
numClasses = len(labelMap.item)
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=numClasses, use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)
COLOURS = np.random.uniform(0, 255, size=(numClasses, 3))
with model.as_default():
    with tf.Session(graph=model) as sess:
        imageTensor = model.get_tensor_by_name("image_tensor:0")
        boxesTensor = model.get_tensor_by_name("detection_boxes:0")
        scoresTensor = model.get_tensor_by_name("detection_scores:0")
        classesTensor = model.get_tensor_by_name("detection_classes:0")
        numDetections = model.get_tensor_by_name("num_detections:0")
        image = cv2.imread(args["image"])
        (H, W) = image.shape[:2]
        output = image.copy()
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        (boxes, scores, labels, N) = sess.run(
            [boxesTensor, scoresTensor, classesTensor, numDetections],
            feed_dict={imageTensor: image})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        labels = np.squeeze(labels)
        N = np.squeeze(N)
        print("[INFO] total number of detections =", N)
        print("[INFO] scores = ", scores[: int(N)])
        dropCount = 0
        for (box, score, label) in zip(boxes, scores, labels):
            if score < args["min_confidence"]:
                dropCount += 1
                continue
            (startY, startX, endY, endX) = box
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)
            label = categoryIdx[label]
            idx = int(label["id"]) - 1
            label = "{}: {:.2f}".format(label["name"], score)
            cv2.rectangle(output, (startX, startY), (endX, endY),
                          COLOURS[idx], 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(output, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOURS[idx], 2)
        print("[INFO] dropped {:d} detections for low score".format(dropCount))
        cv2.imshow("Output", output)
        cv2.waitKey(0)
