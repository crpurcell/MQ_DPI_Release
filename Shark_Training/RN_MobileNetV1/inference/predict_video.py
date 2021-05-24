#!/usr/bin/env python
from object_detection.utils import visualization_utils as vis_util
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
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", default=None,
                help="path to output video")
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
COLOURS =  [(255, 255, 255)]* int(numClasses)
with model.as_default():
    with tf.Session(graph=model) as sess:
        stream = cv2.VideoCapture(args["input"])
        writer = None
        while True:
            (success, image) = stream.read()
            if not success:
                break
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            (H, W) = image.shape[:2]
            if W > H and W > 1000:
                image = imutils.resize(image, width=1000)
            elif H > W and H > 1000:
                image = imutils.resize(image, height=1000)
            (H, W) = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            if args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                         (W, H), True)
            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            for (box, score, label) in zip(boxes, scores, labels):
                if score < args["min_confidence"]:
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOURS[idx], 1)
            if writer is None:
                cv2.imshow("Video", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                writer.write(output)
        if writer is not None:
            writer.release()
        stream.release()
