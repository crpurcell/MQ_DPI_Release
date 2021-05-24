#!/usr/bin/env python
import numpy as np
import argparse
from PIL import Image
import imutils
import cv2
import os
import pprint
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="base path for TFlite detection model")
ap.add_argument("-l", "--labels", required=True,
                help="labels file")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability used to filter weak detections")
ap.add_argument("-D", "--debug", action="store_true",
                help="print debug information [False]")
args = vars(ap.parse_args())
def main():
    with open(args["labels"], "r") as fh:
        labelLst = fh.read().splitlines()[1:]
    print("[INFO] loading the detection model '{}'".format(args["model"]))
    interpreter = tf.contrib.lite.Interpreter(model_path=args["model"])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    imgShape = input_details[0]["shape"][2:0:-1]
    if args["debug"]:
        print("\n[INFO] input details:")
        pprint.pprint(input_details)
        print("\n[INFO] output details:")
        pprint.pprint(output_details)
    print("[INFO] loading and resizing image to {}".format(imgShape))
    img = Image.open(args["image"])
    img.load()
    img = img.resize(imgShape, Image.ANTIALIAS)
    data = np.asarray(img, dtype="int32")
    input_data = np.expand_dims(data,0).astype(np.uint8)
    print("[INFO] running inference ...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    labels = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    boxes = np.squeeze(boxes)
    labels = np.squeeze(labels.astype(np.int32))
    scores = np.squeeze(scores)
    image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]
    dropCount = 0
    for (box, score, label) in zip(boxes, scores, labels):
        if score < args["min_confidence"] or  score > 1:
            dropCount += 1
            continue
        (startY, startX, endY, endX) = box
        startX = int(startX * W)
        startY = int(startY * H)
        endX = int(endX * W)
        endY = int(endY * H)
        label = "{}: {:.2f}".format(labelLst[label], score)
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (255, 255, 255), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    print("[INFO] dropped {:d} detections for low score".format(dropCount))
    cv2.imshow("Output", image)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()
