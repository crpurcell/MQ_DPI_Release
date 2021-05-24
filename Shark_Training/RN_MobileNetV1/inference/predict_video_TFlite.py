#!/usr/bin/env python
import numpy as np
import argparse
from PIL import Image
import imutils
import cv2
import os
import shutil
import pprint
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="base path for TFlite detection model")
ap.add_argument("-l", "--labels", required=True,
                help="labels file")
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", default=None,
                help="path to output video")
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
    imgShape = tuple(input_details[0]["shape"][2:0:-1])
    stream = cv2.VideoCapture(args["input"])
    writer = None
    frmNum = 1
    while True:
        (success, image) = stream.read()
        if not success:
            break
        (H, W) = image.shape[:2]
        original = image.copy()
        output = image.copy()
        imageBGR = cv2.resize(image, imgShape)
        image = cv2.cvtColor(imageBGR.copy(), cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(image, axis=0).astype(np.uint8)
        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                     (W, H), True)
        print("[INFO] running inference ...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])
        labels = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        boxes = np.squeeze(boxes)
        labels = np.squeeze(labels.astype(np.int32))
        scores = np.squeeze(scores)
        for (box, score, label) in zip(boxes, scores, labels):
            if score < args["min_confidence"] or  score > 1 :
                continue
            (startY, startX, endY, endX) = box
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)
            print(label)
            print(score)
            print(box)
        if args["debug"]:
            outPath = os.path.join("DEBUG", "FRM_Original",
                                   "FRM{:06d}.png".format(frmNum))
            cv2.imwrite(outPath, original)
            outPath = os.path.join("DEBUG", "FRM_Annotated",
                                   "FRM{:06d}.png".format(frmNum))
            cv2.imwrite(outPath, output)
            outPath = os.path.join("DEBUG", "FRM_Resized",
                                   "FRM{:06d}.png".format(frmNum))
            cv2.imwrite(outPath, imageBGR)
        if writer is None:
            cv2.imshow("Video", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            writer.write(output)
        frmNum +=1
    if writer is not None:
        writer.release()
    stream.release()
if __name__ == "__main__":
    main()
