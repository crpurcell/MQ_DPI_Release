#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import shutil
import argparse
import json
import gc
import cv2
import numpy as np
from random import shuffle
import math as m
import pandas as pd
def main():
    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-f", "--file", default="",
                    help="filename or partial to filter on [none]")
    ap.add_argument("-l", "--labels", default=[], nargs='+',
                    help="list of allowed class labels []")
    ap.add_argument("-s", "--scale", type=float, default=3,
                    help="scale the image window by dividing by [3].")
    ap.add_argument("-q", "--quality", type=float, default=1,
                    help="filter by minimum quality in range 1-10 [1].")
    ap.add_argument("-r", "--random", action="store_true",
                    help="randomise the frame order")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print extra information")
    args = vars(ap.parse_args())
    boxFilePath = os.path.join(args["dataset"], "BOXES.csv")
    if os.path.exists(boxFilePath):
        print("[INFO] loading box coordinates")
        boxDF = pd.read_csv(boxFilePath, header=None, index_col=0,
                            skipinitialspace=True,
                            names=["x1", "y1", "x2", "y2", "label",
                                   "nXpix", "nYpix", "date", "location",
                                   "qual"])
    print("[INFO] table contains {:d} entries".format(len(boxDF)))
    uniqueLabels = boxDF.label.unique().tolist()
    print("[INFO] Labels in dataset: {}".format(uniqueLabels))
    if len(args["file"]) > 0:
        print("[INFO] filtering for file(s)")
        boxDF = boxDF[boxDF.index.str.contains(args["file"])]
        print("[INFO] table contains {:d} entries".format(len(boxDF)))
    if len(args["labels"]) > 0:
        print("[INFO] filtering for labels: {}".format(args["labels"]))
        boxDF = boxDF[boxDF["label"].isin(args["labels"])]
        print("[INFO] table contains {:d} entries".format(len(boxDF)))
    if args["quality"] > 1:
        print("[INFO] filtering for quality >= {}".format(args["quality"]))
        boxDF = boxDF[boxDF["qual"] >=  args["quality"]]
        print("[INFO] table contains {:d} entries".format(len(boxDF)))
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    print("Press any key to advance to next image  or <q> to quit.")
    imgLst = boxDF.index.unique().tolist()
    if args["random"]:
        shuffle(imgLst)
    for imgFileName in imgLst:
        imgPath = os.path.join(args["dataset"], imgFileName)
        image = cv2.imread(imgPath)
        imgHeight, imgWidth = image.shape[:2]
        cv2.putText(image, imgFileName, (20, 80), cv2.FONT_HERSHEY_DUPLEX,
                    1.8, (0,0,0))
        if args["verbose"]:
            print(imgFileName)
        for indx, row in boxDF.loc[[imgFileName]].iterrows():
            x1, y1, x2, y2, label, nXpix, nYpix, _, _, qual = row.tolist()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            annStr = label + " " + str(qual)
            if args["verbose"]:
                print(annStr)
            cv2.putText(image, annStr, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.resizeWindow('Frame', int(imgWidth / args["scale"]),
                             int(imgHeight / args["scale"]))
        cv2.imshow("Frame", image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
if __name__ == "__main__":
    main()
