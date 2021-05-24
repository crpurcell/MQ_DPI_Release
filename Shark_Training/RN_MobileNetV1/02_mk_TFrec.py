#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import progressbar
import argparse
import progressbar
import json
import os
import sys
import glob
import csv
import pickle
from string import Template
from PIL import Image
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pyimagesearch.utils.tfannotation import TFAnnotation
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--indir", required=True,
                help="path to input directory")
ap.add_argument("-o", "--outdir", required=True,
                help="path to output record directory")
args = vars(ap.parse_args())
def main():
    trainPickle = os.path.join(args["indir"], "ann_train.pickle")
    with open(trainPickle, "rb") as fh:
        trainGrpLst = pickle.load(fh)
    validPickle = os.path.join(args["indir"], "ann_valid.pickle")
    if os.path.exists(validPickle):
        with open(validPickle, "rb") as fh:
            validGrpLst = pickle.load(fh)
    else:
        validGrpLst = []
    imgPathJSON = os.path.join(args["indir"], "imgpath.json")
    with open(imgPathJSON, "r") as fh:
        pDict = json.load(fh)
    classCSV = os.path.join(args["indir"], "classes.csv")
    classDF = pd.read_csv(classCSV, header=None, skipinitialspace=True,
                          index_col=0, names=["indx"])
    if os.path.exists(args["outdir"]):
        exit("[WARN] Record directory '{}' already exists. Exiting ...".             format(args["outdir"]))
    else:
        os.makedirs(args["outdir"])
    classFile = os.path.join(args["outdir"], "classes.pbtxt")
    f = open(classFile, "w")
    for i, row in classDF.iterrows():
        item = ("item {\n"
                "  id: " + str(int(row) + 1) + "\n"
                "  name: '" + row.name + "'\n"
                "}\n")
        f.write(item)
    f.close()
    trainRecFile = os.path.join(args["outdir"], "training.record")
    grp_to_tfrec(trainRecFile, trainGrpLst, pDict["imgpath"], classDF)
    if len(validGrpLst) > 0:
        validRecFile = os.path.join(args["outdir"], "testing.record")
        grp_to_tfrec(validRecFile, validGrpLst, pDict["imgpath"], classDF)
    absExptPath = os.path.abspath(args["indir"])
    absRecPath = os.path.abspath(args["outdir"])
    print("[INFO] writing 'recordpath.json' into the experiment directory")
    outJSON = os.path.join(absExptPath, "recordpath.json")
    with open(outJSON, "w") as fh:
        json.dump({"recordpath": absRecPath}, fh)
def grp_to_tfrec(outFilePath, grpLst, dataPath, classDF):
    print("[INFO] processing '{}'...".format(outFilePath))
    writer = tf.python_io.TFRecordWriter(outFilePath)
    total = 0
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(grpLst),
                                   widgets=widgets).start()
    for i, e in enumerate(grpLst):
        pbar.update(i)
        imageName = e.iloc[0]["imageName"]
        imagePath = os.path.join(dataPath, imageName)
        if not os.path.exists(imagePath):
            continue
        encoded = tf.gfile.GFile(imagePath, "rb").read()
        encoded = bytes(encoded)
        pilImage = Image.open(imagePath)
        (w, h) = pilImage.size[:2]
        encoding = imageName[imageName.rfind(".") + 1:]
        tfAnnot = TFAnnotation()
        tfAnnot.image = encoded
        tfAnnot.encoding = encoding
        tfAnnot.filename = imageName
        tfAnnot.width = w
        tfAnnot.height = h
        isNeg = False
        if "NEG" in e["label"].to_list():
            isNeg = True
        for j, row in e.iterrows():
            xMin = max(row["x1"] / w, 0)
            xMax = min(row["x2"] / w, 1)
            yMin = max(row["y1"] / h, 0)
            yMax = min(row["y2"] / h, 1)
            tfAnnot.xMins.append(xMin)
            tfAnnot.xMaxs.append(xMax)
            tfAnnot.yMins.append(yMin)
            tfAnnot.yMaxs.append(yMax)
            tfAnnot.textLabels.append(row["label"].encode("utf8"))
            tfAnnot.classes.append(int(classDF.loc[row["label"]]) + 1)
            tfAnnot.difficult.append(0)
            total += 1
            if isNeg:
                break
        features = tf.train.Features(feature=tfAnnot.build(isNeg))
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
    writer.close()
    print("[INFO] {} boxes saved".format(total))
if __name__ == "__main__":
    main()
