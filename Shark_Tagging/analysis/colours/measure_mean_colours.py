#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import numpy as np
import math as m
import progressbar
import colorsys
import cv2
import os
import sys
import glob
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-a", "--aperFrac", type=float, default=0.25,
                help="fraction of the image to cover with aperture [0.25].")
args = vars(ap.parse_args())
def main():
    annFile = os.path.join(args['dataset'], "BOXES.csv")
    print("[INFO] reading {}".format(annFile))
    annTab = pd.read_csv(annFile, header=None, skipinitialspace=True,
                         names=["imageName","x1","y1","x2","y2","label",
                                "nXpix", "nYpix", "date", "location", "qual"])
    inPath = os.path.join(args["dataset"], annTab.iloc[0]["imageName"])
    imgBGR = cv2.imread(inPath)
    nYpix, nXpix, nChans = imgBGR.shape
    print("[INFO] images have dimensions (x, y, z) = ({:d}, {:d}, {:d})"
          .format(nXpix, nYpix, nChans))
    if False:
        annTab = annTab.iloc[:10]
    xRad = int(m.floor(nXpix * args["aperFrac"] / 2))
    yRad = int(m.floor(nYpix * args["aperFrac"] / 2))
    print("[INFO] calculating median detection positions", flush=True)
    annTab["xBoxMed"] = (annTab.x1 + annTab.x2) / 2.0
    annTab["yBoxMed"] = (annTab.y1 + annTab.y2) / 2.0
    grpByFrame = annTab.groupby(annTab.imageName)
    aperTab = annTab.groupby(annTab.imageName).median()
    aperTab.drop(columns=["x1", "y1", "x2", "y2"], inplace=True)
    nFrames = len(aperTab)
    nBoxes = len(annTab)
    aperTab["x1"] = np.int32(aperTab["xBoxMed"] - xRad)
    aperTab["x2"] = np.int32(aperTab["xBoxMed"] + xRad)
    aperTab["y1"] = np.int32(aperTab["yBoxMed"] - yRad)
    aperTab["y2"] = np.int32(aperTab["yBoxMed"] + yRad)
    msk = aperTab["x1"] <= 0
    aperTab.loc[msk, "x2"] -= aperTab[msk]["x1"]
    aperTab.loc[msk, "x1"] -= aperTab[msk]["x1"]
    msk = aperTab["y1"] <= 0
    aperTab.loc[msk, "y2"] -= aperTab[msk]["y1"]
    aperTab.loc[msk, "y1"] -= aperTab[msk]["y1"]
    print("[INFO] measuring {:d} frames with {:d} boxes"
          .format(nFrames, nBoxes))
    widgets = ["Measuring Colours: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(aperTab),
                                   widgets=widgets).start()
    rowsLst = []
    i = 0
    for imageName, row in aperTab.iterrows():
        i += 1
        inPath = os.path.join(args["dataset"], imageName)
        imgBGR = cv2.imread(inPath)
        try:
            imgBGR = imgBGR[int(row["y1"]): int(row["y2"]),
                            int(row["x1"]): int(row["x2"]) , :]
        except Exception:
            pass
        (B, G, R) = cv2.mean(imgBGR)[:3]
        (H, S, V) = colorsys.rgb_to_hsv(R, G, B)
        V /= 255.0
        rowDict = {"imageName": imageName,
                   "R":        R,
                   "G":        G,
                   "B":        B,
                   "H":        H,
                   "S":        S,
                   "V":        V}
        rowsLst.append(rowDict)
        pbar.update(i)
    pbar.finish()
    print("[INFO] creating table of frame colours ...", flush=True)
    frmColTab = pd.DataFrame(rowsLst, columns=["imageName", "R", "G", "B",
                                               "H", "S", "V"])
    print("[INFO] merging to create table of detection colours", flush=True)
    boxColTab = pd.merge(annTab, frmColTab, left_on="imageName",
                         right_on="imageName", how="left")
    outFile = os.path.join(args["dataset"], "colours_by_frame.hdf5")
    print("[INFO] saving a colour table to '{}'".format(outFile))
    frmColTab.to_hdf(outFile, key="colourTab", mode="w")
    outFile = os.path.join(args["dataset"], "colours_by_box.hdf5")
    print("[INFO] saving a colour table to '{}'".format(outFile))
    boxColTab.to_hdf(outFile, key="colourTab", mode="w")
if __name__ == "__main__":
    main()
