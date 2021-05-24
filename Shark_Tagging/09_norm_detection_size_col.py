#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import numpy.ma as ma
import numpy.random as rd
from numpy import inf
from scipy.stats import norm
from scipy.special import erfinv
import progressbar
import gc
import cv2
import os
import glob
import json
import argparse
import math as m
import copy
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-s", "--target-size", required=True,
                help="JSON file encoding target size distributions")
ap.add_argument("-c", "--target-colour", required=True,
                help="JSON file encoding target colour distributions")
ap.add_argument("-o", "--outdir", default="OUT",
                help="path to the output directory [OUT]")
ap.add_argument("-l", "--labels", default=[], nargs='+',
                help="list of allowed class labels []")
ap.add_argument("-i", "--imageHeight", type=int, default=1080,
                help="height of output images, in pixels [1080].")
ap.add_argument("-r", "--resolveLim", type=float, default=2.0,
                help="limiting over-resolution factor for zooms [2].")
ap.add_argument("-p", "--save-plots", action="store_true",
                help="save plots of the shift for each class [no]")
ap.add_argument("-D", "--debug", action="store_true",
                help="debug mode (annotates full field)")
ap.add_argument("-f", "--file", default="",
                help="filename or partial to filter on [none]")
ap.add_argument("-v", "--verbose", action="store_true",
                help="print messages rather than a progress bar")
args = vars(ap.parse_args())
def main():
    annFile = os.path.join(args['dataset'], "BOXES.csv")
    print("[INFO] reading {}".format(annFile))
    annTab = pd.read_csv(annFile, header=None, skipinitialspace=True,
                         names=["imageName","x1","y1","x2","y2","label",
                                "nXpix", "nYpix", "date", "location", "qual"])
    if os.path.exists(args["target_size"]):
        with open(args["target_size"], 'r') as FH:
            sizeDistDict = json.load(FH)
    if os.path.exists(args["target_colour"]):
        with open(args["target_colour"], 'r') as FH:
            colDistDict = json.load(FH)
    inFile = os.path.join(args["dataset"], "colours_by_box.hdf5")
    colTab = pd.read_hdf(inFile, "colourTab")
    if len(args["file"]) > 0:
        print("[INFO] filtering for file(s)")
        annTab = annTab[annTab["imageName"].str.contains(args["file"])]
    allowedLabels = args["labels"]
    if len(allowedLabels) > 0:
        print("[INFO] restricting labels to {}".format(allowedLabels))
        annTab = annTab[annTab["label"].isin(allowedLabels)]
        colTab = colTab[colTab["label"].isin(allowedLabels)]
    labels = list(annTab["label"].unique())
    plotDir = os.path.join(args["outdir"], "PLOTS")
    if os.path.exists(args["outdir"]):
        exit("[ERR] output directory already exists \n{}"
             .format(args["outdir"]))
    else:
        print("[INFO] creating output directories ...")
        os.makedirs(plotDir)
    frmColDF = calc_colour_shift(colTab, colDistDict, plotDir)
    annTabP = annTab.copy()
    annTab["boxSize"] = np.sqrt(np.abs(annTab.x2 - annTab.x1) *
                                np.abs(annTab.y2 - annTab.y1))
    annTab["indx"] = range(0, len(annTab))
    annTab["boxShift"] = 0.0
    if args["save_plots"]:
        fig = plt.figure(figsize=(8, 8))
    frmDict = {}
    for i, label in enumerate(labels):
        sizeMed = sizeDistDict["GLO"][0]
        sizeSigma = sizeDistDict["GLO"][1]
        sizeClips = sizeDistDict["GLO_clips"]
        if label in sizeDistDict:
            print("[INFO] found custom size target for '{}'".format(label))
            sizeMed = sizeDistDict[label][0]
            sizeSigma = sizeDistDict[label][1]
            sizeClips = sizeDistDict[label + "_clips"]
        sizeArr = annTab.loc[annTab["label"]==label]["boxSize"].values
        print("[INFO] computing size shift for class '{}'".format(label))
        dS, sCumArr, sSrtArr, sSigma, sMed =                            get_norm_shift(sizeArr,
                                           sigma=sizeSigma,
                                           med=sizeMed,
                                           clipLoFn="shift",
                                           clipHiFn="zero",
                                           absClips=sizeClips)
        annTab.loc[annTab["label"]==label, "boxShift"] = dS
        if args["save_plots"]:
            fig.clf()
            titleStr = "Size distribution for class '{}'.".format(label)
            plot_shift(fig, sizeArr, dS, sCumArr, sSrtArr, sSigma, sMed,
                       "Size (pixels)", titleStr)
            outPath = os.path.join(plotDir, "SizeShift_" + label +".png")
            fig.savefig(outPath)
    print("[INFO] calculating median position and size", flush=True)
    annTab["xBoxMed"] = (annTab.x1 + annTab.x2) / 2.0
    annTab["yBoxMed"] = (annTab.y1 + annTab.y2) / 2.0
    medByFrame = annTab.groupby(annTab.imageName).median()
    medByFrame["zoomFac"] = (medByFrame.boxSize /
                             (medByFrame.boxSize + medByFrame.boxShift))
    medByFrame["zoomFac"].clip(upper=1.0, inplace=True)
    medByFrame.drop(columns=["x1", "y1", "x2", "y2", "indx", "boxShift",
                             "nXpix", "nYpix"],
                    inplace=True)
    medByFrame.rename(columns={"boxSize": "boxSizeMed"}, inplace=True)
    nByFrame = annTab.groupby(annTab.imageName).count()
    fn = lambda obj: obj.loc[rd.choice(obj.index, 1), :]
    randByFrame = annTab.groupby(annTab.imageName, as_index=True).apply(fn)
    randByFrame.set_index(["imageName"], inplace=True)
    annTab.drop(columns=["xBoxMed", "yBoxMed"], inplace=True)
    mask = nByFrame.x1 <= 4
    medByFrame.loc[mask, ["xBoxMed", "yBoxMed"]] =                                randByFrame.loc[mask, ["xBoxMed", "yBoxMed"]]
    print("[INFO] calculating minimum position and size", flush=True)
    minByFrame = annTab.groupby(annTab.imageName).min()
    minByFrame.rename(columns={"x1": "xBoxMin",  "y1": "yBoxMin"},
                      inplace=True)
    minByFrame.drop(columns=["x2", "y2", "boxSize", "indx", "boxShift",
                             "label", "nXpix", "nYpix", "date", "location"],
                    inplace=True)
    print("[INFO] calculating maximum position and size", flush=True)
    maxByFrame = annTab.groupby(annTab.imageName).max()
    maxByFrame.rename(columns={"x2": "xBoxMax",  "y2": "yBoxMax"},
                      inplace=True)
    maxByFrame.drop(columns=["x1", "y1", "boxSize", "indx", "boxShift",
                             "label", "date", "location"], inplace=True)
    print("[INFO] merging to create master table", flush=True)
    frmSzDF = pd.merge(medByFrame, minByFrame, left_on="imageName",
                     right_on="imageName", how="left")
    frmSzDF = pd.merge(frmSzDF, maxByFrame, left_on="imageName",
                     right_on="imageName", how="left")
    frmSzDF.loc[frmSzDF.zoomFac >= 1.0]["zoomFac"] = 1.0
    frmSzDF.loc[frmSzDF.zoomFac <= 0.0]["zoomFac"] = 1.0
    frmSzDF["winYlen"] = frmSzDF.nYpix * frmSzDF.zoomFac
    frmSzDF["zoomLim"] = (args["imageHeight"] /
                          (args["resolveLim"] * frmSzDF.winYlen))
    frmSzDF.loc[frmSzDF.zoomFac <= frmSzDF.zoomLim]["zoomFac"] = frmSzDF.zoomLim
    print("[INFO] zoom factor limits: min = {:.1f} max = {:.1f} std = {:.1f}"
          .format(1/frmSzDF.zoomFac.max(), 1/frmSzDF.zoomFac.min(),
                  (1/frmSzDF.zoomFac).std()))
    frmSzDF["winXlen"] = frmSzDF.nXpix * frmSzDF.zoomFac
    frmSzDF["winYlen"] = frmSzDF.nYpix * frmSzDF.zoomFac
    frmSzDF.loc[:, "canRotate"] = frmSzDF.winXlen <= frmSzDF.nYpix
    frmSzDF.loc[:, "canRotate"] = frmSzDF.canRotate.sample(frac=0.5)
    frmSzDF.canRotate.fillna(value=False, inplace=True)
    winXlenTmp = frmSzDF["winXlen"].copy()
    mask = frmSzDF["canRotate"]
    frmSzDF.loc[mask, "winXlen"] = frmSzDF.loc[mask, "winYlen"]
    frmSzDF.loc[mask, "winYlen"] = winXlenTmp.loc[mask]
    frmSzDF.loc[:, "canFlipLR"] = True
    frmSzDF.loc[:, "canFlipLR"] = frmSzDF.canFlipLR.sample(frac=0.5)
    frmSzDF.canFlipLR.fillna(False, inplace=True)
    frmSzDF.loc[:, "canFlipUD"] = True
    frmSzDF.loc[:, "canFlipUD"] = frmSzDF.canFlipUD.sample(frac=0.5)
    frmSzDF.canFlipUD.fillna(False, inplace=True)
    frmSzDF["xCentMin"] = (frmSzDF.xBoxMed
                           - frmSzDF.winXlen / 2
                           + frmSzDF.boxSizeMed / 2)
    frmSzDF["xCentMax"] = (frmSzDF.xBoxMed
                           + frmSzDF.winXlen / 2
                           - frmSzDF.boxSizeMed / 2)
    frmSzDF["yCentMin"] = (frmSzDF.yBoxMed
                           - frmSzDF.winYlen / 2
                           + frmSzDF.boxSizeMed / 2)
    frmSzDF["yCentMax"] = (frmSzDF.yBoxMed
                           + frmSzDF.winYlen / 2
                           - frmSzDF.boxSizeMed / 2)
    frmSzDF["xCentMin"].clip(lower=frmSzDF.winXlen / 2, inplace=True)
    frmSzDF["xCentMax"].clip(upper=frmSzDF.nXpix - frmSzDF.winXlen / 2,
                             inplace=True)
    frmSzDF["yCentMin"].clip(lower=frmSzDF.winYlen / 2, inplace=True)
    frmSzDF["yCentMax"].clip(upper=frmSzDF.nYpix - frmSzDF.winYlen / 2,
                             inplace=True)
    frmSzDF["rX"] = rd.uniform(low=frmSzDF.xCentMin, high=frmSzDF.xCentMax)
    frmSzDF["rY"] = rd.uniform(low=frmSzDF.yCentMin, high=frmSzDF.yCentMax)
    frmSzDF["rX"] = frmSzDF["rX"].apply(np.round)
    frmSzDF["rY"] = frmSzDF["rY"].apply(np.round)
    frmSzDF["x1"] = frmSzDF.rX - frmSzDF.winXlen / 2
    frmSzDF["x2"] = frmSzDF.rX + frmSzDF.winXlen / 2
    frmSzDF["y1"] = frmSzDF.rY - frmSzDF.winYlen / 2
    frmSzDF["y2"] = frmSzDF.rY + frmSzDF.winYlen / 2
    frmSzDF["xLen"] = frmSzDF.x2 - frmSzDF.x1
    frmSzDF["yLen"] = frmSzDF.y2 - frmSzDF.y1
    x1Shift = (0 - frmSzDF.x1).clip(lower=0)
    frmSzDF.x1 += x1Shift
    frmSzDF.x2 += x1Shift
    x2Shift = (frmSzDF.nXpix - frmSzDF.x2).clip(upper=0)
    frmSzDF.x2 += x2Shift
    frmSzDF.x1 += x1Shift
    y1Shift = (0 - frmSzDF.y1).clip(lower=0)
    frmSzDF.y1 += y1Shift
    frmSzDF.y2 += y1Shift
    y2Shift = (frmSzDF.nYpix - frmSzDF.y2).clip(upper=0)
    frmSzDF.y2 += y2Shift
    frmSzDF.y1 += y1Shift
    if not args["debug"]:
        frmSzDF = frmSzDF.drop(columns=["xBoxMin", "xBoxMax", "yBoxMin",
                                        "yBoxMax", "xCentMin", "xCentMax",
                                        "yCentMin", "yCentMax", "xBoxMed",
                                        "yBoxMed",  "winXlen",
                                        "winYlen", "rX", "rY"])
    zoomFilePath = os.path.join(args["outdir"], "ZOOMS.csv")
    frmSzDF.to_csv(zoomFilePath)
    imageNames = list(annTabP["imageName"].unique())
    nImages = len(imageNames)
    pBarCount = 0
    widgets = ["Processing {:d} Images: ".format(nImages),
               progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    print("[NOTE] progress bar takes time to show and update ...\n")
    pbar = progressbar.ProgressBar(maxval=nImages, widgets=widgets,
                                   redirect_stdout=True)
    for imageName in imageNames:
        pBarCount += 1
        try:
            pbar.update(pBarCount)
        except:
            pass
        try:
            zm = frmSzDF.loc[imageName]
        except KeyError:
            if args["verbose"]:
                print("[ERR] cannot find zoom entry for {}".format(imageName))
            continue
        scaleFacRoot = args["imageHeight"] / zm.nYpix
        imageWidth = int(round(zm.nXpix * scaleFacRoot))
        inPath = os.path.join(args["dataset"], imageName)
        imgBGR = cv2.imread(inPath)
        if args["debug"]:
            cv2.rectangle(imgBGR, (int(zm["xBoxMin"]), int(zm["yBoxMin"])),
                          (int(zm["xBoxMax"]), int(zm["yBoxMax"])),
                          (255,255,0), 2)
            cv2.circle(imgBGR, (int(zm["xBoxMed"]), int(zm["yBoxMed"])),
                          (20),(255,0,255), 2)
            cv2.rectangle(imgBGR, (int(zm["xCentMin"]), int(zm["yCentMin"])),
                          (int(zm["xCentMax"]), int(zm["yCentMax"])),
                          (0, 255, 255), 2)
            cv2.circle(imgBGR, (int(zm["rX"]), int(zm["rY"])),
                          (20),(0,0,255), 2)
            cv2.rectangle(imgBGR, (int(zm["x1"]), int(zm["y1"])),
                          (int(zm["x2"]), int(zm["y2"])),
                          (0, 0, 255), 2)
        else:
            imgBGR = imgBGR[int(zm["y1"]): int(zm["y2"]),
                            int(zm["x1"]): int(zm["x2"]) , :]
        count = 0
        for index, row in annTabP[annTabP.imageName==imageName].iterrows():
            xCent = int(round((row["x1"] + row["x2"]) / 2))
            yCent = int(round((row["y1"] + row["y2"]) / 2))
            if args["debug"]:
                count = 1
                if (xCent >= zm["x2"] or xCent <= zm["x1"] or
                    yCent >= zm["y2"] or yCent <= zm["y1"]):
                    continue
                row["x1"] = int(min(max(row["x1"], zm["x1"]), zm["x2"]))
                row["x2"] = int(min(max(row["x2"], zm["x1"]), zm["x2"]))
                row["y1"] = int(min(max(row["y1"], zm["y1"]), zm["y2"]))
                row["y2"] = int(min(max(row["y2"], zm["y1"]), zm["y2"]))
                cv2.rectangle(imgBGR, (row["x1"], row["y1"]),
                              (row["x2"], row["y2"]),
                              (255,255,255), 2)
                cv2.circle(imgBGR, (xCent, yCent),
                           (20),(255,255,255), 2)
            else:
                xCent -= zm["x1"]
                yCent -= zm["y1"]
                x1  = int(round(row["x1"] - zm["x1"]))
                x2  = int(round(row["x2"] - zm["x1"]))
                y1  = int(round(row["y1"] - zm["y1"]))
                y2  = int(round(row["y2"] - zm["y1"]))
                if (xCent >= zm["xLen"] or xCent <= 0 or
                    yCent >= zm["yLen"] or yCent <= 0):
                    continue
                x1 = int(min(max(x1, 0), zm["xLen"]))
                x2 = int(min(max(x2, 0), zm["xLen"]))
                y1 = int(min(max(y1, 0), zm["yLen"]))
                y2 = int(min(max(y2, 0), zm["yLen"]))
                if zm.canRotate:
                    scaleFacY = args["imageHeight"] / zm["xLen"]
                else:
                    scaleFacY = args["imageHeight"] / zm["yLen"]
                x1 = int(round(x1 * scaleFacY))
                x2 = int(round(x2 * scaleFacY))
                y1 = int(round(y1 * scaleFacY))
                y2 = int(round(y2 * scaleFacY))
                if zm.canRotate:
                    x1old = copy.copy(x1)
                    x2old = copy.copy(x2)
                    x1 = imageWidth - y1
                    x2 = imageWidth - y2
                    y1 = x1old
                    y2 = x2old
                if zm.canFlipLR:
                    x1 = imageWidth - x1
                    x2 = imageWidth - x2
                if zm.canFlipUD:
                    y1 = args["imageHeight"] - y1
                    y2 = args["imageHeight"] - y2
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                outCSV = os.path.join(args["outdir"], "BOXES.csv")
                csvFH = open(outCSV, "a")
                if args["verbose"]:
                    print("[INFO] writing box entry in '{}' ..."
                          .format(outCSV))
                csvFH.write("%s, %d, %d, %d, %d, %s, %d, %d, %s, %s, %d\n" %                            (imageName, x1, y1, x2, y2, row["label"],
                             imageWidth, args["imageHeight"],
                             row["date"], row["location"], row["qual"]))
                csvFH.close()
                gc.collect()
                count += 1
        if count <= 0:
            if args["verbose"]:
                print("[ERR] skipping as no valid objects")
            continue
        if zm.canRotate:
            tmpWidth = args["imageHeight"]
            tmpHeight = imageWidth
        else:
            tmpWidth = imageWidth
            tmpHeight = args["imageHeight"]
        if args["debug"]:
            tmpWidth = imageWidth
            tmpHeight = args["imageHeight"]
        try:
            if args["verbose"]:
                print("[INFO] resizing frame to {:d} x {:d} pixels"
                      .format(imageWidth, args["imageHeight"]))
            imgBGR = cv2.resize(imgBGR, (tmpWidth, tmpHeight))
        except Exception:
            if args["verbose"]:
                print("[WARN] failed to resize frame, skipping ...")
            continue
        if zm.canRotate:
            imgBGR = cv2.rotate(imgBGR, cv2.ROTATE_90_CLOCKWISE)
        if zm.canFlipLR:
            imgBGR =  cv2.flip(imgBGR, 1)
        if zm.canFlipUD:
            imgBGR =  cv2.flip(imgBGR, 0)
        try:
            cl = frmColDF.loc[imageName]
        except KeyError:
            if args["verbose"]:
                print("[ERR] cannot find colour entry for {}"
                      .format(imageName))
            continue
        if args["verbose"]:
            print("[INFO] correcting colour of frame")
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
        imgHSV = imgHSV.astype("f8")
        imgHSV[:, :, 0] += cl["dH"] * 180 
        imgHSV[:, :, 0] %= 181            
        imgHSV[:, :, 1] += cl["dS"] * 255 
        imgHSV[:, :, 1] = np.clip(imgHSV[:, :, 1], 0.0, 255.0)
        imgHSV[:, :, 2] += cl["dV"] * 255 
        imgHSV[:, :, 2] = np.clip(imgHSV[:, :, 2], 0.0, 255.0)
        imgHSV = imgHSV.astype("uint8")
        imgBGR = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
        outPath = os.path.join(args["outdir"], imageName)
        if args["verbose"]:
            print("[INFO] processing {}".format(outPath))
            print("[INFO] zoom factor {:.1f}".format(1/zm["zoomFac"]))
        cv2.imwrite(outPath, imgBGR)
    pbar.finish()
def get_norm_shift(x, sigma=None, med=None, sigClip=3.0, clipLoFn="shift",
                   clipHiFn="zero", absClips=[None, None]):
    if sigma is None:
        sigma = MAD(x)
    if med is None:
        med = np.median(x)
    nData = len(x)
    clipPlus = med + sigma * sigClip
    clipMin = med - sigma * sigClip
    if absClips[0] is not None:
        if clipMin < absClips[0]:
            clipMin = absClips[0]
    if absClips[1] is not None:
        if clipPlus > absClips[1]:
            clipPlus = absClips[1]
    cumArr = np.array(range(nData))/float(nData) + 1/(2 * nData)
    indxSrt = x.argsort()
    xSrtArr = x[indxSrt]
    xSrtTransArr = med + sigma * np.sqrt(2) * erfinv( 2 * cumArr - 1)
    xSrtTransArr = np.where(xSrtTransArr == -inf, xSrtArr, xSrtTransArr)
    if clipLoFn == "shift":
        shiftMArr = rd.uniform(clipMin, med, xSrtTransArr.shape)
        xSrtTransArr = np.where(xSrtTransArr < clipMin, shiftMArr,
                                xSrtTransArr)
    else:
        xSrtTransArr = np.where(xSrtTransArr < clipMin, xSrtArr,
                                xSrtTransArr)
    if clipHiFn == "shift":
        shiftPArr = rd.uniform(med, clipPlus, xSrtTransArr.shape)
        xSrtTransArr = np.where(xSrtTransArr > clipPlus, shiftPArr,
                                xSrtTransArr)
    else:
        xSrtTransArr = np.where(xSrtTransArr > clipPlus, xSrtArr,
                                xSrtTransArr)
    xSrtDiffArr = xSrtTransArr - xSrtArr
    xDiffArr = xSrtDiffArr[indxSrt.argsort()]
    return xDiffArr, cumArr, xSrtArr, sigma, med
def norm_cdf(mean=0.0, std=1.0, N=50, xArr=None):
    if xArr is None:
        x = np.linspace(mean-4.0*std, mean+4.0*std, N)
    else:
        x = xArr
    y = norm.cdf(x, loc=mean, scale=std)
    return x, y
def MAD(a, c=0.6745, axis=None):
    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)
    return m
def plot_shift(fig, xArr, xDiffArr, cumArr, xSrtArr, sigma, xMed, xLabel,
               title, lim01=False):
    ax1 = fig.add_subplot(2,2,1)              
    ax1.xaxis.set_visible(False)
    ax2 = fig.add_subplot(2,2,2, sharex=ax1)  
    ax2.yaxis.tick_right()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_label_position("right")
    ax3 = fig.add_subplot(2,2,3, sharex=ax1)  
    ax4 = fig.add_subplot(2,2,4, sharex=ax1)  
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    H = 1.0/ np.sqrt(2.0 * np.pi * sigma**2.0)
    xNorm = np.linspace(xMed-3*sigma, xMed+3*sigma, 1000)
    yNorm = H * np.exp(-0.5 * ((xNorm-xMed)/sigma)**2.0)
    fwhm = sigma * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    nBins = 25
    n, b, p = ax1.hist(xArr, nBins, density=1,
                       facecolor="lightgrey", histtype='step', fill=True,
                       linewidth=2)
    ax1.plot(xNorm, yNorm, color='k', linestyle="--", linewidth=2)
    ax1.set_title(title)
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(r'Normalised Counts')
    ax2.step(xSrtArr, cumArr, where="mid")
    xCDF, yCDF = norm_cdf(mean=xMed, std=sigma, N=1000)
    ax2.plot(xCDF, yCDF, color='k', linewidth=2, linestyle="--", zorder=1)
    ax2.set_title(r'CDF of Data vs Normal')
    ax2.set_xlabel(xLabel)
    ax2.set_ylabel(r'Normalised Counts')
    yTransArr = xArr + xDiffArr
    n, b, p = ax3.hist(yTransArr, nBins, density=1,
                       facecolor="lightgrey", histtype='step', fill=True,
                       linewidth=2)
    ax3.plot(xNorm, yNorm, color='k', linestyle="--", linewidth=2)
    ax3.set_title(r'New distribution of Data')
    ax3.set_xlabel(xLabel)
    ax3.set_ylabel(r'Normalised Counts')
    ax4.axhline(0.0, color="r",  linestyle="--", zorder=0)
    ax4.scatter(xArr, xDiffArr, s=14, marker="x")
    ax4.set_title(r'Shift in Distribution')
    ax4.set_xlabel(xLabel)
    ax4.set_ylabel(r'Shift')
    if lim01:
        ax1.set_xlim(0,1)
def calc_colour_shift(colTab, distDict, plotDir=None):
    shiftTab = colTab.copy()
    shiftTab["dH"] = 0.0
    shiftTab["dS"] = 0.0
    shiftTab["dV"] = 0.0
    shiftTab.drop(columns=["x1", "y1", "x2", "y2", "xBoxMed",  "yBoxMed",
                           "R", "G", "B"], inplace=True)
    labels = list(shiftTab["label"].unique())
    if plotDir is not None:
        fig = plt.figure(figsize=(8, 8))
    for label in labels:
        print("[INFO] calculating color shift for '{}' class".format(label))
        hueMed, hueSigma = distDict.get("H", [None, None])
        satMed, satSigma = distDict.get("S", [None, None])
        valMed, valSigma = distDict.get("V", [None, None])
        hueClips = distDict.get("H_clips", [None, None])
        satClips = distDict.get("S_clips", [None, None])
        valClips = distDict.get("V_clips", [None, None])
        if any(key.startswith(label) for key in distDict):
            print("[INFO] found custom colour target for '{}'".format(label))
            hueMed, hueSigma = distDict.get(label + "_H", [None, None])
            satMed, satSigma = distDict.get(label + "_S", [None, None])
            valMed, valSigma = distDict.get(label + "_V", [None, None])
            hueClips = distDict.get(label + "_H_clips", [None, None])
            satClips = distDict.get(label + "_S_clips", [None, None])
            valClips = distDict.get(label + "_V_clips", [None, None])
        hueArr = shiftTab.loc[shiftTab["label"]==label]["H"].values
        dH, hueCumArr, hueSrtArr, hueSigma, hueMed =                                            get_norm_shift(hueArr,
                                                           sigma=hueSigma,
                                                           med=hueMed,
                                                           absClips=hueClips,
                                                           clipLoFn="zero",
                                                           clipHiFn="zero")
        shiftTab.loc[shiftTab["label"]==label, "dH"] = dH
        satArr = shiftTab.loc[shiftTab["label"]==label]["S"].values
        dS, satCumArr, satSrtArr, satSigma, satMed =                                            get_norm_shift(satArr,
                                                           sigma=satSigma,
                                                           med=satMed,
                                                           absClips=satClips,
                                                           clipLoFn="zero",
                                                           clipHiFn="zero")
        shiftTab.loc[shiftTab["label"]==label, "dS"] = dS
        valArr = shiftTab.loc[shiftTab["label"]==label]["V"].values
        dV, valCumArr, valSrtArr, valSigma, valMed =                                            get_norm_shift(valArr,
                                                           sigma=valSigma,
                                                           med=valMed,
                                                           absClips=valClips,
                                                           clipLoFn="zero",
                                                           clipHiFn="zero")
        shiftTab.loc[shiftTab["label"]==label, "dV"] = dV
        if plotDir is not None:
            fig.clf()
            titleStr = "Hue distribution for class '{}'.".format(label)
            plot_shift(fig, hueArr, dH, hueCumArr, hueSrtArr, hueSigma, hueMed,
                       "Colour Hue", titleStr, lim01=True)
            outPath = os.path.join(plotDir, "HueShift_" + label +".png")
            fig.savefig(outPath)
            print("[INFO] saved '{}'".format(outPath))
            fig.clf()
            titleStr = "Sat distribution for class '{}'.".format(label)
            plot_shift(fig, satArr, dS, satCumArr, satSrtArr, satSigma, satMed,
                       "Colour Saturation", titleStr, lim01=True)
            outPath = os.path.join(plotDir, "SatShift_" + label +".png")
            fig.savefig(outPath)
            print("[INFO] saved '{}'".format(outPath))
            fig.clf()
            titleStr = "Val distribution for class '{}'.".format(label)
            plot_shift(fig, valArr, dV, valCumArr, valSrtArr, valSigma, valMed,
                       "Colour Value", titleStr, lim01=True)
            outPath = os.path.join(plotDir, "ValShift_" + label +".png")
            fig.savefig(outPath)
            print("[INFO] saved '{}'".format(outPath))
    shiftTab.drop(columns=["H", "S", "V"], inplace=True)
    print("[INFO] calculating median colour shift for all images", flush=True)
    frmColDF = shiftTab.groupby(shiftTab.imageName).median()
    return frmColDF
if __name__ == "__main__":
    main()
