#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import numpy as np
import random
import progressbar
import argparse
import json
import os
import sys
import glob
import csv
import re
import pickle
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outdir", required=True,
                help="path to the output directory")
ap.add_argument("-d", "--dataset", required=True,
                help="path to directory containing images")
ap.add_argument("-v", "--validfrac", type=float, default=0.02,
                help="fraction of data to use for validation [0.02]")
ap.add_argument("-l", "--labels", default=[], nargs='+',
                help="list of allowed class labels []")
ap.add_argument("-m", "--merge",  action='append', nargs='+', default=[],
                help="list of classes to merge []")
ap.add_argument("-c", "--clip", type=int, default=0,
                help="clip the total number of images in each class [no clip]")
ap.add_argument("-s", "--sample", type=float, default=0.0,
                help="oversample class if < frac of overlaps [no oversample]")
ap.add_argument("-r", "--dateRange",  action='append', nargs=2, default=[],
                help='filter for data between dates [no filter]')
ap.add_argument("-nD", "--notDate", action="store_true",
                help="treat date selection as exclusion mask")
ap.add_argument("-p", "--places", default=[], nargs='+', metavar="beachname",
                help="list of allowed places []")
ap.add_argument("-nP", "--notPlace", action="store_true",
                help="treat place selection as exclusion mask")
ap.add_argument("-D", "--debug", action="store_true",
                help="print debug information")
ap.add_argument("-b", "--batch", action="store_false",
                help="batch mode (do not pause)")
args = vars(ap.parse_args())
def parse_daterange(dateRange):
    startDateStr = dateRange[0]
    endDateStr = dateRange[1]
    dateRe = re.compile(r".*(\d{4})-(\d{2})-(\d{2}).*")
    if dateRe.match(startDateStr) and dateRe.match(endDateStr):
        return startDateStr, endDateStr
    else:
        return None, None
startDateLst = []
endDateLst = []
for e in args["dateRange"]:
    startDateStr, endDateStr = parse_daterange(e)
    startDateLst.append(startDateStr)
    endDateLst.append(endDateStr)
annFile = os.path.join(args['dataset'], "BOXES.csv")
print("[INFO] loading {}".format(annFile))
annTab = pd.read_csv(annFile, header=None, skipinitialspace=True,
                     names=["imageName","x1","y1","x2","y2","label",
                            "nXpix", "nYpix", "date", "location", "qual"])
annTab.drop(columns=["nXpix", "nYpix"], inplace=True)
s = annTab[annTab.label == "NEG"].duplicated(subset=["imageName"])
indices = s[s].index
annTab.drop(indices, inplace=True)
print("[INFO] dropped {:d} duplicate empty (NEG) images".format(len(indices)))
allowedLabels = args["labels"]
if len(allowedLabels) > 0:
    print("[INFO] restricting labels to {}".format(allowedLabels))
    annTab = annTab[annTab["label"].isin(allowedLabels)]
annTab["date"] = pd.to_datetime(annTab["date"])
maskLst = []
for startDateStr, endDateStr in zip(startDateLst, endDateLst):
    if startDateStr is not None and endDateStr is not None:
        print("[INFO] filtering between dates {} and {}".format(startDateStr,
                                                                endDateStr))
        mask = (annTab["date"] >= startDateStr) & (annTab["date"] <= endDateStr)
        maskLst.append(mask)
for mask in maskLst[1:]:
    maskLst[0] = maskLst[0] | mask
if len(maskLst) > 0:
    if args["notDate"]:
        print("[INFO] treating date mask as exclusion")
        maskLst[0] = ~maskLst[0]
    print("[INFO] applying date mask accepting {:d} / {:d} boxes"
          .format(maskLst[0].sum(), len(maskLst[0])))
    annTab = annTab[maskLst[0]]
if len(args["places"]) > 0:
    mask = annTab["location"].isin(args["places"])
    if args["notPlace"]:
        print("[INFO] treating place mask as exclusion")
        mask = ~mask
    print("[INFO] applying place mask accepting {:d} / {:d} boxes"
          .format(mask.sum(), len(mask)))
    annTab = annTab[mask]
if len(annTab) == 0:
    exit("[ERR] table has no entries - check selection criteria. Exiting ...")
else:
    print("[INFO] processing data from {}".format(annTab["location"].unique()))
annTab.drop(columns=["date", "location"], inplace=True)
uniqueLabels = annTab["label"].unique().tolist()
uniqueLabels.sort()
labLookup = dict(zip(uniqueLabels, uniqueLabels))
for mergeLst in args["merge"]:
    if len(mergeLst) < 2:
        exit("[ERR] list of keys to be merged is too short: "
             "{}".format(mergeLst))
    print("[INFO] merging labels ['{}']<-{}".format(mergeLst[0], mergeLst[1:]))
    for k in mergeLst[1:]:
        labLookup[k] = mergeLst[0]
for k, v in labLookup.items():
    annTab.loc[annTab.label == k, "label"] = v
uniqueLabels = annTab["label"].unique().tolist()
uniqueLabels.sort()
nClasses = len(uniqueLabels)
print("[INFO] labels in dataset: {}".format(uniqueLabels))
labGroups = annTab.groupby("label")
boxCntSer = labGroups.count()["imageName"]
nTargetGlobal = boxCntSer.max()
if args["clip"] > 0:
    nTargetGlobal = args["clip"]
print("[INFO] aiming for {:d} samples in each class".format(nTargetGlobal))
posCntLst = []
olapCntLst = []
for label in uniqueLabels:
    imgLst = annTab[annTab.label == label]["imageName"].unique().tolist()
    posCntLst.append(len(imgLst))
    tmpTab = annTab[annTab["imageName"].isin(imgLst)]
    imgLst = tmpTab[tmpTab.label != label]["imageName"].unique().tolist()
    olapCntLst.append(len(imgLst))
imgCntSer = pd.Series(posCntLst, uniqueLabels)
olapCntSer = pd.Series(olapCntLst, uniqueLabels)
olapFracSer = olapCntSer / imgCntSer
olapFracSer.sort_values(ascending=False, inplace=True)
overlapTab = np.zeros((nClasses, nClasses), dtype=np.int)
for i in range(nClasses):
    labX = uniqueLabels[i]
    imgXlst = set(labGroups["imageName"].get_group(labX))
    for j in range(i+1, nClasses):
        labY = uniqueLabels[j]
        imgYlst = set(labGroups["imageName"].get_group(labY))
        overlapTab[j, i] = len(imgXlst & imgYlst)
print("[INFO] image overlap matrix:\n")
for i in range(nClasses):
    print(uniqueLabels[i], end=" ")
    for j in range(0, i+1):
        print("{:6d}".format(overlapTab[i, j]), end=" ")
    print("       "*(nClasses - j -1), end="")
    print(" {:9d}".format(boxCntSer[uniqueLabels[i]]), end="")
    print(" {:9d}".format(imgCntSer[uniqueLabels[i]]))
print("       " + "    ".join(uniqueLabels), end="")
print("     #BOXES   #IMAGES")
print("\nPercentage of images with overlaps in class:")
print("  ", end=" ")
for i in range(nClasses):
    print("    {:3.0f}".format(olapFracSer[uniqueLabels[i]]*100), end="")
print("\n")
if args["batch"]:
    input("\nPress <Return> to continue ...")
splitSer = pd.Series(np.zeros(len(uniqueLabels), dtype="i8"), uniqueLabels,
                     name="splitCount")
splitSer.index.name = "label"
validGrpLst = []
testGrpLst = []
trainGrpLst = []
print("[INFO] splitting off validation set {:2.2f}%".format(
    args["validfrac"]*100))
for label, olapFracCnt in olapFracSer.items():
    nTarget = int(min(nTargetGlobal, boxCntSer[label]) * args["validfrac"]
                  - splitSer[label])
    if nTarget <=0:
        continue
    imgGroups = annTab[annTab["label"] == label].groupby("imageName")
    imgCntTab = imgGroups.count()["label"].to_frame("nCurrentObjects")
    imgCntTab.reset_index(inplace=True)
    imgCntTab = imgCntTab.sample(frac=1).reset_index(drop=True)
    cumSumArr = imgCntTab["nCurrentObjects"].cumsum().to_numpy()
    indx = np.argwhere(cumSumArr >= nTarget)
    imgSel = imgCntTab["imageName"][:indx[0, 0]+1]
    selAnnTab = annTab[annTab["imageName"].isin(imgSel)]
    indices = annTab[annTab["imageName"].isin(imgSel)].index
    annTab.drop(indices, inplace=True)
    selGrpLst = [df for _, df in selAnnTab.groupby("imageName")]
    random.shuffle(selGrpLst)
    validGrpLst += selGrpLst
    selTab = pd.concat(selGrpLst, ignore_index=True)
    labSelGroups = selTab.groupby("label")
    splitCntSer = labSelGroups.count()["imageName"]
    splitCntSer.name = "splitCount"
    splitSer = splitSer.add(splitCntSer, fill_value=0)
random.shuffle(validGrpLst)
print("[INFO] validation split (number of boxes):")
print(splitSer)
masterImgGroups = annTab.groupby("imageName")
masterImgCountTab= masterImgGroups.count()["label"].to_frame("nObjects")
masterImgCountTab.reset_index(inplace=True)
masterImgCountTab["splitCount"] = 0
splitSer[:] = 0
for label, olapFracCnt in olapFracSer.items():
    print("[INFO] processing the '{}' class with {:02.2f}% overlap".          format(label, olapFracCnt*100))
    nTarget = max(0, nTargetGlobal - splitSer[label])
    print("[INFO] > previously selected boxes {:d}".          format(int(splitSer[label])))
    print("[INFO] > target boxes to select is {:d}".format(int(nTarget)))
    if nTarget == 0:
        print("[INFO] > target number of boxes already reached")
        continue
    imgPosGroups = annTab[annTab["label"] == label].groupby("imageName")
    imgPosCntTab = imgPosGroups.count()["label"].to_frame("nCurrentObjects")
    imgPosCntTab.reset_index(inplace=True)
    imgLst = imgPosGroups.groups.keys()
    annLocTab = annTab[annTab["imageName"].isin(imgLst)]
    imgNegGroups = annLocTab[annLocTab["label"] != label].groupby("imageName")
    imgNegCntTab = imgNegGroups.count()["label"].to_frame("nOtherObjects")
    imgNegCntTab.reset_index(inplace=True)
    imgCntTab = imgPosCntTab.merge(imgNegCntTab, how="left",
                                   left_on="imageName", right_on="imageName")
    imgCntTab.fillna(0, inplace=True)
    if False:
        print("Percentage free = {:02.2f} ({:d})".format(
            np.sum(imgCntTab["nOtherObjects"] == 0) *100 / len(imgCntTab),
            len(imgCntTab)))
        print(imgCntTab.head())
    imgCntTab = imgCntTab.merge(masterImgCountTab, how="left",
                                left_on="imageName", right_on="imageName")
    imgCntTab = imgCntTab[imgCntTab["splitCount"] == 0]
    if len(imgCntTab) == 0:
        print("[INFO] > all images in this class have already been split")
        continue
    imgCntTab = imgCntTab.sample(frac=1).reset_index(drop=True)
    cumSumArr = imgCntTab["nCurrentObjects"].cumsum().to_numpy()
    indx = np.argwhere(cumSumArr >= nTarget)
    if len(indx) == 0:
        imgSel = imgCntTab["imageName"]
    else:
        imgSel = imgCntTab["imageName"][:indx[0, 0]+1]
    selAnnTab = annLocTab[annLocTab["imageName"].isin(imgSel)]
    selGrpLst = [df for _, df in selAnnTab.groupby("imageName")]
    random.shuffle(selGrpLst)
    tmp = pd.concat(selGrpLst).reset_index(drop=True)
    curBoxCnt = len(tmp[tmp.label == label])
    print("[INFO] > currently available boxes: {:d}".format(curBoxCnt))
    if args["sample"] > 0 and args["sample"] < 1 and curBoxCnt < nTarget:
        if olapFracCnt <= args["sample"]:
            imgCntTab = imgCntTab[imgCntTab["nOtherObjects"] == 0]
            cumSumArr = imgCntTab["nCurrentObjects"].cumsum().to_numpy()
            print("[INFO] > replicating using {:d} images ".                  format(len(imgCntTab)), end="")
            print("containing {:d} boxes".format(cumSumArr[-1]))
            n = int((nTarget - curBoxCnt) / cumSumArr[-1])
            r = nTarget - cumSumArr[-1] * (1 + n)
            if r <= 0:
                r = 0
            indx = np.argwhere(cumSumArr >= r)
            if len(indx) == 0:
                indx = 0
            else:
                indx = indx[0, 0]
            repLst = []
            if n > 0:
                imgSel = imgCntTab["imageName"]
                selAnnTab = annLocTab[annLocTab["imageName"].isin(imgSel)]                            .reset_index(drop=True)
                tmpGrpLst = [df for _, df in selAnnTab.groupby("imageName")]
                print("[INFO] > replicating whole table x {:d}".format(n))
                repLst = tmpGrpLst * n
                random.shuffle(repLst)
            if indx > 0:
                imgSel = imgCntTab["imageName"].iloc[:indx+1]
                selAnnTab = annLocTab[annLocTab["imageName"].isin(imgSel)]                            .reset_index(drop=True)
                tmpGrpLst = [df for _, df in selAnnTab.groupby("imageName")]
                print("[INFO] > replicating to index {:d}".format(indx))
                repLst += tmpGrpLst
                random.shuffle(repLst)
            if len(repLst)>1:
                selGrpLst += repLst
    trainGrpLst += selGrpLst
    selTab = pd.concat(selGrpLst, ignore_index=True)
    labSelGroups = selTab.groupby("label")
    splitCntSer = labSelGroups.count()["imageName"]
    splitCntSer.name = "splitCount"
    splitSer = splitSer.add(splitCntSer, fill_value=0)
    if args["debug"]:
        print(splitSer)
random.shuffle(validGrpLst)
random.shuffle(trainGrpLst)
print("[INFO] merging master tables (make take a little while) ... ")
trainTab = pd.concat(trainGrpLst).reset_index(drop=True)
if args["validfrac"] > 0:
    validTab = pd.concat(validGrpLst).reset_index(drop=True)
trainCntSer = trainTab.groupby("label").count()["imageName"]
print("\n", "-"*80)
print("[INFO] box count for training data:")
print(trainCntSer)
if args["validfrac"] > 0:
    validCntSer = validTab.groupby("label").count()["imageName"]
    print("[INFO] box count for validation data:")
    print(validCntSer)
outDirs = [os.path.join(args["outdir"], "models"),
           os.path.join(args["outdir"], "inference"),
           os.path.join(args["outdir"], "tflite")]
for outDir in outDirs:
    if not os.path.exists(outDir):
        os.makedirs(outDir)
absDataPath = os.path.abspath(args["dataset"])
trainTab["imageName"] = trainTab["imageName"].apply(lambda x:
                                            os.path.join(absDataPath, x))
if args["validfrac"] > 0:
    validTab["imageName"] = validTab["imageName"].apply(lambda x:
                                                os.path.join(absDataPath, x))
outAnnCSV = os.path.join(args["outdir"], "ann_train.csv")
print("[INFO] creating training annotation CSV in the output directory:")
print("\t[{}]".format(outAnnCSV))
trainTab.to_csv(outAnnCSV, header=False, index=False)
outAnnPickle = os.path.join(args["outdir"], "ann_train.pickle")
print("[INFO] creating training annotation PICKLE in the output directory:")
print("\t[{}]".format(outAnnPickle))
with open(outAnnPickle, "wb") as fh:
    pickle.dump(trainGrpLst, fh)
if args["validfrac"] > 0:
    outAnnCSV = os.path.join(args["outdir"], "ann_valid.csv")
    print("[INFO] creating a validation annotation" +
          "file in the output directory:")
    print("\t[{}]".format(outAnnCSV))
    validTab.to_csv(outAnnCSV, header=False, index=False)
    outAnnPickle = os.path.join(args["outdir"], "ann_valid.pickle")
    print("[INFO] creating validation annotation" +
          "PICKLE in the output directory:")
    print("\t[{}]".format(outAnnPickle))
    with open(outAnnPickle, "wb") as fh:
        pickle.dump(validGrpLst, fh)
outLst = zip(uniqueLabels, range(len(uniqueLabels)))
outClassCSV = os.path.join(args["outdir"], "classes.csv")
print("[INFO] creating class CSV file in the output directory:")
print("\t[{}]".format(outClassCSV))
with open(outClassCSV, "w") as FH:
    writer = csv.writer(FH)
    for row in outLst:
        writer.writerow(row)
outLabelMap = os.path.join(args["outdir"], "tflite", "labelmap.txt")
print("[INFO] creating labelmap.txt file in the output directory:")
print("\t[{}]".format(outLabelMap))
with open(outLabelMap, "w") as FH:
    FH.write("???\n")
    for label in uniqueLabels:
        FH.write("{}\n".format(label))
absImgPath = os.path.abspath(args["dataset"])
outJSON = os.path.join(args["outdir"], "imgpath.json")
with open(outJSON, "w") as fh:
    json.dump({"imgpath": absImgPath}, fh)
