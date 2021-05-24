#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re
import shutil
import traceback
import time
import argparse
import subprocess
import json
import gc
import cv2
import random
import numpy as np
import math as m
import progressbar
from Imports.util_video import FileVideoGrab
from Imports.util_video import FrameStamps
from Imports.util_video import get_framerates
def main():
    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("indxFile", nargs="?", default="videoIndx.json",
                    help="name of the index file [videoIndx.json].")
    ap.add_argument('-o', '--outDir', default="../DATA_FRAMES/frames_OUT",
                    help="output directory [../DATA_FRAMES/frames_OUT]")
    ap.add_argument("-s", "--skipFrm", type=int, default=3,
                    help="cutout every [3]rd frame to avoid correlation.")
    ap.add_argument("-c", "--classSkip",  action='append', nargs='+',
                    default=[], help="custom list of skipFrm per class")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print messages rather than a progress bar")
    ap.add_argument("-r", "--randomise", action="store_true",
                    help="randomise the video processing order")
    args = vars(ap.parse_args())
    if os.path.exists(args["indxFile"]):
        with open(args["indxFile"], 'r') as FH:
            videoFileIndx = json.load(FH)
            videoFileLst = list(videoFileIndx.keys())
            fileParmTab = np.array(list(videoFileIndx.values()))
            fileIndex = fileParmTab[:,0].tolist()
    else:
        exit("[ERR] Index file '%s' does not exist!" % args["indxFile"])
    classSkip = {}
    for skipPair in args["classSkip"]:
        if len(skipPair) < 2:
            exit("[ERR] malformed custom skip argument: {}".format(skipPair))
        try:
            classSkip[skipPair[0]]= int(skipPair[1])
            print("[INFO] custom frame skip for class '{}' = {:d}"                  .format(skipPair[0], classSkip[skipPair[0]]))
            print("[WARN]: Custom values for frame skips will result " +
                  "in unboxed objects in frames with multiple classes. " +
                  "Do NOT use for classes with significant overlap!")
        except Exception:
            exit("[ERR] could not parse skip argument: {}".format(skipPair))
        if classSkip[skipPair[0]] < 1:
            exit("[ERR] custom skip cannot be < 1.")
    outCSV = os.path.join(args["outDir"], "BOXES.csv")
    if os.path.exists(outCSV):
        print("[WARN] deleting existing CSV file {} ...".format(outCSV))
        os.remove(outCSV)
    nFiles = len(videoFileLst)
    pBarCount = 0
    widgets = ["Processing {:d} Videos: ".format(nFiles),
               progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    print("[NOTE] progress bar takes time to show and update ...\n")
    pbar = progressbar.ProgressBar(maxval=nFiles, widgets=widgets,
                                   redirect_stdout=True)
    if args["randomise"]:
        tmp = list(zip(videoFileLst, fileIndex))
        random.shuffle(tmp)
        videoFileLst, fileIndex = zip(*tmp)
    failLst = []
    traceLst = []
    for i in range(nFiles):
        videoFile = videoFileLst[i]
        outPrefix = "File{:05d}".format(fileIndex[i])
        try:
            cutout_tracks(videoFile,
                          outPrefix,
                          args["outDir"],
                          args["skipFrm"],
                          classSkip,
                          args["verbose"])
        except Exception:
            failLst.append(videoFile)
            traceMsg = traceback.format_exc()
            traceLst.append(traceMsg)
            if args["verbose"]:
                print("[WARN] failed on videofile '{}'".format(videoFile))
                print(traceMsg)
        pBarCount += 1
        try:
            pbar.update(pBarCount)
        except:
            pass
    pbar.finish()
    if len(failLst) > 0:
        print("[WARN] failed on the following videofiles:")
        for failFile, failMsg in zip(failLst, traceLst):
            print("\n> {}".format(failFile))
            print(failMsg, "\n")
    print("[INFO] copying {} to {} ".format(args["indxFile"], args["outDir"]))
    shutil.copy(args["indxFile"], args["outDir"])
def cutout_tracks(videoFile, outPrefix=None, outDir="DATA", skipFrm=3,
                  classSkip={}, verbose=False):
    dateStr, location = parse_date_loc(videoFile)
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if outPrefix is None:
        outPrefix = videoFile
    print("\n[INFO] processing video file '{}'".format(videoFile))
    cap = cv2.VideoCapture(videoFile)
    try:
        Nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        FPS = float(cap.get(cv2.CAP_PROP_FPS))
        if Nfrm <= 0 or FPS <= 0 or m.isnan(FPS):
            runTime_s, FPS, Nfrm = get_framerates(videoFile)
        else:
            runTime_s = Nfrm / FPS
        fc = FrameStamps(Nfrm, runTime_s)
    except Exception:
        if verbose:
            print("[WARN] failed to query properties of video file '{}'"
                  .format(videoFile))
        cap.release()
        return
    nXpix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nYpix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print("[INFO] output dimensions (x, y) = ({:d}, {:d})"
              .format(nXpix, nYpix))
    cap.release()
    path, tmp = os.path.split(videoFile)
    trackFile, dummy = os.path.splitext(tmp)
    trackFile = os.path.join(path, trackFile + "_tracks.json")
    if not os.path.exists(trackFile):
        if verbose:
            print("[WARN] trackfile does not exist '{}'!".format(trackFile))
        return
    else:
        with open(trackFile, 'r') as FH:
            trackLst = json.load(FH)
            nTracks = len(trackLst)
        if verbose:
            print("[INFO] found {:d} tracks in '{}'".format(nTracks,
                                                            trackFile))
    for trk, tD in enumerate(trackLst):
        if verbose:
            print("[INFO] processing track  {:d}".format(trk + 1))
        labStr = tD["label"]
        curSkipFrm = skipFrm
        if labStr in classSkip:
            curSkipFrm = classSkip[labStr]
        if verbose:
            print("[INFO] will extract every {:d} frames for {} class."                  .format(curSkipFrm, labStr))
        qual = float(tD["qual"])
        if qual > 0 and qual < 1:
            qual *= 10
        qual = int(round(qual))
        if labStr == "NEG":
            qual = 10
        lastFrm = tD["frmLst"][-1]
        validFrmLst = list(range(0, lastFrm, curSkipFrm))
        fvs = FileVideoGrab(videoFile, validFrmLst[0], validFrmLst[-1])
        fvs.start()
        time.sleep(1.0)
        while fvs.running():
            frame, frm, time_s = fvs.read()
            if frm not in validFrmLst:
                continue
            if frm not in tD["frmLst"]:
                continue
            i = tD["frmLst"].index(frm)
            x     = tD["xLst"][i]
            y     = tD["yLst"][i]
            xHalfSide = tD["xHalfSideLst"][i]
            yHalfSide = tD["yHalfSideLst"][i]
            if m.isnan(x):
                continue
            x1 = int(round(x - xHalfSide))
            x2 = int(round(x + xHalfSide))
            y1 = int(round(y - yHalfSide))
            y2 = int(round(y + yHalfSide))
            if labStr == "NEG":
                x1 = 0
                x2 = nXpix
                y1 = 0
                y2 = nYpix
            outRoot = "{}_Frm{:06d}".format(outPrefix, frm)
            outImgName = outRoot + ".jpg"
            outImgPath = os.path.join(outDir, outImgName)
            if verbose:
                print("\n[INFO] processing '{}' Track {:d} [{}]"
                      .format(outImgName, trk + 1, labStr))
            if labStr != "NEG":
                if (x<=m.ceil(xHalfSide) or x>=m.floor(nXpix-xHalfSide) or
                    y<=m.ceil(yHalfSide) or y>=m.floor(nYpix-yHalfSide)):
                    if verbose:
                        print("[WARN] skipping due to edge overlap ...")
                    continue
                if ((2. * xHalfSide +1 >= nXpix) or
                    (2. * yHalfSide +1 >= nYpix)):
                    if verbose:
                        print("[WARN] skipping due to large aperture ...")
                    continue
            if os.path.exists(outImgPath):
                if verbose:
                    print("[WARN] pre-existing image on disk ...")
            else:
                if verbose:
                    print("[INFO] accessed frame {:d} at time {:.2f}s"
                          .format(frm, time_s))
                cv2.imwrite(outImgPath, frame)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            outCSV = os.path.join(outDir, "BOXES.csv")
            csvFH = open(outCSV, "a")
            if verbose:
                print("[INFO] writing box entry in '{}' ...".format(outCSV))
            csvFH.write("%s, %d, %d, %d, %d, %s, %d, %d, %s, %s, %d\n" %                        (outImgName, x1, y1, x2, y2, labStr,
                         nXpix, nYpix, dateStr, location, qual))
            csvFH.close()
    cap.release()
    gc.collect()
def parse_date_loc(videoPath):
    yearStr = "2000"
    monthStr = "01"
    dayStr = "01"
    location = "unknown"
    path, fileName = os.path.split(videoPath)
    fileRoot, _ = os.path.splitext(fileName)
    dateRe = re.compile(r".*(\d{4})\.(\d{2})\.(\d{2}).*")
    match = dateRe.match(fileRoot)
    if match:
        yearStr = match.group(1)
        monthStr = match.group(2)
        dayStr = match.group(3)
    dateRe = re.compile(r".*(\d{4})-(\d{2})-(\d{2}).*")
    match = dateRe.match(path)
    if match:
        yearStr = match.group(1)
        monthStr = match.group(2)
        dayStr = match.group(3)
    beaches = ["byron", "evans", "lennox", "lighthouse",
               "kingscliff", "ballina", "redhead"]
    locRe = re.compile(r".*(" + "|".join(beaches) + ").*")
    match = locRe.match(videoPath.lower())
    if match:
        location = match.group(1)
    dateStr = "-".join([yearStr, monthStr, dayStr])
    return dateStr, location
if __name__ == "__main__":
    main()
