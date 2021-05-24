#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import copy
import shutil
import time
import subprocess
import argparse
import json
import gc
import cv2
import numpy as np
import math as m
from Imports.util_video import FileVideoGrab
from Imports.util_video import FrameStamps
from Imports.util_video import get_framerates
def main():
    parser = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("videoFile", metavar="videoFile.mp4", nargs=1,
                     help="Name of the video file.")
    parser.add_argument('-t', nargs=2, metavar="time", type=str,
                        dest='timeRange', default=["0:0", "0:0"],
                        help='Time range [m:s m:s] (minutes:seconds)')
    parser.add_argument("-s", dest="scaleFac", type=float, default=3,
                        help="shrink native resolution factor [3].")
    parser.add_argument("-z", dest="zoomStep", type=float, default=0.001,
                        help="Size of the zoom step [0.001].")
    parser.add_argument("-i", dest="frmInt", type=int, default=60,
                        help="Frame interval in miliseconds [40]")
    args = parser.parse_args()
    videoFile = args.videoFile[0]
    startTime_s, endTime_s = parse_timerange(args.timeRange)
    scaleFac = args.scaleFac
    zoomStep = args.zoomStep
    frmInt = args.frmInt
    follow_target(videoFile, startTime_s, endTime_s, scaleFac, zoomStep,
                  frmInt)
def follow_target(videoFile, startTime_s=None, endTime_s=None,
                  scaleFac=3, zoomStep=2, frmInt=30):
    root, ext = os.path.splitext(videoFile)
    cap = cv2.VideoCapture(videoFile)
    Nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS))
    if Nfrm <= 0 or FPS <= 0 or np.isnan(FPS):
        runTime_s, FPS, Nfrm = get_framerates(videoFile)
    else:
        runTime_s = Nfrm / FPS
    print("\n[INFO] total # frames = %s" % Nfrm)
    print("[INFO] frame rate = %.3f frames/s" % FPS)
    print("[INFO] run time = %d min %02.1f s" % (int(runTime_s//60),
                                                 runTime_s%60))
    nXpix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nYpix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspectRatio = float(nXpix) / nYpix
    print("[INFO] video dimensions = %d x %d pix (x, y)." % (nXpix, nYpix))
    nXpixScreen = int(nXpix / scaleFac)
    nYpixScreen = int(nYpix / scaleFac)
    cap.release()
    outFile = root + "_tracks.json"
    trackLOD = None
    if os.path.exists(outFile):
        trackLOD = parse_tracks(outFile)
    xHalfSide = 0.025
    yHalfSide = xHalfSide * aspectRatio
    cv2.namedWindow('Main Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Main Video', nXpixScreen, nYpixScreen)
    if startTime_s is None or endTime_s is None:
        startTime_s = 0.0
        endTime_s = runTime_s
    fc = FrameStamps(Nfrm, runTime_s)
    startFrm = fc.time2frm(startTime_s)
    endFrm =  fc.time2frm(endTime_s)
    print("[INFO] displaying time range %.1f - %.1f sec (frames %d - %d)." %
          (startTime_s, endTime_s, startFrm, endFrm))
    cs = MouseCoordStore(nXpix, nYpix)
    cv2.setMouseCallback('Main Video', cs.read_point)
    frmLst = []
    xLst = []
    yLst = []
    xHalfSideLst = []
    yHalfSideLst = []
    defaultDoStore = False
    print("[INFO] starting video read thread...")
    fvs = FileVideoGrab(videoFile, startFrm, endFrm)
    fvs.start()
    time.sleep(1.0)
    frame, frm, time_s = fvs.read()
    frameClean = copy.deepcopy(frame)
    print("\nZ = ZOOM, A = UNZOOM, X = TOGGLE CAPTURE")
    print("X / S = ADJUST ASPECT RATIO, Q = END TRACK\n")
    progress(40, 0)
    while fvs.running():
        ls = progress(40, (100.0 * (frm - startFrm + 1)/ (endFrm-startFrm+1)))
        cv2.putText(frame, ls, (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0,0,0))
        frmTime = frm/float(FPS)
        timeStr = "Time: %d:%02.1f Frm: %d / %d" %                  (int(frmTime//60), frmTime%60, frm, Nfrm)
        cv2.putText(frame, timeStr, (20, 170), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0,0,0))
        intStr = "Int: %.1f ms" % frmInt
        cv2.putText(frame, intStr, (20, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0,0,0))
        if trackLOD is not None:
            frame = annotate_old_tracks(frm, frame, trackLOD)
        if (cs.xyRel[0] > xHalfSide and
            cs.xyRel[0] < (1 - xHalfSide) and
            cs.xyRel[1] > yHalfSide and
            cs.xyRel[1] < (1 - yHalfSide)):
            doStore = defaultDoStore
        else:
            doStore = False
        xR, yR = cs.xyRel[0], cs.xyRel[1]
        xP, yP = cs.rel2wld(xR, yR, doInt=True)
        xHalfSideP = xHalfSide * nXpix
        yHalfSideP = yHalfSide * nYpix
        x1 = int(xP - xHalfSideP)
        x2 = int(xP + xHalfSideP)
        y1 = int(yP - yHalfSideP)
        y2 = int(yP + yHalfSideP)
        l = 20
        if doStore == True:
            boxColour = (255, 0 ,255)
        else:
            boxColour = (0, 0, 255)
            cv2.putText(frame, "OFF", (int(x2 + l), int(y2 + l)),
                        cv2.FONT_HERSHEY_TRIPLEX, 2.0, boxColour)
        cv2.rectangle(frame, (x1, y1), (x2, y2), boxColour, 2)
        cv2.line(frame, (xP-l//2, yP), (xP+l//2, yP), boxColour, 2)
        cv2.line(frame, (xP, yP-l//2), (xP, yP+l//2), boxColour,2)
        if not cs.pause:
            if doStore == True:
                frmLst.append(frm)
                xLst.append(xP)
                yLst.append(yP)
                xHalfSideLst.append(xHalfSideP)
                yHalfSideLst.append(yHalfSideP)
            else:
                frmLst.append(frm)
                xLst.append(np.nan)
                yLst.append(np.nan)
                xHalfSideLst.append(np.nan)
                yHalfSideLst.append(np.nan)
        cv2.imshow('Main Video', frame)
        k = cv2.waitKey(frmInt) & 0xFF
        if k == ord("z"):     
            if xHalfSide > 0.0001 and yHalfSide > 0.001:
                xHalfSide -= zoomStep
                yHalfSide -= zoomStep * aspectRatio
        if k == ord("a"):   
            xHalfSide += zoomStep
            yHalfSide += zoomStep * aspectRatio
        if k == ord("s"):     
            if xHalfSide > 0.0001:
                xHalfSide -= zoomStep
                yHalfSide += zoomStep * aspectRatio
        if k == ord("x"):   
            if yHalfSide > 0.0001:
                xHalfSide += zoomStep
                yHalfSide -= zoomStep * aspectRatio
        if k == ord("w"):     
            frmInt += 10
        if k == ord("e"):   
            frmInt = max(frmInt - 10, 1)
        if k == ord("c"):   
            defaultDoStore = not(defaultDoStore)
        if k == ord("q"):
            break
        if not cs.pause:
            frame, frm, time_s = fvs.read()
            frameClean = copy.deepcopy(frame)
        else:
            frame = copy.deepcopy(frameClean)
    cv2.destroyAllWindows()
    gc.collect()
    labelStr = input("\nEnter label [HAM]: ") or "HAM"
    qual = int(input("\nEnter quality [5]: ") or "5")
    commentStr = input("\nEnter comments []: ") or ""
    trackDict = {"frmLst"       : frmLst,
                 "xLst"         : xLst,
                 "yLst"         : yLst,
                 "xHalfSideLst" : xHalfSideLst,
                 "yHalfSideLst" : yHalfSideLst,
                 "label"        : labelStr,
                 "qual"         : qual,
                 "timeRng"      : [startTime_s, endTime_s],
                 "comments"     : commentStr,
                 "pickle"       : ""}
    outFile = root + "_tracks.json"
    if os.path.exists(outFile):
        with open(outFile, 'r') as FH:
            trackLst = json.load(FH)
            nTracks = len(trackLst)
        print("\n[INFO] found %d tracks in existing file." % nTracks)
        c = input("[O]verwrite or [A]ppend? [a]: ") or "a"
        if c == "o" or c == "O":
            trackLst = [trackDict]
            nTracks = 1
        else:
            trackLst.append(trackDict)
            nTracks += 1
    else:
        trackLst = [trackDict]
        nTracks = 1
    with open(outFile, 'w') as FH:
        json.dump(trackLst, FH)
    print("[INFO] wrote track to %s." % outFile)
    outIndx = "videoIndx.json"
    if os.path.exists(outIndx):
        with open(outIndx, 'r') as FH:
            indxDict = json.load(FH)
        fileParmTab = np.array(list(indxDict.values()))
        if videoFile in indxDict:
            fileIndex = indxDict[videoFile][0]
        else:
            fileIndex = int(np.max(fileParmTab[:,0])) + 1
    else:
        indxDict = {}
        fileIndex = 0
    indxDict[videoFile] = [fileIndex, nTracks]
    with open(outIndx, 'w') as FH:
        json.dump(indxDict, FH)
    print("[INFO] indexed video in %s." % outIndx)
    if False:
        for i in range(len(frmLst)):
            print("FRM %d: (%s, %s, %s, %s)" % (frmLst[i], xLst[i],
                                                yLst[i], xHalfSideLst[i],
                                                yHalfSideLst[i]))
def parse_timerange(timeRange):
    try:
        startTime_s = [float(x) for x in timeRange[0].split(":")]
        startTime_s = startTime_s[0]*60.0 + startTime_s[1]
        endTime_s = [float(x) for x in timeRange[1].split(":")]
        endTime_s = endTime_s[0]*60.0 + endTime_s[1]
    except Exception:
        startTime_s = None
        endTime_s = None
    if startTime_s==endTime_s:
        startTime_s = None
        endTime_s = None
    return startTime_s, endTime_s
class MouseCoordStore:
    def __init__(self, nXpix, nYpix):
        self.pause = True
        self.xyWorld = [0, 0]
        self.xyRel = [0, 0]
        self.nXpix = nXpix
        self.nYpix = nYpix
    def read_point(self, event, x, y, flags, param):
        self.xyWorld = [x, y]
        self.xyRel = [x / float(self.nXpix), y / float(self.nYpix)]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pause = not(self.pause)
    def wld2rel(self, x, y):
        return x / self.nXpix, y / self.nYpix
    def rel2wld(self, xR, yR, doInt=False):
        if doInt is True:
            x = int(round(xR * self.nXpix))
            y = int(round(yR * self.nYpix))
        else:
            x = xR * self.nXpix
            y = yR * self.nYpix
        return x, y
def progress(width, percent, printTerm=True):
    marks = m.floor(width * (percent / 100.0))
    spaces = m.floor(width - marks)
    loader = '[' + ('=' * int(marks)) + (' ' * int(spaces)) + ']'
    if printTerm:
        sys.stdout.write("  %s %d%%\r" % (loader, percent))
        if percent >= 100:
            sys.stdout.write("\n")
        sys.stdout.flush()
    loader = '[' + ('*' * int(marks)) + (' ' * int(spaces)) + ']'
    barStr = ("%s %d%%" % (loader, percent))
    return barStr
def parse_tracks(trackFile):
    with open(trackFile, 'r') as FH:
        trackLst = json.load(FH)
        nTracks = len(trackLst)
    print("[INFO] Found %d tracks in existing file" % nTracks)
    trackLOD = []
    for trkIndx, tD in enumerate(trackLst):
        track1st = True
        frmDict = {}
        for i in range(len(tD["frmLst"])):
            if not m.isnan(tD["xLst"][i]):
                frmDict[tD["frmLst"][i]] = [tD["xLst"][i],
                                            tD["yLst"][i],
                                            tD["xHalfSideLst"][i],
                                            tD["yHalfSideLst"][i],
                                            tD["label"],
                                            tD["pickle"]]
                if track1st:
                    print(" >  Track {:d} starts at frame {:d} (zero-indexed)"
                          .format(trkIndx + 1, tD["frmLst"][i]))
                    track1st = False
        trackLOD.append(frmDict)
    return trackLOD
def annotate_old_tracks(frm, frame, trackLOD):
    nXpix = 1
    nYpix = 1
    for tD in trackLOD:
        if frm in tD:
            x = int(tD[frm][0] * nXpix)
            y = int(tD[frm][1] * nYpix)
            xHalfSide = int(tD[frm][2] * nXpix)
            yHalfSide = int(tD[frm][3] * nYpix)
            x1 = x - xHalfSide
            x2 = x + xHalfSide
            y1 = y - yHalfSide
            y2 = y + yHalfSide
            cv2.rectangle(frame, (x1, y1), (x2, y2), (155, 50, 55), 2)
    return frame
if __name__ == "__main__":
    main()
