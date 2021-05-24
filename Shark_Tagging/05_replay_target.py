#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import shutil
import time
import argparse
import subprocess
import json
import gc
import cv2
import numpy as np
import math as m
from Imports.util_video import FileVideoGrab
from Imports.util_video import FrameStamps
from Imports.util_video import get_framerates
def main():
    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("videoFile", metavar="videoFile.mp4", nargs=1,
                     help="Name of the video file.")
    ap.add_argument("-i", dest="frmInt", type=int, default=1,
                        help="Frame interval in miliseconds [1]")
    ap.add_argument('-f', dest='startFrame', type=int, default=0,
                        help="Starting frame [0]")
    ap.add_argument("-t", "--trackNumShow", action="store_true",
                        help="annotate using the track number [False]")
    ap.add_argument("-s", dest="scaleFac", type=float, default=3,
                        help="shrink native resolution factor [3].")
    ap.add_argument('-o', '--outVid', default=None,
                    help="output video file [None]")
    args = vars(ap.parse_args())
    annotate_target(args["videoFile"][0], args["frmInt"], args["startFrame"],
                    args["trackNumShow"], args["scaleFac"], args["outVid"])
def annotate_target(videoFile, frmInt=1, startFrm=0, trackNumShow=False,
                     scaleFac=3, outVid=None):
    pauseState = False
    cap = cv2.VideoCapture(videoFile)
    Nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS))
    if Nfrm <= 0 or FPS <= 0 or np.isnan(FPS):
        runTime_s, FPS, Nfrm = get_framerates(videoFile)
    else:
        runTime_s = Nfrm / FPS
    runTime_s = Nfrm / FPS
    fc = FrameStamps(Nfrm, runTime_s)
    print("\n[INFO] total # frames = %s" % Nfrm)
    print("[INFO] frame rate = %.3f frames/s" % FPS)
    print("[INFO] run time = %d min %02.1f s" % (int(runTime_s//60),
                                                 runTime_s%60))
    nXpix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nYpix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] video dimensions = %d x %d pix (x, y)." % (nXpix, nYpix))
    nXpixScreen = int(nXpix / scaleFac)
    nYpixScreen = int(nYpix / scaleFac)
    cap.release()
    path, tmp = os.path.split(videoFile)
    trackFile, dummy = os.path.splitext(tmp)
    trackFile = path + "/" + trackFile + "_tracks.json"
    if os.path.exists(trackFile):
        trackLOD = parse_tracks(trackFile)
    else:
        print("[ERR] Trackfile does not exist: \n'%s'\n" % trackFile)
        return
    cv2.namedWindow('Main Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Main Video', nXpixScreen, nYpixScreen)
    print("[INFO] starting video read thread...")
    fvs = FileVideoGrab(videoFile, startFrm)
    fvs.start()
    time.sleep(1.0)
    frame, frm, time_s = fvs.read()
    writer = None
    if outVid is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(outVid, fourcc, FPS,
                                 (nXpix, nYpix), True)
    progress(40, 0, False)
    while fvs.running():
        if trackLOD is not None:
            frame = annotate_old_tracks(frm, frame, trackLOD, trackNumShow)
        draw = frame.copy()
        ls = progress(40, (100.0 * (frm - startFrm + 1)/ (Nfrm-startFrm+1)))
        cv2.putText(frame, ls, (20, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.8,
                    (0,0,0))
        frmTime = frm/float(FPS)
        timeStr = "Time: %d:%02.1f Frm: %d / %d" %                  (int(frmTime//60), frmTime%60, frm, Nfrm)
        cv2.putText(frame, timeStr, (20, 170), cv2.FONT_HERSHEY_TRIPLEX, 2.0,
                    (0,0,0))
        intStr = "Int: %.1f ms" % frmInt
        cv2.putText(frame, intStr, (20, 240), cv2.FONT_HERSHEY_TRIPLEX, 2.0,
                    (0,0,0))
        cv2.imshow('Main Video', frame)
        k = cv2.waitKey(frmInt) & 0xFF
        if k == ord("q"):
            break
        if k == ord("p"):
            pauseState = not(pauseState)
        if k == ord("w"):     
            frmInt += 10
        if k == ord("e"):   
            frmInt = max(frmInt - 10, 1)
        if writer is not None:
            writer.write(draw)
        if not pauseState:
            frame, frm, time_s = fvs.read()
    cv2.destroyAllWindows()
    gc.collect()
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
    track1st = True
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
                    print(" >  Track {:d} starts at frame {:d}"
                          .format(trkIndx + 1, tD["frmLst"][i]))
                    track1st = False
        trackLOD.append(frmDict)
    return trackLOD
def annotate_old_tracks(frm, frame, trackLOD, trackNumShow=False):
    nXpix = 1
    nYpix = 1
    for trackNum, tD in enumerate(trackLOD):
        if frm in tD:
            x     = int(tD[frm][0] * nXpix)
            y     = int(tD[frm][1] * nXpix)
            xHalfSide = int(tD[frm][2] * nXpix)
            yHalfSide = int(tD[frm][3] * nXpix)
            if trackNumShow:
                label = str(trackNum + 1)
            else:
                label = tD[frm][4]
            x1 = x - xHalfSide
            x2 = x + xHalfSide
            y1 = y - yHalfSide
            y2 = y + yHalfSide
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            l = 50
            cv2.putText(frame, label, (x + xHalfSide, y - yHalfSide - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 0, 255), 1,
                        cv2.LINE_AA)
    return frame
if __name__ == "__main__":
    main()
