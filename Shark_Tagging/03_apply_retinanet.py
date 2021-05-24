#!/usr/bin/env python
from __future__ import print_function
import cProfile, pstats, io
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.colors import label_color
from keras.backend import tensorflow_backend
import progressbar
import json
import gc
import cv2
import os
import csv
import argparse
import numpy as np
import time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Imports.boxtracker import BoxTracker
from Imports.util_video import FileVideoGrab
from Imports.util_video import get_framerates
from Imports.util_video import FrameStamps
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to frozen model weights")
ap.add_argument("-l", "--labels", required=True,
                help="labels file in CSV format")
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-c", "--min-confidence", type=float, default=0.6,
                help="detection confidence threshold [0.6]")
ap.add_argument("-t", "--threshIOUself", type=float, default=0.3,
                help="IOU threshold for duplicates in a frame [0.3]")
ap.add_argument('-r', nargs=2, metavar="time", type=str,
                dest='timeRange', default=["0:0", "0:0"],
                help='Time range [m:s m:s] (minutes:seconds)')
ap.add_argument("-b", "--batch", action="store_true",
                help="batch mode: default to append to existing JSON [no]")
ap.add_argument("-o", "--outShow", action="store_true",
                help="show the output frame-by-frame [False]")
ap.add_argument("-s", dest="sideMinMax", nargs=2,  metavar="pix", type=int,
                default=[800, 4096],
                help="min and max frame size for network [800, 4096]")
ap.add_argument("-sc", dest="screenFac", type=float, default=3,
                help="shrink screen resolution by [3]")
args = vars(ap.parse_args())
def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner
def main():
    tensorflow_backend.set_session(get_session())
    print()
    if args["batch"]:
        print("[WARN] batch mode enabled: will append to existing tracks")
    with open(args["labels"], "r") as FH:
        reader = csv.reader(FH)
        labelDict = {int(row[1]): row[0] for row in reader}
    print("[INFO] loading model, this may take a few seconds ...")
    model = models.load_model(args["model"], backbone_name='resnet50')
    cap = cv2.VideoCapture(args["input"])
    Nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS))
    if Nfrm <= 0 or FPS <= 0 or np.isnan(FPS):
        runTime_s, FPS, Nfrm = get_framerates(args["input"])
    else:
        runTime_s = Nfrm / FPS
    runTime_s = Nfrm / FPS
    fc = FrameStamps(Nfrm, runTime_s)
    print("\n[INFO] total # frames = %s" % Nfrm)
    print("[INFO] frame rate = %.3f frames/s" % FPS)
    print("[INFO] run time = %d min %02.1f s" % (int(runTime_s//60),
                                                 runTime_s%60))
    sideMin, sideMax = sorted(args["sideMinMax"])
    nXpix = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    nYpix = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] video dimensions = %d x %d pix (x, y)." % (nXpix, nYpix))
    nXpixScreen = int(nXpix / args["screenFac"])
    nYpixScreen = int(nYpix / args["screenFac"])
    cap.release()
    print("[INFO] frames will be scaled within the following limits:")
    print("\tMIN-SIDE = {:d} pix".format(sideMin))
    print("\tMAX-SIDE = {:d} pix".format(sideMax))
    scale = compute_resize_scale([nYpix, nXpix, 3], sideMin, sideMax)
    print("[INFO] aspect-preserving scale factor is {:.1f}".format(scale))
    print("[INFO] injest frame size = {:d} x {:d} pix (x, y)."
          .format(int(nXpix * scale), int(nYpix *scale)))
    if args["outShow"]:
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', nXpixScreen, nYpixScreen)
    startTime_s, endTime_s = parse_timerange(args["timeRange"])
    if startTime_s is None or endTime_s is None:
        startTime_s = 0.0
        endTime_s = runTime_s
    fc = FrameStamps(Nfrm, runTime_s)
    startFrm = fc.time2frm(startTime_s)
    endFrm =  fc.time2frm(endTime_s)
    print("[INFO] displaying time range %.1f - %.1f sec (frames %d - %d)." %
          (startTime_s, endTime_s, startFrm, endFrm))
    tracker = BoxTracker(maxGone=25, iouThreshSelf=args["threshIOUself"])
    nFrmSel = endFrm - startFrm
    pBarCount = 0
    widgets = ["Processing Video: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=nFrmSel, widgets=widgets)
    print("[INFO] starting video read thread...")
    fvs = FileVideoGrab(args["input"], startFrm, endFrm, resizeScale=scale)
    fvs.start()
    time.sleep(1.0)
    print("[INFO] predicting labels for each frame ...")
    startProcTime = time.time()
    processTime = 0
    while fvs.running():
        frame, frmNum, time_s = fvs.read()
        draw = frame.copy()
        frame = preprocess_image(frame)
        boxes, scores, labels =                        model.predict_on_batch(np.expand_dims(frame, axis=0))
        boxes /= scale
        ind = scores > args["min_confidence"]
        boxes = boxes[ind, :]
        scores = scores[ind]
        labels = labels[ind]
        boxes, scores, labels = tracker.update(boxes, scores, labels, frmNum)
        fontPad = 3
        fontScale = 1.5
        fontThickness = 3
        boxThickness = 3
        fontColour = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for box, score, label in zip(boxes, scores, labels):
            if args["outShow"]:
                color = label_color(label)
                b = box * scale
                b = b.astype(int)
                caption = "{} {:1.1f}".format(labelDict[label], score)
                cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]),
                              color, boxThickness)
                txtWidth, txtHeight = cv2.getTextSize(caption, font, fontScale,
                                                      fontThickness)[0]
                labBoxCoords = ((b[0], b[1]), (b[0] + txtWidth + 2 * fontPad,
                                               b[1] - txtHeight - 2 * fontPad))
                cv2.rectangle(draw, labBoxCoords[0], labBoxCoords[1], color,
                              cv2.FILLED)
                cv2.putText(draw, caption, (b[0] + fontPad, b[1] - fontPad),
                            font,  fontScale, fontColour, fontThickness)
        if args["outShow"]:
            cv2.imshow("Video", draw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        pBarCount += 1
        try:
            pbar.update(pBarCount)
        except:
            pass
    pbar.finish()
    endProcTime = time.time()
    root, ext = os.path.splitext(args["input"])
    outFile = root + "_tracks.json"
    doAppend = True
    if os.path.exists(outFile) and not args["batch"]:
        print("\n[WARN] found existing track file")
        c = input("[O]verwrite or [A]ppend? [a]: ") or "a"
        if c == "o" or c == "O":
            doAppend = False
    tracker.save_json(outFile, labelDict, doAppend)
    print("[INFO] inference on all frames took {:.1f} seconds"
          .format(endProcTime - startProcTime))
    cv2.destroyAllWindows()
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
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
def compute_resize_scale(imgShape, sideMin=800, sideMax=1333):
    (rows, cols, _) = imgShape
    smallestSide, largestSide = sorted([rows, cols])
    scale = 1.0
    if largestSide >= sideMax:
        scale = sideMax / largestSide
    if smallestSide <= sideMin:
        scale = sideMin / smallestSide
    return scale
if __name__ == "__main__":
    main()
