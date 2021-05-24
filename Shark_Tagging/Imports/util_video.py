#!/usr/bin/env python
import os
import datetime
import subprocess
from threading import Thread
from queue import Queue
import cv2
import time
import numpy as np
class FrameStamps:
    def __init__(self, Nfrm, runTime_s):
        self.Nfrm = Nfrm
        self.runTime_s = runTime_s
    def frm2time(self, frm):
        time_s = max( min( float(frm) * self.runTime_s /
                           self.Nfrm, self.runTime_s), 0.0)
        return time_s
    def time2frm(self, time_s):
        frm = max(int(round( time_s * self.Nfrm / self.runTime_s )), 0)
        return frm
class FPS:
    def __init__(self):
        self._start = None   
        self._end = None     
        self._numFrames = 0  
    def start(self):
        self._start = datetime.datetime.now()
        return self
    def stop(self):
        self._end = datetime.datetime.now()
    def update(self):
        self._numFrames += 1
    def elapsed(self):
        return (self._end - self._start).total_seconds()
    def fps(self):
        return self._numFrames / self.elapsed()
class FileVideoGrab:
    def __init__(self, path, startFrame=None, endFrame=None, queueSize=128,
                 resizeScale=1.0):
        self.cap = cv2.VideoCapture(path)
        self.resizeScale = resizeScale
        self.Nfrm = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.runTime_s = self.Nfrm / self.FPS
        if startFrame is not None:
            self.startFrame = max(0, startFrame)
        else:
            self.startFrame = 0
        if endFrame is not None:
            self.endFrame = min(self.Nfrm, endFrame)
        else:
            self.endFrame = self.Nfrm
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
    def start(self):
        self.thread.start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                grabbed =  self.cap.grab()
                if not grabbed:
                    self.stop()
                    return
                frmNum =  int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                time_s = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if frmNum < self.startFrame:
                    continue
                if frmNum > self.endFrame:
                    self.stop()
                    return
                success, frame =  self.cap.retrieve()
                if self.resizeScale != 1.0:
                    frame = cv2.resize(frame, None,
                                       fx=self.resizeScale,
                                       fy=self.resizeScale)
                self.Q.put([frame, frmNum, time_s])
            else:
                time.sleep(0.1)
    def read(self):
        return self.Q.get()
    def more(self):
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.Q.qsize() > 0
    def running(self):
        return self.more() or not self.stopped
    def stop(self):
        self.stopped = True
def get_framerates(videoFile, verbose=False):
    cmd = "which exiftool ".format(videoFile)
    status, msg = subprocess.getstatusoutput(cmd)
    if status:
        exit("[ERR] The 'exiftool' command line utility is not installed.\n" +
             "      Please install using 'sudo apt install exiftool'.")
    else:
        if verbose:
            print("[INFO] using the exiftool command to query frame rates")
    cmd = "exiftool -n -Duration {}".format(videoFile)
    status, msg = subprocess.getstatusoutput(cmd)
    runTime_s = float(msg.split(":")[-1])
    cmd = "exiftool -VideoFrameRate {}".format(videoFile)
    status, msg = subprocess.getstatusoutput(cmd)
    FPS = float(msg.split(":")[-1])
    Nfrm = int(round(runTime_s * FPS))
    return runTime_s, FPS, Nfrm
class FrameCache:
    def __init__(self, videoPath, cacheFile="frame.cache"):
        self.cap = cv2.VideoCapture(videoPath)
        self.Nfrm = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.runTime_s = self.Nfrm / self.FPS
        self.nXpix = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nYpix = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cacheFile = cacheFile
        print("[INFO] using NUMPY MemMap file as a cache:")
        print("       > {}".format(self.cacheFile))
        self.frameCache = np.memmap(self.cacheFile, dtype='uint8',
                                    mode='w+',
                                    shape=(self.Nfrm, self.nYpix,
                                           self.nXpix, 3))
        self.frameCacheIndx = [0] * self.Nfrm
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
    def start(self):
        self.thread.start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            grabbed =  self.cap.grab()
            if not grabbed:
                self.stop()
                return
            frmNum =  int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            success, frame =  self.cap.retrieve()
            self.frameCache[frmNum, :, :, :] = frame.copy()
            self.frameCacheIndx[frmNum] = 1
    def read(self, frameNum):
        if frameNum >  self.Nfrm or frameNum < 0:
            print("[WARN] frame query outside of valid range")
            return False, None
        while True:
            if self.frameCacheIndx[frameNum] != 0:
                return True, self.frameCache[frameNum]
            time.sleep(0.1)
    def query_state(self):
        nFramesRead = sum(self.frameCacheIndx)
        cmd = "du {}".format(self.cacheFile)
        status, msg = subprocess.getstatusoutput(cmd)
        size_GB = float(msg.split("\t")[0]) / 1024**2
        return nFramesRead, size_GB
    def running(self):
        return not self.stopped
    def stop(self):
        self.stopped = True
        self.cap.release()
    def cleanup(self):
        os.remove(self.cacheFile)
