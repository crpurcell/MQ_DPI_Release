#!/usr/bin/env python
import os
import shutil
import copy
import csv
import json
import math as m
import traceback
import cv2
import numpy as np
from .util_video import FrameStamps
from .util_video import FrameCache
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
class TrackManager:
    def __init__(self, videoFilePath, useCache=False, cacheFile="frame.cache",
                 debug=False):
        self.cap = None
        self.frameBGR = None
        self.frameBGRnoAnn = None
        self.Nfrm = None
        self.FPS = None
        self.runTime_s = None
        self.fc = None
        self.nXpix = None
        self.nYpix = None
        self.useCache = useCache
        self.cacheFile = cacheFile
        self.frameCache = None
        self.frameCacheIndx = None
        self.trackRecs = None
        self.trackMetas = None
        self.trackMetaMask = None
        self.trackSummary = []
        self.trackRecsUndo = None
        self.trackMetasUndo = None
        self.trackMetaMaskUndo = None
        self.currentTrackUndo = None
        self.editRec = None
        self.dType = [("frm",       "u8"),  
                      ("x",         "f8"),  
                      ("y",         "f8"),  
                      ("xHalfSide", "f8"),  
                      ("yHalfSide", "f8"),  
                      ("label",     "a5"),  
                      ("score",     "f8"),  
                      ("mask",      "i8")]  
        self.debug = debug
        self.currentFrame = None
        self.currentTrack = None
        self.doAnnTrackNum = True
        self.outDir, videoFile = os.path.split(videoFilePath)
        self.videoFileRoot, ext = os.path.splitext(videoFile)
        success = self._read_video(videoFilePath)
        if success:
            return
        self.trackFilePath = os.path.join(self.outDir, self.videoFileRoot
                                          + "_tracks.json")
        success = self._read_tracks(self.trackFilePath)
        if success:
            exit("[ERR] Failed to read tracks! Check the trackfile format:\n"
                 + "{}".format(self.trackFilePath))
        self.editRec = self._get_empty_trackrec()
    def _read_video(self, videoFilePath):
        if not os.path.exists(videoFilePath):
            print("[ERR] video file missing {}".format(videoFilePath))
            return 1
        try:
            self.cap = cv2.VideoCapture(videoFilePath)
            self.Nfrm = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.FPS = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.runTime_s = self.Nfrm / self.FPS
            self.nXpix = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.nYpix = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.useCache:
                self.frameCache = FrameCache(videoFilePath, self.cacheFile)
                self.frameCache.start()
            self.fc = FrameStamps(self.Nfrm, self.runTime_s)
            self.set_frame(0)
            print("[INFO] number of frames = {:d}".format(self.Nfrm))
            print("[INFO] frame rate = {:.1f} frames/s".format(self.FPS))
            print("[INFO] run time = {:d} min {:02.1f} s"
                  .format(int(self.runTime_s // 60), self.runTime_s % 60))
            return 0
        except Exception:
            print("[ERR] error reading video {}".format(videoFilePath))
            if self.debug:
                print(traceback.format_exc())
            return 1
    def _read_tracks(self, trackFilePath):
        outDir, trackFile = os.path.split(trackFilePath)
        if not os.path.exists(trackFilePath):
            print("[WARN] track file missing {}".format(trackFile))
            print("[WARN] creating blank track as a convenience")
            self.trackRecs = [self._get_empty_trackrec()]
            self.trackMetas = [self._get_empty_trackmeta()]
            self._create_track_summary()
            return 0
        try:
            with open(trackFilePath, 'r') as FH:
                trackLst = json.load(FH)
                nTracks = len(trackLst)
            print("[INFO] found {:d} tracks".format(nTracks))
            self.trackRecs, self.trackMetas = self._tracklist_to_recs(trackLst)
            self._create_track_summary()
            return 0
        except Exception:
            print("[ERR] error reading tracks {}".format(trackFile))
            if self.debug:
                print(traceback.format_exc())
            return 1
    def _create_track_summary(self):
        self.trackSummary = []
        for i, mD in enumerate(self.trackMetas):
            self.trackSummary.append( (i+1, mD["length"], mD["label"],
                                       mD["qual"]))
    def _tracklist_to_recs(self, trackLst):
        trackRecs = []
        trackMetas = []
        for trkIndx, tD in enumerate(trackLst):
            print("[INFO] loading track {:d}".format(trkIndx + 1))
            trackRec = self._get_empty_trackrec(self.Nfrm)
            for i in range(len(tD["frmLst"])):
                frm = tD["frmLst"][i]
                if not m.isnan(tD["xLst"][i]):
                    trackRec["x"][frm]         = tD["xLst"][i]
                    trackRec["y"][frm]         = tD["yLst"][i]
                    trackRec["xHalfSide"][frm] = tD["xHalfSideLst"][i]
                    trackRec["yHalfSide"][frm] = tD["yHalfSideLst"][i]
                    trackRec["label"][frm]     = tD["label"]
                    if "scoreLst" in tD:
                        trackRec["score"][frm]     = tD["scoreLst"][i]
                    else:
                        trackRec["score"][frm] = tD["qual"]
                    trackRec["mask"][frm]      = True
            mDict = {}
            mDict["label"] = tD["label"]
            mDict["qual"] = tD["qual"]
            mDict["comments"] = tD["comments"]
            mDict["pickle"] = tD["pickle"]
            mDict["length"] = np.sum(trackRec["mask"])
            mDict["use"] = True
            trackRecs.append(trackRec)
            trackMetas.append(mDict)
        return trackRecs, trackMetas
    def _tracks_to_json(self):
        trackLst = []
        for indx in range(len(self.trackRecs)):
            tR = self.trackRecs[indx]
            tM = self.trackMetas[indx]
            msk = np.nonzero(tR["mask"])
            trackDict = {"frmLst"       : tR["frm"][msk].tolist(),
                         "xLst"         : tR["x"][msk].tolist(),
                         "yLst"         : tR["y"][msk].tolist(),
                         "xHalfSideLst" : tR["xHalfSide"][msk].tolist(),
                         "yHalfSideLst" : tR["yHalfSide"][msk].tolist(),
                         "label"        : tM["label"],
                         "qual"         : tM["qual"],
                         "timeRng"      : [0.0, 0.0],
                         "comments"     : tM["comments"],
                         "pickle"       : tM["pickle"]}
            trackLst.append(trackDict)
        return trackLst
    def _get_empty_trackrec(self, nRows=None):
        if nRows is None:
            nRows = self.Nfrm
        trackRec = np.empty((nRows,), dtype=self.dType)
        trackRec[:] = np.nan
        trackRec["frm"] = range(nRows)
        trackRec["label"] = ""
        trackRec["mask"] = False
        return trackRec
    def _get_empty_trackmeta(self):
        mDict = {}
        mDict["label"] = "NULL"
        mDict["qual"] = 10
        mDict["comments"] = "Empty track"
        mDict["pickle"] = ""
        mDict["length"] = 0
        mDict["use"] = True
        return mDict
    def _calc_rect_x1y1(self, trackRec, frameNum):
        if not trackRec["mask"][frameNum]:
            return 0, 0, 0, 0
        x = trackRec["x"][frameNum]
        y = trackRec["y"][frameNum]
        xHalfSide = trackRec["xHalfSide"][frameNum]
        yHalfSide = trackRec["yHalfSide"][frameNum]
        x1 = round(int(x - xHalfSide))
        x2 = round(int(x + xHalfSide))
        y1 = round(int(y - yHalfSide))
        y2 = round(int(y + yHalfSide))
        return x1, y1, x2, y2
    def _annotate_frame(self, linewidth=2):
        if self.currentTrack is None:
            return
        for trackNum, trackRec in enumerate(self.trackRecs):
            x1, y1, x2, y2 = self._calc_rect_x1y1(trackRec, self.currentFrame)
            if trackNum == self.currentTrack:
                colour = (255, 0, 255)          
            else:
                colour = (0, 255, 255)          
            if not(x1 == x2 or y1 == y2):
                cv2.rectangle(self.frameBGR, (x1, y1), (x2, y2),
                              colour, linewidth)
                if self.doAnnTrackNum:
                    cv2.putText(self.frameBGR, str(trackNum + 1), (x2+5, y2+5),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.0, colour)
        x1, y1, x2, y2 = self._calc_rect_x1y1(self.editRec, self.currentFrame)
        if not(x1 == x2 or y1 == y2):
            cv2.rectangle(self.frameBGR, (x1, y1), (x2, y2),
                          (255, 0, 0), linewidth)
    def _save_undo(self):
        self.trackRecsUndo = copy.deepcopy(self.trackRecs)
        self.trackMetasUndo = copy.deepcopy(self.trackMetas)
        self.trackMetaMaskUndo = copy.deepcopy(self.trackMetaMask)
        self.currentTrackUndo = copy.deepcopy(self.currentTrack)
    def _nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    def _interp(self, y):
        y = y.copy()
        nans, x = self._nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y
    def _interpolate_rec(self, trackRec, frm1, frm2):
        frm1 = max(int(frm1), 0)
        frm2 = min(int(frm2), self.Nfrm-1)
        msk =  trackRec["mask"].copy()
        msk[:frm1] = 0
        msk[frm2+1:] = 0
        lim1 = np.min(np.nonzero(msk))
        lim2 = np.max(np.nonzero(msk))
        frm1 = max(lim1, frm1)
        frm2 = min(lim2, frm2)
        frm1 = max(int(frm1), 0)
        frm2 = min(int(frm2), self.Nfrm-1)
        trackRec = trackRec.copy()
        tRsec = trackRec[frm1:frm2+1]
        msk = tRsec["mask"] == False
        tRsec["x"][msk] = np.nan
        tRsec["y"][msk] = np.nan
        tRsec["xHalfSide"][msk] = np.nan
        tRsec["yHalfSide"][msk] = np.nan
        tRsec["x"] =  self._interp(tRsec["x"])
        tRsec["y"] =  self._interp(tRsec["y"])
        tRsec["xHalfSide"] =  self._interp(tRsec["xHalfSide"])
        tRsec["yHalfSide"] =  self._interp(tRsec["yHalfSide"])
        tRsec["mask"] = True
        trackRec[frm1:frm2+1] = tRsec
        return trackRec
    def _query_cache_state(self):
        nRead = 0
        if self.useCache:
            return self.frameCache.query_state()
    def set_frame(self, frameNum=None, preSeek=100):
        if frameNum is None:
            frameNum = self.currentFrame
        if preSeek > 0:
            setFrm = max(0, frameNum - preSeek)
        else:
            setFrm = 0
        if self.useCache:
            grabbed, frame = self.frameCache.read(frameNum)
            if grabbed:
                self.frameBGRnoAnn = frame.copy()
                self.frameBGR = frame.copy()
                self.currentFrame = frameNum
                self._annotate_frame()
                return 0
            else:
                return 1
        else:
            if self.currentFrame is not None:
                if frameNum <= self.currentFrame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, setFrm)
        success = 1
        while True:
            grabbed =  self.cap.grab()
            if not grabbed:
                return success
            frm =  int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) -1
            if frm == frameNum:
                _, frame =  self.cap.retrieve()
                self.frameBGR = frame.copy()
                self.currentFrame = frameNum
                self.frameBGRnoAnn = self.frameBGR.copy()
                self._annotate_frame()
                return
    def set_track(self, trackNum):
        self.currentTrack = trackNum
        self.frameBGR = self.frameBGRnoAnn.copy()
        self._annotate_frame()
    def get_track(self, trackNum=None):
        if trackNum is None:
            trackNum = self.currentTrack
        return self.trackRecs[trackNum].copy()
    def get_edit(self):
        return self.editRec.copy()
    def get_tracks_mask(self, recalculate=True):
        if recalculate:
            self.trackMetaMask = np.zeros((self.Nfrm,), dtype="i8")
            self.trackMetaMask[:] = False
            for trackRec in self.trackRecs:
                self.trackMetaMask = np.where(trackRec["mask"], True,
                                              self.trackMetaMask)
        return self.trackMetaMask
    def cull_tracks(self, cullLim=10):
        trackLst = []
        for i, mD in enumerate(self.trackMetas):
            if mD["length"] <= cullLim:
                trackLst.append(i)
        if len(trackLst) > 0:
            print("[INFO] culling tracks {}".format(trackLst))
            self.delete_tracks(trackNums=trackLst, saveUndo=True)
    def delete_tracks(self, trackNums=None, saveUndo=True):
        if saveUndo:
            self._save_undo()
        if trackNums is None:
            trackNums = [self.currentTrack]
        trackNums.sort()
        for indx in trackNums[::-1]:
            self.trackRecs.pop(indx)
            self.trackMetas.pop(indx)
        self._create_track_summary()
        self.set_track(max(0, trackNums[0] -1))
    def split_track(self, saveUndo=True):
        if saveUndo:
            self._save_undo()
        self.trackRecs.append(self._get_empty_trackrec())
        trOld = self.trackRecs[self.currentTrack]
        trNew = self.trackRecs[-1]
        k = self.currentFrame
        trNew["x"][k:] = trOld["x"][k:]
        trNew["y"][k:] = trOld["y"][k:]
        trNew["xHalfSide"][k:] = trOld["xHalfSide"][k:]
        trNew["yHalfSide"][k:] = trOld["yHalfSide"][k:]
        trNew["label"][k:] = trOld["label"][k:]
        trNew["score"][k:] = trOld["score"][k:]
        trNew["mask"][k:] = trOld["mask"][k:]
        trOld["mask"][k:] = False
        self.trackMetas.append(self.trackMetas[self.currentTrack].copy())
        mOld = self.trackMetas[self.currentTrack]
        mNew = self.trackMetas[-1]
        mOld["length"] = np.sum(trOld["mask"])
        mNew["length"] = np.sum(trNew["mask"])
        self._create_track_summary()
    def merge_tracks(self, trackNums=[], saveUndo=True):
        if len(trackNums) < 2:
            print("[WARN] select at least two tracks")
            return
        if saveUndo:
            self._save_undo()
        mergeLst = trackNums.copy()
        mergeLst.sort()
        trDst = self.trackRecs[mergeLst[0]]
        for indx in mergeLst[::-1][:-1]:
            trSrc = self.trackRecs[indx]
            msk = trSrc["mask"] == True
            trDst[msk] = trSrc[msk]
            self.trackRecs.pop(indx)
            self.trackMetas.pop(indx)
        mDst = self.trackMetas[mergeLst[0]]
        mDst["length"] = np.sum(trDst["mask"])
        self._create_track_summary()
        self.set_track(mergeLst[0])
    def undo(self):
        if not self.trackRecsUndo is None:
            trackRecsTmp = copy.deepcopy(self.trackRecs)
            trackMetasTmp = copy.deepcopy(self.trackMetas)
            trackMetaMaskTmp = copy.deepcopy(self.trackMetaMask)
            currentTrackTmp = copy.deepcopy(self.currentTrack)
            self.trackRecs = copy.deepcopy(self.trackRecsUndo)
            self.trackMetas = copy.deepcopy(self.trackMetasUndo)
            self.trackMetaMask = copy.deepcopy(self.trackMetaMaskUndo)
            self.currentTrack = copy.deepcopy(self.currentTrackUndo)
            self.trackRecsUndo = copy.deepcopy(trackRecsTmp)
            self.trackMetasUndo = copy.deepcopy(trackMetasTmp)
            self.trackMetaMaskUndo = copy.deepcopy(trackMetaMaskTmp)
            self.currentTrackUndo = copy.deepcopy(currentTrackTmp)
        self._create_track_summary()
    def relabel_track(self, label, trackNums=[], saveUndo=True):
        if saveUndo:
            self._save_undo()
        if len(trackNums) < 1:
            trackNums = [self.currentTrack]
        for trackNum in trackNums:
            self.trackRecs[trackNum]["label"] = label
            self.trackMetas[trackNum]["label"] = label
        self._create_track_summary()
    def setqual_track(self, qual, trackNums=[], saveUndo=True):
        qual = int(qual)
        qual = min([qual, 10])
        qual = max([qual, 0])
        if saveUndo:
            self._save_undo()
        if len(trackNums) < 1:
            trackNums = [self.currentTrack]
        for trackNum in trackNums:
            self.trackMetas[trackNum]["qual"] = qual
            self.trackMetas[trackNum]["qual"] = qual
        self._create_track_summary()
    def add_neg_track(self, frm1, frm2, saveUndo=True):
        if saveUndo:
            self._save_undo()
        frm1 = max(int(frm1), 0)
        frm2 = min(int(frm2), self.Nfrm-1)
        self.trackRecs.append(self._get_empty_trackrec())
        trNew = self.trackRecs[-1]
        trNew["x"][frm1:frm2+1] = self.nXpix // 2
        trNew["y"][frm1:frm2+1] = self.nYpix // 2
        trNew["xHalfSide"][frm1:frm2+1] = self.nXpix // 2
        trNew["yHalfSide"][frm1:frm2+1] = self.nYpix // 2
        trNew["label"][frm1:frm2+1] = "NEG"
        trNew["score"][frm1:frm2+1] = 1.0
        trNew["mask"][frm1:frm2+1] = True
        mDict = {}
        mDict["label"] = "NEG"
        mDict["qual"] = 10
        mDict["comments"] = "Added by GUI"
        mDict["pickle"] = ""
        mDict["length"] = np.sum(trNew["mask"])
        mDict["use"] = True
        self.trackMetas.append(mDict)
        trNums = list(range(len(self.trackRecs) -1))
        self.flag_framerange(frm1, frm2, trackNums=trNums, saveUndo=False)
        self._create_track_summary()
    def interpolate_track(self, frm1, frm2, saveUndo=True):
        if saveUndo:
            self._save_undo()
        trackRec = self.trackRecs[self.currentTrack]
        trackRec = self._interpolate_rec(trackRec, frm1, frm2)
        self.trackRecs[self.currentTrack] = trackRec
        mR = self.trackMetas[self.currentTrack]
        mR["length"] = np.sum(trackRec["mask"])
        self._create_track_summary()
    def flag_framerange(self, frm1, frm2, trackNums=[], flagAllTracks=None,
                        saveUndo=True):
        if saveUndo:
            self._save_undo()
        frm1 = max(int(frm1), 0)
        frm2 = min(int(frm2), self.Nfrm-1)
        if len(trackNums) < 0:
            trackNums = [self.currentTrack]
        if flagAllTracks:
            trackNums = list(range(len(self.trackRecs)))
        trackNums.sort()
        for indx in trackNums[::-1]:
            tR = self.trackRecs[indx]
            tR["mask"][frm1:frm2+1] = False
            mR = self.trackMetas[indx]
            mR["length"] = np.sum(tR["mask"])
        self._create_track_summary()
    def save_json(self):
        trackLst = self._tracks_to_json()
        trackFileBak = os.path.join(self.outDir, self.videoFileRoot
                                          + "_tracks.orig")
        if (not os.path.exists(trackFileBak)
            and os.path.exists(self.trackFilePath)):
            print("[INFO] backing up original tracks")
            shutil.copyfile(self.trackFilePath, trackFileBak)
        with open(self.trackFilePath, 'w') as FH:
            json.dump(trackLst, FH)
        print("[INFO] wrote tracks to {}.".format(self.trackFilePath))
    def put_box_edit(self, coords):
        x1, y1, x2, y2 = coords
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        xHalfSide = (x2 - x1) / 2
        yHalfSide = (y2 - y1) / 2
        k = self.currentFrame
        self.editRec["x"][k] = x
        self.editRec["y"][k] = y
        self.editRec["xHalfSide"][k] = xHalfSide
        self.editRec["yHalfSide"][k] = yHalfSide
        self.editRec["mask"][k] = True
    def clear_edits(self):
        self.editRec = self._get_empty_trackrec()
    def create_track_from_edit(self):
        eR = self.editRec.copy()
        if np.sum(eR["mask"]) == 0:
            print("[INFO] please draw at least one box")
            return
        frm1 = np.min(np.nonzero(eR["mask"]))
        frm2 = np.max(np.nonzero(eR["mask"]))
        eR = self._interpolate_rec(eR, frm1, frm2)
        eR["label"] = "EDIT"
        eR["score"] = 1.0
        self.trackRecs.append(eR)
        mDict = {}
        mDict["label"] = "EDIT"
        mDict["qual"] = 10
        mDict["comments"] = "Added by GUI"
        mDict["pickle"] = ""
        mDict["length"] = np.sum(eR["mask"])
        mDict["use"] = True
        self.trackMetas.append(mDict)
        self._create_track_summary()
    def cleanup(self):
        print("[INFO] cleaning up temporary files ... ", end="", flush=True)
        self.cap.release()
        if self.useCache:
            self.frameCache.stop()
            self.frameCache.cleanup()
        print("done", flush=True)
