#!/usr/bin/env python
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import os
import json
import itertools
import operator
class BoxTracker():
    def __init__(self, maxGone=20, iouThreshSelf=0.3):
        self.nextObjectID = 0              
        self.boxDict = OrderedDict()       
        self.scoreDict = OrderedDict()     
        self.centDict = OrderedDict()      
        self.labelDict = OrderedDict()     
        self.goneCntDict = OrderedDict()   
        self.maxGone = maxGone             
        self.iouThreshSelf = iouThreshSelf 
        self.tracks = OrderedDict()
    def  _calc_IOUs(self, boxes1, boxes2):
        x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAarea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBarea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAarea + np.transpose(boxBarea) - interArea)
        return iou
    def _register(self, box, score, label, centroid, frm):
        self.boxDict[self.nextObjectID] = box
        self.scoreDict[self.nextObjectID] = score
        self.labelDict[self.nextObjectID] = label
        self.centDict[self.nextObjectID] = centroid
        self.goneCntDict[self.nextObjectID] = 0
        track = TimeTrack()
        track.update(frm, centroid, box, score, label)
        self.tracks[self.nextObjectID] = track
        self.nextObjectID += 1
    def _deregister(self, objectID):
        del self.boxDict[objectID]
        del self.scoreDict[objectID]
        del self.labelDict[objectID]
        del self.centDict[objectID]
        del self.goneCntDict[objectID]
    def update(self, boxes, scores, labels, frm):
        boxes = boxes.copy()
        scores = scores.copy()
        labels = labels.copy()
        if len(scores) == 0:
            for objectID in list(self.goneCntDict.keys()):
                self.goneCntDict[objectID] += 1
                if self.goneCntDict[objectID] > self.maxGone:
                    self._deregister(objectID)
            return boxes.copy(), scores.copy(), labels.copy()
        dropLst = []  
        iouArr = self._calc_IOUs(boxes, boxes)
        triBool = ~np.tril(np.ones_like(iouArr)).astype(np.bool)
        rows, cols = np.nonzero(triBool *  iouArr > self.iouThreshSelf)
        for row, col in zip(rows, cols):
            if scores[row] >= scores[col]:
                dropLst.append(col)
            else:
                dropLst.append(row)
        boxes = np.delete(boxes, dropLst, axis=0)
        scores = np.delete(scores, dropLst)
        labels = np.delete(labels, dropLst)
        inputCentroids = np.zeros((len(scores), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        xHalfSide = (endX - startX) / 2
        yHalfSide = (endY - startY) / 2
        if len(self.scoreDict) == 0:
            for i in range(0, len(scores)):
                self._register(boxes[i, :], scores[i], labels[i],
                               inputCentroids[i], frm)
        else:
            objectIDs = list(self.centDict.keys())
            storedCentroids = np.array(list(self.centDict.values()))
            D = dist.cdist(np.array(storedCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.boxDict[objectID] = boxes[col, :]
                self.scoreDict[objectID] = scores[col]
                self.labelDict[objectID] = labels[col]
                self.centDict[objectID] = inputCentroids[col]
                self.goneCntDict[objectID] = 0
                self.tracks[objectID].update(frm,
                                             inputCentroids[col],
                                             boxes[col, :],
                                             scores[col],
                                             labels[col])
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.goneCntDict[objectID] += 1
                    if self.goneCntDict[objectID] > self.maxGone:
                        self._deregister(objectID)
            else:
                for col in unusedCols:
                    self._register(boxes[col, :], scores[col], labels[col],
                                   inputCentroids[col], frm)
        return boxes.copy(), scores.copy(), labels.copy()
    def save_json(self, outFile, labelDict=None, doAppend=True):
        if os.path.exists(outFile):
            if doAppend:
                with open(outFile, 'r') as FH:
                    trackLst = json.load(FH)
                    nTracks = len(trackLst)
                print("[INFO] appending to {:d} tracks in existing file"
                      .format(nTracks))
            else:
                print("[WARN] overwriting existing file")
                trackLst = []
        else:
            trackLst = []
        for objectID, track in self.tracks.items():
            trackLst.append(track.get_trackdict(labelDict))
        nTracks = len(trackLst)
        with open(outFile, 'w') as FH:
            json.dump(trackLst, FH)
        print("[INFO] wrote {:d} tracks to {}.".format(nTracks, outFile))
class TimeTrack():
    def __init__(self):
        self.frmLst = []
        self.xLst = []
        self.yLst = []
        self.xHalfSideLst = []
        self.yHalfSideLst = []
        self.label = ""
        self.labelLst = []
        self.qual = ""
        self.scoreLst = []
        self.comments = ""
        self.pickle = ""
    def get_trackdict(self, labelDict=None):
        labelNum = max(set(self.labelLst), key = self.labelLst.count)
        if labelDict is not None:
            if labelNum in labelDict:
                self.label = labelDict[labelNum]
            else:
                self.label = str(labelNum)
        self.qual = int(round(np.median(self.scoreLst)*10))
        trackDict = {"frmLst"       : self.frmLst,
                     "xLst"         : self.xLst,
                     "yLst"         : self.yLst,
                     "xHalfSideLst" : self.xHalfSideLst,
                     "yHalfSideLst" : self.yHalfSideLst,
                     "label"        : self.label,
                     "labelLst"     : self.labelLst,
                     "qual"         : self.qual,
                     "scoreLst"     : self.scoreLst,
                     "comments"     : self.comments,
                     "pickle"       : self.pickle}
        return trackDict
    def update(self, frm, centroid, box,  score, label):
        self.frmLst.append(int(frm))
        self.xLst.append(int(round(centroid[0])))
        self.yLst.append(int(round(centroid[1])))
        self.xHalfSideLst.append(float((box[2] - box[0]) / 2))
        self.yHalfSideLst.append(float((box[3] - box[1]) / 2))
        self.labelLst.append(int(label))
        self.scoreLst.append(float(score))
