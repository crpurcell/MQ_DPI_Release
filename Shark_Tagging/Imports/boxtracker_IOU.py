#!/usr/bin/env python
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class BoxTracker():
    def __init__(self, maxGone=50, iouThresh=0.3, iouThreshSelf=None):
        self.nextObjectID = 0             
        self.boxDict = OrderedDict()      
        self.scoreDict = OrderedDict()    
        self.labelDict = OrderedDict()    
        self.goneCntDict = OrderedDict()  
        self.maxGone = maxGone            
        self.iouThresh = iouThresh        
        if iouThreshSelf == None:
            self.iouThreshSelf = iouThresh
        else:
            self.iouThreshSelf = iouThreshSelf
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
    def _register(self, box, score, label):
        self.boxDict[self.nextObjectID] = box
        self.scoreDict[self.nextObjectID] = score
        self.labelDict[self.nextObjectID] = label
        self.goneCntDict[self.nextObjectID] = 0
        self.nextObjectID += 1
    def _deregister(self, objectID):
        print("DE-Registering box")
        del self.boxDict[objectID]
        del self.labelDict[objectID]
        del self.scoreDict[objectID]
        del self.goneCntDict[objectID]
    def update(self, boxes, scores, labels):
        print()
        print("-" * 80)
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
        if len(self.scoreDict) == 0:
            for i in range(0, len(scores)):
                self._register(boxes[i, :], scores[i], labels[i])
        else:
            print("MATCHING EXISTING BOXES")
            objectIDs = list(self.boxDict.keys())
            storedBoxes = np.array(list(self.boxDict.values()))
            storedScores = np.array(list(self.scoreDict.values()))
            print("{:d} OLD BOXES, {:d} NEW BOXES"
                  .format(len(storedBoxes), len(boxes)))
            iouArr = self._calc_IOUs(boxes, storedBoxes)
            print(iouArr)
            unusedRows = list(range(iouArr.shape[1]))
            unusedCols = list(range(iouArr.shape[0]))
            rows, cols = np.nonzero(iouArr > self.iouThresh)
            print(rows, cols)
            print("MATCHED {:d} BOXES ACROSS FRAMES".format(len(rows)))
            for row in np.unique(rows):
                col = iouArr[row].argmax()
                if row not in unusedRows or col not in unusedCols:
                    continue
                objectID = objectIDs[row]
                self.boxDict[objectID] = boxes[col, :]
                self.scoreDict[objectID] = scores[col]
                self.labelDict[objectID] = labels[col]
                self.goneCntDict[objectID] = 0
                unusedRows.remove(row)
                unusedCols.remove(col)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.goneCntDict[objectID] += 1
                if self.goneCntDict[objectID] > self.maxGone:
                        self._deregister(objectID)
            for col in unusedCols:
                self._register(boxes[col, :], scores[col], labels[col])
        print()
        print("Next ObjectID:", self.nextObjectID)
        print("Tracking {:d} objects".format(len(self.scoreDict)))
        print("Tracking {:d} gones".format(len(self.goneCntDict)))
        print()
        return boxes.copy(), scores.copy(), labels.copy()
