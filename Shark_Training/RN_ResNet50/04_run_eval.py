#!/usr/bin/env python
import os
import copy
import json
import argparse
import keras
import tensorflow as tf
import csv
import numpy as np
import cv2
import time
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras.backend import tensorflow_backend
from keras_retinanet.utils.visualization import draw_detections
from keras_retinanet.utils.visualization import draw_annotations
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet import models
import progressbar
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to frozen model weights")
ap.add_argument("-l", "--labels", required=True,
                help="labels file in CSV format")
ap.add_argument("-a", "--annotations", required=True,
                help="path to the dataset annotation file.")
ap.add_argument("-o", "--outdir", required=True,
                help="path to the output directory")
ap.add_argument("-c", "--min-confidence", type=float, default=0.05,
                help="minimum detection confidence threshold [0.05]")
ap.add_argument("-cs", "--confidence-samples", type=float, nargs='+',
                default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                help="list of confidence limits to sample at [0.1, ..., 0.9]")
ap.add_argument("-D", "--debug", action="store_true",
                help="show debugging output (images and boxes)")
ap.add_argument("-ts", "--threshIOUself", type=float, default=0.3,
                help="eliminate duplicate boxes in each frame > IOU  [0.3]")
ap.add_argument("-tm", "--threshIOUmatch", type=float, default=0.3,
                help="match ground-truth and detected boxes > IOU  [0.3]")
ap.add_argument("-s", dest="sideMinMax", nargs=2,  metavar="pix", type=int,
                default=[1080, 1920],
                help="min and max frame size for network [1080, 4096]")
ap.add_argument("--config",
                help="Path to an anchor configuration file.")
ap.add_argument("--gpu",
                help="ID of the GPU to use (as reported by nvidia-smi).")
args = ap.parse_args()
def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sideMin, sideMax = sorted(args.sideMinMax)
    args.image_min_side = sideMin
    args.image_max_side = sideMax
    confThreshLst = args.confidence_samples
    print("[INFO] will sample at confidence limits {}".format(confThreshLst))
    tensorflow_backend.set_session(get_session())
    with open(args.labels, "r") as FH:
        reader = csv.reader(FH)
        labelDict = {int(row[1]): row[0] for row in reader}
    print("[INFO] loading model, this may take a few seconds ...")
    model = models.load_model(args.model, backbone_name='resnet50')
    generator = create_generator(args, preprocess_image)
    if os.path.exists(args.outdir):
        exit("[ERR] output directory already exists \n{}"
             .format(args.outdir))
    else:
        print("[INFO] creating output directory ...")
        os.makedirs(args.outdir)
    if args.debug:
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    detectionsRawDict = {}
    nImages = generator.size()
    print("[INFO] running inference on {:d} images".format(nImages))
    for i in progressbar.progressbar(range(nImages), prefix='Inference: '):
        imagePath = generator.image_names[i]
        imagePathRoot, imageName = os.path.split(imagePath)
        imageRaw = generator.load_image(i)
        imageNorm = generator.preprocess_image(imageRaw.copy())
        image, scale = generator.resize_image(imageNorm)
        nYpix, nXpix = imageRaw.shape[:2]
        boxes, scores, labels =                        model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        detectionsRawDict[imagePath] =                            [boxes, scores, labels, imageName, imagePathRoot]
        if args.debug:
            draw_annotations(imageRaw,
                             generator.load_annotations(i),
                             label_to_name=generator.label_to_name,
                             color=(255, 255, 255))
            cv2.imshow("Image", imageRaw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    annMasterDF = load_anntable_gen(generator)
    for minConfidence in confThreshLst:
        print("\n\n[INFO] analysing at confidence threshold {:.2f}"
              .format(minConfidence))
        detDF = process_detections(detectionsRawDict,
                                   minConfidence,
                                   args.threshIOUself,
                                   generator)
        annDF = annMasterDF.copy()
        annDF = annDF[annDF["imageName"].isin(detDF["imageName"])]
        confusionArr, annDF, detDF = match_detections(generator,
                                                      annDF,
                                                      detDF,
                                                      args.threshIOUmatch)
        TP = np.diagonal(confusionArr)[:-1]
        FP = confusionArr[:-1, :-1].sum(axis=1) - TP + confusionArr[-1, :-1]
        FN = confusionArr[:-1, -1]
        precisionArr = TP / (TP + FP)
        recallArr = TP / (TP + FN)
        classLabels = list(generator.classes.keys())
        for i, label in enumerate(classLabels):
            print("{}: TP={:d}, FP={:d}, FN={:d} P={:5.2f}, R={:5.2f}"
                  .format(label, TP[i], FP[i], FN[i], precisionArr[i],
                          recallArr[i]))
        confusionArr = confusionArr.astype("float")
        annTots = confusionArr[:-1, :].sum(axis=1)[:, np.newaxis]
        confusionArr[:-1, :] = confusionArr[:-1, :] / annTots
        confusionArr[-1, :-1] = confusionArr[-1, :-1] / annTots.T[0]
        confusionArr *= 100
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        sns.heatmap(confusionArr, annot=True,
                    xticklabels=classLabels + ["FN"],
                    yticklabels=classLabels + ["FP"], ax=ax,
                    fmt="02.0f",
                    cmap="plasma")
        plt.yticks(rotation=0, ha='right')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title("Minimum IoU = {:.2f}, Minimum Confidence = {:.2f}"
                     .format(args.threshIOUmatch, minConfidence))
        outName = args.outdir + "/ConfMatrix_IoU_{:03.2f}_Conf_{:03.2f}.svg"                      .format(args.threshIOUmatch, minConfidence)
        print("[INFO] saving confusion matrix plot to '{}'".format(outName))
        fig.savefig(outName)
        outName = args.outdir + "/AnnTable_IoU_{:03.2f}_Conf_{:03.2f}.pkl"                      .format(args.threshIOUmatch, minConfidence)
        print("[INFO] saving annotation table to '{}'".format(outName))
        annDF.to_pickle(outName)
        outName = args.outdir + "/DetTable_IoU_{:03.2f}_Conf_{:03.2f}.pkl"                      .format(args.threshIOUmatch, minConfidence)
        print("[INFO] saving detection table to '{}'".format(outName))
        detDF.to_pickle(outName)
    print("\n[INFO] analysing mAP at min confidence threshold {:.2f}"
          .format(args.min_confidence))
    detDF = process_detections(detectionsRawDict,
                               args.min_confidence,
                               args.threshIOUself,
                               generator)
    annDF = annMasterDF.copy()
    annDF = annDF[annDF["imageName"].isin(detDF["imageName"])]
    confusionArr, annDF, detDF = match_detections(generator,
                                                  annDF,
                                                  detDF,
                                                  args.threshIOUmatch)
    classLabels = list(generator.classes.keys())
    prDict = {}
    apDict = {}
    apLst = []
    numTruthLst = []
    for i, label in enumerate(classLabels):
        mpre, mrec, ap, numTruths = calculate_pr_ap(detDF, annDF, label)
        prDict[label] = [mpre, mrec, ap]
        apDict[label] = ap
        apLst.append(ap)
        numTruthLst.append(numTruths)
        print("{:5.0f} instances of class {} with AP = {:.4f}"
              .format(numTruths, label, ap))
    fig = plt_prcurves(prDict)
    outName = args.outdir + "/PR_curves.svg"
    print("[INFO] saving precision-recall plot to '{}'".format(outName))
    fig.savefig(outName)
    mAP1 = sum([a * b for a, b in zip(numTruthLst, apLst)]) / sum(numTruthLst)
    print("mAP weighted = {:.4f}".format(mAP1))
    mAP2 = sum(apLst) / sum(x > 0 for x in numTruthLst)
    print("mAP          = {:.4f}".format(mAP2))
    apDict["mAP_weighted"] = mAP1
    apDict["mAP"] = mAP2
    outName = args.outdir + "/AP_values.json"
    with open(outName, 'w') as FH:
        json.dump(apDict, FH)
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
def create_generator (args, preprocess_image):
    common_args = {
        "batch_size"       : 1,
        "config"           : args.config,
        "image_min_side"   : args.image_min_side,
        "image_max_side"   : args.image_max_side,
        "preprocess_image" : preprocess_image,
    }
    validation_generator = CSVGenerator(
        args.annotations,
        args.labels,
        **common_args
    )
    return validation_generator
def  calc_IOUs(boxes1, boxes2):
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
def match_detections(generator, annDF, detDF, threshIOUmatch):
    nClasses = generator.num_classes()
    confusionArr = np.zeros((nClasses +1, nClasses + 1), dtype=np.int64)
    imageLst = annDF.imageName.unique().tolist()
    nImages = len(imageLst)
    for i in progressbar.progressbar(range(nImages),
                                     prefix='Matching Boxes: '):
        imageName = imageLst[i]
        annTmp = annDF[annDF.imageName == imageName]
        detTmp = detDF.loc[detDF.imageName == imageName]
        annBoxes = annTmp[["x1Ann", "y1Ann",
                           "x2Ann", "y2Ann"]].to_numpy()
        detBoxes = detTmp[["x1Pred", "y1Pred",
                           "x2Pred", "y2Pred"]].to_numpy()
        iouArr = calc_IOUs(annBoxes, detBoxes)
        rows, cols = np.nonzero(iouArr > args.threshIOUmatch)
        for row in np.unique(rows):
            col = iouArr[row].argmax()
            iou = iouArr[row][col]
            annIdx = annTmp.index[row]
            detIdx = detTmp.index[col]
            annDF.loc[annIdx, "labelPred"] = detDF.loc[detIdx, "labelPred"]
            detDF.loc[detIdx, "labelAnn"] = annDF.loc[annIdx, "labelAnn"]
            annDF.loc[annIdx, "score"] = detDF.loc[detIdx, "score"]
            annDF.loc[annIdx, "IOU"] = iou
            detDF.loc[detIdx, "IOU"] = iou
            j = generator.name_to_label(detDF.loc[detIdx, "labelAnn"])
            k = generator.name_to_label(annDF.loc[annIdx, "labelPred"])
            confusionArr[j, k] += 1
    for labelStr, labelVal in generator.classes.items():
        annTmp = annDF.loc[(annDF.labelAnn == labelStr) &
                           (annDF.labelPred == "")]
        nFalseNeg = int(annTmp.imageName.count())
        confusionArr[labelVal, nClasses] = nFalseNeg
        detTmp = detDF.loc[(detDF.labelPred == labelStr) &
                           (detDF.labelAnn == "")]
        nFalsePos = int(detTmp.imageName.count())
        confusionArr[nClasses, labelVal] = nFalsePos
    return confusionArr, annDF, detDF
def process_detections(detectionsRawDict, minConfidence, threshIOUself,
                       generator):
    detectionsRawDict = copy.copy(detectionsRawDict)
    imagePathLst = list(detectionsRawDict.keys())
    nImages = len(imagePathLst)
    detectLOD = []
    print("[INFO] processing detections in {:d} images".format(nImages))
    for i in progressbar.progressbar(range(nImages),
                                     prefix='Weeding Detections: '):
        imagePath = imagePathLst[i]
        boxes, scores, labels, imageName, imagePathRoot =                                                detectionsRawDict[imagePath]
        ind = scores > minConfidence
        boxes = boxes[ind, :]
        scores = scores[ind]
        labels = labels[ind]
        dropLst = []
        iouArr = calc_IOUs(boxes, boxes)
        triBool = ~np.tril(np.ones_like(iouArr)).astype(np.bool)
        rows, cols = np.nonzero(triBool *  iouArr > args.threshIOUself)
        for row, col in zip(rows, cols):
            if scores[row] >= scores[col]:
                dropLst.append(col)
            else:
                dropLst.append(row)
        boxes = np.delete(boxes, dropLst, axis=0)
        scores = np.delete(scores, dropLst)
        labels = np.delete(labels, dropLst)
        for box, score, label in zip(boxes, scores, labels):
            detectDict = {"imageName": imageName,
                          "x1Pred":     box[0],
                          "y1Pred":     box[1],
                          "x2Pred":     box[2],
                          "y2Pred":     box[3],
                          "score":      score,
                          "labelPred":  generator.labels[label]}
            detectLOD.append(detectDict)
    detDF = pd.DataFrame(detectLOD)
    detDF["labelAnn"] = ""
    detDF["IOU"] = 0.0
    return detDF
def load_anntable_gen(generator):
    annLOD = []
    nImages = generator.size()
    for i in range(nImages):
        anns = generator.load_annotations(i)
        imagePathRoot, imageName = os.path.split(generator.image_names[i])
        for box, label in zip(anns["bboxes"], anns["labels"]):
            annDict = {"imageName": imageName,
                       "x1Ann":        box[0],
                       "y1Ann":        box[1],
                       "x2Ann":        box[2],
                       "y2Ann":        box[3],
                       "labelAnn":     generator.labels[label]}
            annLOD.append(annDict)
    annDF = pd.DataFrame(annLOD)
    annDF["labelPred"] = ""
    annDF["IOU"] = 0.0
    annDF["score"] = 0.0
    return annDF
def calculate_pr_ap(detDF, annDF, label=""):
    if len(label) > 0:
        annClassDF = annDF[annDF.labelAnn == label]             
        detClassDF = detDF[(detDF.labelAnn == label) |          
                           (detDF.labelPred == label)].copy()   
    else:
        annClassDF = annDF.copy()
        detClassDF = detDF.copy()
    numTruths = annClassDF["imageName"].count()                 
    detClassDF.sort_values(by=["score"], ascending=False, inplace=True)
    if detClassDF.empty:
        return None, None, 0.0, 0.0
    detClassDF["TP"] = 0
    detClassDF["FP"] = 0
    detClassDF.loc[(detClassDF.labelAnn == detClassDF.labelPred), "TP"] = 1
    detClassDF.loc[(detClassDF.labelAnn != detClassDF.labelPred), "FP"] = 1
    detClassDF["AccTP"] = detClassDF.TP.cumsum()
    detClassDF["AccFP"] = detClassDF.FP.cumsum()
    detClassDF["Precision"] = detClassDF.AccTP / (detClassDF.AccTP +
                                                  detClassDF.AccFP)
    detClassDF["Recall"] = detClassDF.AccTP / numTruths
    precision = detClassDF.Precision.to_numpy()
    recall = detClassDF.Recall.to_numpy()
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return mpre, mrec, ap, numTruths
def plt_prcurves(prDict):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for k, v in prDict.items():
        label = k
        mpre, mrec, ap = v
        try:
            sns.lineplot(x=mrec, y=mpre, label=label, ax=ax, palette="tab10",
                         dashes=True, estimator=None)
        except Exception:
            print("[INFO] not enough data for {} P-R curve".format(label))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    return fig
if __name__ == "__main__":
    main()
