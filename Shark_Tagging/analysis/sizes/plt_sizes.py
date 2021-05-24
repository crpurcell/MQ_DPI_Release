#!/usr/bin/env python
from __future__ import print_function
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import numpy.ma as ma
import colorsys
import math as m
import numpy as np
import json
import os
import sys
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-l", "--labels", default=[], nargs='+',
                help="list of allowed class labels []")
ap.add_argument("-m", "--merge",  action='append', nargs='+', default=[],
                help="list of classes to merge []")
ap.add_argument("-s", "--show", action="store_true",
                help="show the figures (default = save) [N]")
ap.add_argument("--log", action="store_true",
                help="log scale on the y-axis")
args = vars(ap.parse_args())
def main():
    annFile = os.path.join(args['dataset'], "BOXES.csv")
    print("[INFO] reading {}".format(annFile))
    annTab = pd.read_csv(annFile, header=None, skipinitialspace=True,
                         names=["imageName","x1","y1","x2","y2","label",
                                "nXpix", "nYpix", "date", "location", "qual"])
    allowedLabels = args["labels"]
    if len(allowedLabels) > 0:
        print("[INFO] restricting labels to {}".format(allowedLabels))
        annTab = annTab[annTab["label"].isin(allowedLabels)]
    annTab = annTab[annTab.label != "NEG"]
    annTab["boxSize"] = np.sqrt(np.abs(annTab.x2 - annTab.x1) *
                                np.abs(annTab.y2 - annTab.y1))
    uniqueLabels = annTab["label"].unique().tolist()
    uniqueLabels.sort()
    labLookup = dict(zip(uniqueLabels, uniqueLabels))
    for mergeLst in args["merge"]:
        if len(mergeLst) < 2:
            exit("[ERR] list of keys to be merged is too short: "
             "{}".format(mergeLst))
        print("[INFO] merging labels ['{}']<-{}".format(mergeLst[0],
                                                        mergeLst[1:]))
        for k in mergeLst[1:]:
            labLookup[k] = mergeLst[0]
    for k, v in labLookup.items():
        annTab.loc[annTab.label == k, "label"] = v
    uniqueLabels = annTab["label"].unique().tolist()
    uniqueLabels.sort()
    nClasses = len(uniqueLabels)
    print("[INFO] labels in dataset: {}".format(uniqueLabels))
    outDir = os.path.join(args["dataset"],"PLOTS_SIZE")
    if not args["show"]:
        print("[INFO] will save figures to {}".format(outDir))
        if not os.path.exists(outDir):
            os.makedirs(outDir)
    fig1 = plt.figure(figsize=(12, 7))
    plot_class_grid(fig1, annTab, log=args["log"])
    fig2 = plt.figure(figsize=(8, 5))
    gMed =  annTab["boxSize"].median()
    gStd = annTab["boxSize"].std()
    print("[INFO] Global Median: {:.3f},  Global Stdev:  {:.3f}".          format(gMed, gStd))
    H = 1.0/ np.sqrt(2.0 * np.pi * gStd**2.0)
    xNorm = np.linspace(gMed-3*gStd, gMed+3*gStd, 1000)
    yNorm = H * np.exp(-0.5 * ((xNorm-gMed)/gStd)**2.0)
    ax2 = fig2.add_subplot(111, sharex=fig1.axes[0])
    sns.distplot(annTab["boxSize"], bins=30, kde=False, rug=False,
                 norm_hist=True, label="All Classes", ax=ax2)
    ax2.legend()
    ax2.plot(xNorm, yNorm, color='k', linestyle="--", linewidth=2)
    ax2.set_xlabel("Size (pixels)")
    ax2.yaxis.set_visible(False)
    xRange = annTab["boxSize"].max() - annTab["boxSize"].min()
    xMin = annTab["boxSize"].min() - xRange * 0.05
    xMax = annTab["boxSize"].max() + xRange * 0.05
    ax2.set_xlim(xMin, xMax)
    if args["log"]:
        ax2.set_yscale('log')
    outFig1 = os.path.join(outDir, "fig_size_grid.pdf")
    outFig2 = os.path.join(outDir, "fig_size_ensemble.pdf")
    if args["show"]:
        fig1.show()
        fig2.show()
        input("Press <Return> to continue ...")
    else:
        fig1.savefig(outFig1)
        fig2.savefig(outFig2)
def plot_class_grid(fig, df,  nCols=5, axAspect=3, log=False):
    labels = sorted(list(df["label"].unique()))
    nRows = m.ceil(len(labels)/nCols)
    bBox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    figWidth = bBox.width
    figHeight = figWidth * nRows / (2 * axAspect)
    fig.set_size_inches(figWidth, figHeight, forward=True)
    medMedLst = []
    medStdLst = []
    medMinLst = []
    medMaxLst = []
    axLst = []
    for i, label in enumerate(labels):
        if i>0:
            shareX = axLst[0]
        else:
            shareX = None
        axLst.append(fig.add_subplot(nRows,nCols,i+1, sharex=shareX))
        axLst[-1].yaxis.set_visible(False)
        axLst[-1].grid(True)
        axLst[-1].xaxis.tick_top()
        axLst[-1].xaxis.set_label_position("top")
        if i>nCols-1:
            plt.setp(axLst[-1].get_xticklabels(), visible=False)
        else:
            axLst[-1].set_xlabel("Size (pixels)")
        dfClass = df[df["label"]==label]
        gMed =  dfClass["boxSize"].median()
        gStd = dfClass["boxSize"].std()
        gMin =  dfClass["boxSize"].min()
        gMax =  dfClass["boxSize"].max()
        medMedLst.append(gMed)
        medStdLst.append(gStd)
        medMinLst.append(gMin)
        medMaxLst.append(gMax)
        print("[INFO] {} class: med = {:1f}, std = {:1f}, "
              .format(label, gMed, gStd), end="")
        print("min = {:1f}, max = {:1f}".format(gMin, gMax))
        labelStr = label + " " + str(len(dfClass))
        sns.distplot(dfClass["boxSize"], bins=30, kde=False, rug=False,
                     norm_hist=True, label=labelStr, ax=axLst[-1])
        axLst[-1].legend()
        xRange = dfClass["boxSize"].max() - dfClass["boxSize"].min()
        xMin = dfClass["boxSize"].min() - xRange * 0.05
        xMax = dfClass["boxSize"].max() + xRange * 0.05
        axLst[-1].set_xlim(xMin, xMax)
        if log:
            axLst[-1].set_yscale('log')
    fig.tight_layout()
    print("[INFO] ensemble median = {:1f}".format(np.median(medMedLst)))
    print("[INFO] ensemble stdev = {:1f}".format(np.median(medStdLst)))
    print("[INFO] ensemble min = {:1f}".format(np.median(medMinLst)))
    print("[INFO] ensemble max = {:1f}".format(np.median(medMaxLst)))
if __name__ == "__main__":
    main()
