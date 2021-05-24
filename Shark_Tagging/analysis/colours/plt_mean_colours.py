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
import json
import os
import sys
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="input dataset directory")
ap.add_argument("-c", "--column", default="A",
                help="column name to plot (H S V R G B A) [A]")
ap.add_argument("-s", "--show", action="store_true",
                help="show the figures (default = save) [N]")
args = vars(ap.parse_args())
def main():
    inFile = os.path.join(args["dataset"], "colours_by_box.hdf5")
    df = pd.read_hdf(inFile, "colourTab")
    colCodeLst = ["H",  "S", "V"]
    if args["column"] in colCodeLst:
        colCodeLst = [args["column"]]
    print("[INFO] plotting colours {}".format(colCodeLst))
    outDir = os.path.join(args["dataset"],"PLOTS_COLOUR")
    if not args["show"]:
        print("[INFO] will save figures to {}".format(outDir))
        if not os.path.exists(outDir):
            os.makedirs(outDir)
    xLabels = {"H": "Colour Hue",
               "S": "Colour Saturation",
               "V": "Colour Value",
               "R": "Red",
               "G": "Green",
               "B": "Blue"}
    fig1 = plt.figure(figsize=(10, 7))
    fig2 = plt.figure(figsize=(8, 5))
    distPropDict = {}
    for colCode in colCodeLst:
        fig1.clf()
        fig2.clf()
        xLabel = xLabels[colCode]
        print("\n[INFO] plotting '{}'".format(xLabel))
        plot_class_grid(fig1, df, colCode, xLabel)
        gMed =  df[colCode].median()
        gStd = MAD(df[colCode].values)
        distPropDict[colCode] = [gMed, gStd]
        print("[INFO] Global Median: {:.3f},  Global Stdev:  {:.3f}".              format(gMed, gStd))
        H = 1.0/ np.sqrt(2.0 * np.pi * gStd**2.0)
        xNorm = np.linspace(gMed-3*gStd, gMed+3*gStd, 1000)
        yNorm = H * np.exp(-0.5 * ((xNorm-gMed)/gStd)**2.0)
        ax2 = fig2.add_subplot(111, sharex=fig1.axes[0])
        sns.distplot(df[colCode], kde=True, rug=False,
                     norm_hist=True, label="All Classes", ax=ax2)
        ax2.legend()
        ax2.plot(xNorm, yNorm, color='k', linestyle="--", linewidth=2)
        ax2.set_xlabel(xLabel)
        ax2.yaxis.set_visible(False)
        ax2.set_xlim(0,1)
        outFig1 = os.path.join(outDir, "fig_" + colCode + "_grid.pdf")
        outFig2 = os.path.join(outDir, "fig_" + colCode + "_ensemble.pdf")
        if args["show"]:
            fig1.show()
            fig2.show()
            input("Press <Return> to continue ...")
        else:
            fig1.savefig(outFig1)
            fig2.savefig(outFig2)
    if not args["show"]:
        outFile = os.path.join(outDir, "ensemble_meanstd.json")
        with open(outFile, 'w') as FH:
            json.dump(distPropDict, FH)
def plot_class_grid(fig, df, colCode, xLabel=None,  nCols=3, axAspect=4):
    labels = list(df["label"].unique())
    nRows = m.ceil(len(labels)/nCols)
    bBox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    figWidth = bBox.width
    figHeight = figWidth * nRows / (2 * axAspect)
    fig.set_size_inches(figWidth, figHeight, forward=True)
    if xLabel is None:
        xLabel = colCode
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
            axLst[-1].set_xlabel(xLabel)
        dfClass = df[df["label"]==label]
        gMed =  dfClass[colCode].median()
        gStd = dfClass[colCode].std()
        gMin =  dfClass[colCode].min()
        gMax =  dfClass[colCode].max()
        print("[INFO] {} class: med = {:1f}, std = {:1f}, "
              .format(label, gMed, gStd), end="")
        print("min = {:1f}, max = {:1f}".format(gMin, gMax))
        labelStr = label + " " + str(len(dfClass))
        sns.distplot(dfClass[colCode], bins=30, kde=True, rug=False,
                     norm_hist=True, label=labelStr, ax=axLst[-1])
        axLst[-1].legend()
        axLst[-1].set_xlim(0,1)
    axLst[0].set_xlim(0,1)
    fig.tight_layout()
def MAD(a, c=0.6745, axis=None):
    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)
    return m
if __name__ == "__main__":
    main()
