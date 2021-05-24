#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import shutil
import json
from glob import glob
import argparse
import numpy as np
def main():
    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("rootDir", nargs="?", default="../DATA_VIDEOS",
                    help="directory containing data and tracks")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print long messages")
    args = vars(ap.parse_args())
    trackFileLst = [y for x in os.walk(args["rootDir"], followlinks=True)
                    for y in glob(os.path.join(x[0], '*_tracks.json'))]
    if args["verbose"]:
        print("[INFO] found the following JSON files:")
        for e in trackFileLst:
            print(e)
        print("[INFO] total of {:d} JSON files".format(len(trackFileLst)))
        input("\nPress <Return> to continue ...")
    outIndxFile = "videoIndx.json"
    if os.path.exists(outIndxFile):
        print("[INFO] reading existing file {}".format(outIndxFile))
        with open(outIndxFile, 'r') as FH:
            indxDict = json.load(FH)
        fileParmTab = np.array(list(indxDict.values()))
        try:
            maxIndx = int(np.max(fileParmTab[:,0]))
        except Exception:
            print("[WARN] could not determine max index (empty index file?)")
            maxIndx = -1
    else:
        indxDict = {}
        maxIndx = -1
    print("[INFO] starting at index {:d}".format(maxIndx + 1))
    trackCount = 0
    for fileCount, trackFile in enumerate(trackFileLst):
        with open(trackFile, 'r') as FH:
            trackLst = json.load(FH)
            nTracks = len(trackLst)
            trackCount += nTracks
        videoFile = trackFile[:-12] + ".mp4"
        if args["verbose"]:
            print("[INFO] processing {}".format(videoFile))
        if videoFile in indxDict:
            fileIndex = indxDict[videoFile][0]
            print("[INFO] used existing index {:d}".format(fileIndex))
        else:
            fileIndex = maxIndx + 1
            maxIndx += 1
            print("[INFO] added new index {:d}".format(fileIndex))
        indxDict[videoFile] = [fileIndex, nTracks]
    outIndxFile = "videoIndx.json"
    if os.path.exists(outIndxFile):
        shutil.copy(outIndxFile, outIndxFile + ".bak")
    with open(outIndxFile, 'w') as FH:
        FH.write(json.dumps(indxDict, indent=2))
    print("Indexed track files for {:d} videos containing {:d} tracks."          .format(fileCount + 1, trackCount))
if __name__ == "__main__":
    main()
