#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from glob import glob
import argparse
def main():
    parser = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("rootDir", metavar="DataDir", nargs="?",
                        default="../DATA_VIDEOS", help="data directory.")
    args = parser.parse_args()
    rootDir = args.rootDir
    for path, folders, files in os.walk(rootDir, topdown=False):
        for f in files:
            root, ext = os.path.splitext(f)
            newName = root + ext.lower()
            newName = newName.replace(' ', '_')
            specials = [r' ', r',', r'&', r'[', r']', r'(', r')', r'__']
            for c in specials:
                if c in newName:
                    newName = newName.replace(c, '_')
            oldName = os.path.join(path, f)
            newName = os.path.join(path, newName)
            print(oldName, "-->")
            print(newName, "\n")
            os.rename(oldName, newName)
        for i in range(len(folders)):
            newName = folders[i].replace(' ', '_')
            specials = [r' ', r',', r'&', r'[', r']', r'(', r')', r'__']
            for c in specials:
                if c in newName:
                    newName = newName.replace(c, '_')
            oldName = os.path.join(path, folders[i])
            newName = os.path.join(path, newName)
            print(oldName, "-->")
            print(newName, "\n")
            os.rename(oldName, newName)
if __name__ == "__main__":
    main()
