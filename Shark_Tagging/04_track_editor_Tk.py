#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import cv2
import imutils
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import tkinter.messagebox as tkMessageBox
import tkinter.filedialog as tkFileDialog
import tkinter.simpledialog as tkSimpleDialog
from tkinter.scrolledtext import ScrolledText as tkScrolledText
from PIL import Image, ImageTk
from Imports.util_tk import ScrolledTreeTab
from Imports.util_tk import TrackScrubber
from Imports.track_manager import TrackManager
class App(ttk.Frame):
    def __init__(self, parent, videoFile, imageWidth, seek=100, pad=5,
                 cacheDir="", bgColour=None, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.videoFile = videoFile
        self.seek = seek
        self.pad = pad
        self.bgColour=bgColour
        self.parent.protocol("WM_DELETE_WINDOW", self._applicationExit)
        _, videoFileName = os.path.split(videoFile)
        self.parent.title("SharkAI Track Editor: '{}'".format(videoFileName))
        dpiScale = imageWidth / 1300
        useCache = False
        cacheFile = ""
        if len(cacheDir) > 0:
            if os.path.isdir(cacheDir):
                cacheFile = os.path.join(cacheDir, "frame.cache")
                useCache = True
            else:
                print("[WARN] cache path is not a valid directory")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.videoPane = VideoPanel(self, imageWidth, bgColour=self.bgColour,
                                    pad=pad)
        self.videoPane.grid(column=0, row=0, sticky="NEW")
        self.controlPane = ControlPanel(self, bgColour=bgColour, pad=pad)
        self.controlPane.grid(column=1, row=0, sticky="NSEW")
        handleSize = int(round(8 * dpiScale))
        barHeight = int(round(20 * dpiScale))
        self.scrubber = TrackScrubber(self, width=imageWidth + 350,
                                      handlesize=handleSize,
                                      barHeight=barHeight,
                                      widgetPadding=pad)
        self.scrubber.grid(column=0, row=1, columnspan=2, sticky="N")
        if useCache:
            self.cacheMon = CacheMonitor(self)
            self.cacheMon.grid(column=0, row=2, columnspan=2, sticky="NSEW")
        self.trackMan = TrackManager(self.videoFile,
                                     useCache=useCache,
                                     cacheFile=cacheFile,
                                     debug=True)
        if useCache:
            self.cacheMon._set(self.trackMan)
            self.cacheMon._update_progress()
        tmp = np.array(self.trackMan.trackSummary)
        self.controlPane.trackTab.insert_rows(tmp,
                                    ("Track", "Length", "Label", "Quality"))
        self.controlPane.trackTab.set_row_selected(0)
        self.trackMan.set_track(0)
        resolutionStr = "{:d} x {:d} pixels".format(self.trackMan.nXpix,
                                                    self.trackMan.nYpix)
        self.videoPane.set_resolution(resolutionStr)
        self._update_video_display()
        self.scrubber.set_range(xMin=0, xMax=self.trackMan.Nfrm-1,
                                runTime_s = self.trackMan.runTime_s)
        self._update_track_display(updateMask=True)
        self.parent.bind("<<box_drawn>>",
                         lambda event : self._on_box_drawn(event))
        self.parent.bind("<<toggle_ann>>",
                         lambda event : self._on_toggle_ann(event))
        self.parent.bind("<<edit_clear>>",
                         lambda event : self._on_edit_clear(event))
        self.parent.bind("<<create_edit>>",
                         lambda event : self._on_create_edit(event))
        self.parent.bind("<<track_selected>>",
                         lambda event : self._on_select_track(event))
        self.parent.bind("<<select_frame>>",
                         lambda event : self._on_select_frame(event))
        self.parent.bind("<<cull_tracks>>",
                         lambda event : self._on_cull_tracks(event))
        self.parent.bind("<<del_track>>",
                         lambda event : self._on_del_track(event))
        self.parent.bind("<<split_track>>",
                         lambda event : self._on_split_track(event))
        self.parent.bind("<<merge_track>>",
                         lambda event : self._on_merge_track(event))
        self.parent.bind("<<undo>>",
                         lambda event : self._on_undo(event))
        self.parent.bind("<<relab_track>>",
                         lambda event : self._on_relab_track(event))
        self.parent.bind("<<setqual_track>>",
                         lambda event : self._on_setqual_track(event))
        self.parent.bind("<<intrp_track>>",
                         lambda event : self._on_interp_track(event))
        self.parent.bind("<<flag_frames>>",
                         lambda event : self._on_flag_frames(event))
        self.parent.bind("<<neg_track>>",
                         lambda event : self._on_add_neg_track(event))
        self.parent.bind("<<save>>",
                         lambda event : self._on_save(event))
        self.parent.bind("<<quit>>",
                         lambda event : self._on_quit(event))
        self.parent.bind("<Left>",
                         lambda event : self._on_retard(event))
        self.parent.bind("<Right>",
                         lambda event : self._on_advance(event))
        self.parent.minsize(self.parent.winfo_width(),
                            self.parent.winfo_height())
        self.parent.resizable(False, False)
    def _applicationExit(self):
        print("[INFO] exiting")
        self.parent.destroy()
        self.trackMan.cleanup()
        cv2.destroyAllWindows()
    def _update_video_display(self):
        try:
            self.videoPane.display_frame(self.trackMan.frameBGR)
        except:
            print("[WARN] error displaying frame")
    def _update_track_display(self, updateMask=False):
        tRec = self.trackMan.get_track()
        self.scrubber.update_tracklines(tRec["frm"], tRec["mask"])
        if updateMask:
            mask = self.trackMan.get_tracks_mask()
            self.scrubber.update_trackmask(mask)
    def _update_edit_display(self):
        eRec = self.trackMan.get_edit()
        self.scrubber.update_editlines(eRec["frm"], eRec["mask"])
    def _refresh_tracks(self):
        self.controlPane.trackTab.clear_entries()
        tmp = np.array(self.trackMan.trackSummary)
        self.controlPane.trackTab.insert_rows(tmp,
                                    ("Track", "Length", "Label", "Quality"))
        self.controlPane.trackTab.set_row_selected(self.trackMan.currentTrack)
    def _on_box_drawn(self, event=None):
        coords = self.videoPane.get_box()
        self.trackMan.put_box_edit(coords)
        self.trackMan.set_track(self.trackMan.currentTrack)
        self._update_video_display()
        self._update_edit_display()
    def _on_toggle_ann(self, event=None):
        if self.trackMan.doAnnTrackNum == True:
            self.trackMan.doAnnTrackNum = False
        else:
            self.trackMan.doAnnTrackNum = True
        self.trackMan.set_frame(preSeek=self.seek)
        self._update_video_display()
    def _on_edit_clear(self, event=None):
        self.trackMan.clear_edits()
        self.trackMan.set_track(self.trackMan.currentTrack)
        self._update_video_display()
        self._update_edit_display()
    def _on_create_edit(self, event=None):
        print("[INFO] creating new track via interpolation")
        self.trackMan.create_track_from_edit()
        self._refresh_tracks()
        mask = self.trackMan.get_tracks_mask()
        self.scrubber.update_trackmask(mask)
        self.event_generate("<<edit_clear>>")
    def _on_select_track(self, event=None):
        trackNum = int(event.widget.get_text_selected()[0]) -1
        self.trackMan.set_track(trackNum)
        self._update_video_display()
        self._update_track_display()
    def _on_select_frame(self, event=None):
        frm = int(self.scrubber.valueHandle.get())
        self.trackMan.set_frame(frm, self.seek)
        self._update_video_display()
    def _on_cull_tracks(self, event=None):
        cullLim = self.controlPane.cullLim.get()
        print("[INFO] culling tracks with counts below {:d}".format(cullLim))
        self.trackMan.cull_tracks(cullLim)
        self._refresh_tracks()
        mask = self.trackMan.get_tracks_mask()
        self.scrubber.update_trackmask(mask)
    def _on_del_track(self, event=None):
        tracks = self.controlPane.trackTab.get_all_indx_selected()
        print("[INFO] deleting tracks {}".format(tracks))
        self.trackMan.delete_tracks(tracks)
        self._refresh_tracks()
        mask = self.trackMan.get_tracks_mask()
        self.scrubber.update_trackmask(mask)
    def _on_split_track(self, event=None):
        print("[INFO] splitting track {:d}"
              .format(self.trackMan.currentTrack))
        self.trackMan.split_track()
        self._refresh_tracks()
    def _on_merge_track(self, event=None):
        tracks = self.controlPane.trackTab.get_all_indx_selected()
        print("[INFO] merging tracks {}".format(tracks))
        self.trackMan.merge_tracks(tracks)
        self._refresh_tracks()
    def _on_undo(self, event=None):
        print("[INFO] undoing last operation")
        self.trackMan.undo()
        self._refresh_tracks()
        mask = self.trackMan.get_tracks_mask(recalculate=False)
        self.scrubber.update_trackmask(mask)
    def _on_interp_track(self, event=None):
        xRng =  self.scrubber.get_selrange()
        diff = abs(xRng[1] - xRng[0])
        if diff == 0:
            print("[WARN] please select a frame range first")
        else:
            print("[INFO] interpolating track between frames", xRng)
            self.trackMan.interpolate_track(xRng[0], xRng[1])
            self._refresh_tracks()
            mask = self.trackMan.get_tracks_mask()
            self.scrubber.update_trackmask(mask)
    def _on_add_neg_track(self, event=None):
        xRng =  self.scrubber.get_selrange()
        diff = abs(xRng[1] - xRng[0])
        if diff == 0:
            print("[WARN] please select a frame range first")
        else:
            print("[INFO] adding NEG track track between frames", xRng)
            self.trackMan.add_neg_track(xRng[0], xRng[1])
            self._refresh_tracks()
            mask = self.trackMan.get_tracks_mask()
            self.scrubber.update_trackmask(mask)
    def _on_flag_frames(self, event=None):
        xRng =  self.scrubber.get_selrange()
        diff = abs(xRng[1] - xRng[0])
        if diff == 0:
            print("[WARN] please select a frame range first")
        else:
            tracks = self.controlPane.trackTab.get_all_indx_selected()
            print("[INFO] flagging frames in range", xRng)
            if self.controlPane.flagOpt.get() == "Selected Tracks":
                print("[INFO] applying to tracks {}".format(tracks))
                self.trackMan.flag_framerange(xRng[0], xRng[1], tracks)
            else:
                print("[INFO] applying to ALL tracks")
                self.trackMan.flag_framerange(xRng[0], xRng[1],
                                              flagAllTracks=True)
        self._refresh_tracks()
        mask = self.trackMan.get_tracks_mask()
        self.scrubber.update_trackmask(mask)
    def _on_relab_track(self, event=None):
        labelNew = self.controlPane.relabTrack.get()
        tracks = self.controlPane.trackTab.get_all_indx_selected()
        print("[INFO] re-labelling tracks {} as {}".format(tracks, labelNew))
        self.trackMan.relabel_track(labelNew, tracks)
        self._refresh_tracks()
    def _on_setqual_track(self, event=None):
        qualNew = int(self.controlPane.qualTrack.get())
        tracks = self.controlPane.trackTab.get_all_indx_selected()
        print("[INFO] setting quality of tracks {} to {}".
              format(tracks, qualNew))
        self.trackMan.setqual_track(qualNew, tracks)
        self._refresh_tracks()
    def _on_save(self, event=None):
        print("[INFO] saving JSON tracks")
        self.trackMan.save_json()
    def _on_quit(self, event=None):
        self._applicationExit()
    def _on_retard(self, event=None):
        self.scrubber._retard_frame()
    def _on_advance(self, event=None):
        self.scrubber._advance_frame()
class VideoPanel(ttk.Frame):
    def __init__(self, parent, width=1080, aspect=1.7,  bgColour=None,
                 pad=3, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        if bgColour is None:
            bgColour = ttk.Style().lookup("TFrame", "background")
        self.padArgs = {"padx": pad,
                        "pady": pad}
        self.width = width
        self.height = width // aspect
        self.heightOrig = None
        self.widthOrig = None
        self.scale = None
        self.canvas = tk.Canvas(self, width=self.width, height=self.height,
                                background=bgColour, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=3, **self.padArgs)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.resTitle = ttk.Label(self, text="Native Video Resolution: ",
                                  anchor="e")
        self.resTitle.grid(column=0, row=1, sticky="W", **self.padArgs)
        self.resolution = tk.StringVar()
        self.resLab = ttk.Label(self, textvariable=self.resolution, width=30,
                                 anchor="w")
        self.resLab.grid(column=1, row=1, sticky="EW", **self.padArgs)
        self.annBtn = ttk.Button(self, text=u"Track Annotations On \u2611",
                                 width=23,
                                 command=self._toggle_ann)
        self.annBtn.grid(row=1, column=2, sticky="E", **self.padArgs)
        self.x = 0
        self.y = 0
        self.rect = None
        self.x1 = 0.0
        self.y1 = 0.0
        self.x2 = 0.0
        self.y2 = 0.0
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
    def _on_press(self, event):
        self.x1 = self.canvas.canvasx(event.x)
        self.y1 = self.canvas.canvasy(event.y)
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.x, self.y,
                                                     1, 1, outline="blue",
                                                     width=1, tag="rect")
    def _on_drag(self, event):
        self.x2 = self.canvas.canvasx(event.x)
        self.y2 = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.x1, self.y1, self.x2, self.y2)
    def _on_release(self, event):
        self.canvas.delete("rect")
        self.rect = None
        self.event_generate("<<box_drawn>>")
    def _toggle_ann(self):
        if self.annBtn.config("text")[-1] == u"Track Annotations On \u2611":
            self.annBtn.config(text=u"Track Annotations Off \u2610")
        else:
            self.annBtn.config(text=u"Track Annotations On \u2611")
        self.event_generate("<<toggle_ann>>")
    def display_frame(self, frame):
        frameRGBA = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.heightOrig, self.widthOrig = frameRGBA.shape[:2]
        frameRGBA = imutils.resize(frameRGBA, width=self.width)
        self.scale = self.width / self.widthOrig
        pilImg = Image.fromarray(frameRGBA)
        self.canvas.image = ImageTk.PhotoImage(image=pilImg)
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor="nw",
                                 tag="frame_image")
        self.canvas.tag_lower("frame_image")
    def get_box(self):
        xLst = [self.x1, self.x2]
        yLst = [self.y1, self.y2]
        xLst.sort()
        yLst.sort()
        x1, x2 = xLst
        y1, y2 = yLst
        x1 = max(int(round(x1 / self.scale)), 0)
        y1 = max(int(round(y1 / self.scale)), 0)
        x2 = min(int(round(x2 / self.scale)), self.widthOrig -1)
        y2 = min(int(round(y2 / self.scale)), self.heightOrig -1)
        return x1, y1, x2, y2
    def set_resolution(self, resolutionString):
        self.resolution.set(resolutionString)
class ControlPanel(ttk.Frame):
    def __init__(self, parent,  bgColour=None, pad=3, font="Helvetica 10 bold",
                 *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.font = font
        if bgColour is None:
            bgColour = ttk.Style().lookup("TFrame", "background")
        self.padArgs = {"padx": pad,
                        "pady": pad}
        self.padLabArgs = {"padx": pad,
                           "pady": pad * 2}
        self.rowconfigure(3, weight=1)
        self.rowconfigure(9, weight=1)
        self.rowconfigure(14, weight=1)
        self.trackTabLab = ttk.Label(self, text="Tracks in Video:",
                                     anchor="center", font=self.font)
        self.trackTabLab.grid(column=0, row=0, columnspan=2, sticky="NSEW",
                              **self.padLabArgs)
        self.trackTab = ScrolledTreeTab(self, virtEvent="<<track_selected>>")
        self.trackTab.name_columns(("Track", "Length", "Label", "Quality"))
        self.trackTab.grid(column=0, row=1, columnspan=2, sticky="EW",
                           **self.padArgs)
        self.cullBtn = ttk.Button(self, text=u"Cull Tracks \u2264 n Frames:",
                                  width=20,
                        command=lambda: self.event_generate("<<cull_tracks>>"))
        self.cullBtn.grid(column=0, row=2, sticky="EW", **self.padArgs)
        cullLst = [1, 3, 5, 10, 15, 20, 30, 50]
        self.cullLim = tk.IntVar()
        self.cullComb = ttk.Combobox(self,
                                      textvariable=self.cullLim,
                                      values=cullLst, width=15)
        self.cullComb.current(4)
        self.cullComb.grid(column=1, row=2, sticky="EW", **self.padArgs)
        self.opLab = ttk.Label(self,
                               text=("Operations on Selected Tracks:"),
                               anchor="center", font=self.font)
        self.opLab.grid(column=0, row=4, columnspan=2, sticky="EW",
                        **self.padLabArgs)
        self.delBtn = ttk.Button(self, text="Delete Tracks", width=20,
                        command=lambda: self.event_generate("<<del_track>>"))
        self.delBtn.grid(column=0, row=5, sticky="EW", **self.padArgs)
        self.splitBtn = ttk.Button(self, text="Split Current Track", width=20,
                        command=lambda: self.event_generate("<<split_track>>"))
        self.splitBtn.grid(column=1, row=5, sticky="EW", **self.padArgs)
        self.mergeBtn = ttk.Button(self, text="Merge Tracks", width=20,
                        command=lambda: self.event_generate("<<merge_track>>"))
        self.mergeBtn.grid(column=0, row=6, sticky="EW", **self.padArgs)
        self.undoBtn = ttk.Button(self, text="Undo Last", width=20,
                        command=lambda: self.event_generate("<<undo>>"))
        self.undoBtn.grid(column=1, row=6, sticky="EW", **self.padArgs)
        self.relabBtn = ttk.Button(self, text="Re-label Tracks as:", width=20,
                        command=lambda: self.event_generate("<<relab_track>>"))
        self.relabBtn.grid(column=0, row=7, sticky="EW", **self.padArgs)
        relabLst = ["BUL", "HAM", "WHI", "WHA", "SUR",
                    "SWI", "DOL", "BAI", "GAM"]
        self.relabTrack = tk.StringVar()
        self.relabComb = ttk.Combobox(self,
                                      textvariable=self.relabTrack,
                                      values=relabLst, width=15)
        self.relabComb.current(0)
        self.relabComb.grid(column=1, row=7, sticky="EW", **self.padArgs)
        self.setqualBtn = ttk.Button(self, text="Set Track Quality:",
                                     width=20,
                    command=lambda: self.event_generate("<<setqual_track>>"))
        self.setqualBtn.grid(column=0, row=8, sticky="EW", **self.padArgs)
        qualLst = ["1", "2", "3", "4", "5", "6", "7", "8", "9","10"]
        self.qualTrack = tk.StringVar()
        self.setqualComb = ttk.Combobox(self,
                                      textvariable=self.qualTrack,
                                      values=qualLst, width=15)
        self.setqualComb.current(4)
        self.setqualComb.grid(column=1, row=8, sticky="EW", **self.padArgs)
        self.selLab = ttk.Label(self,
                            text="Operations on Selected Frames or Key Boxes:",
                            anchor="center", font=self.font)
        self.selLab.grid(column=0, row=10, columnspan=2, sticky="EW",
                         **self.padLabArgs)
        self.flagBtn = ttk.Button(self, text="Flag Frame Range for:", width=21,
                        command=lambda: self.event_generate("<<flag_frames>>"))
        self.flagBtn.grid(column=0, row=11, sticky="EW", **self.padArgs)
        flagOptLst = ["Selected Tracks", "All Tracks"]
        self.flagOpt = tk.StringVar()
        self.flagComb = ttk.Combobox(self, state="readonly",
                                     textvariable=self.flagOpt,
                                     values=flagOptLst, width=15)
        self.flagComb.current(0)
        self.flagComb.grid(column=1, row=11, sticky="EW", **self.padArgs)
        self.interpBtn = ttk.Button(self, text="Interpolate Gaps", width=20,
                        command=lambda: self.event_generate("<<intrp_track>>"))
        self.interpBtn.grid(column=0, row=12, sticky="EW", **self.padArgs)
        self.negBtn = ttk.Button(self, text="Add NEG Track", width=20,
                        command=lambda: self.event_generate("<<neg_track>>"))
        self.negBtn.grid(column=1, row=12, sticky="EW", **self.padArgs)
        self.createTrkBtn = ttk.Button(self, text="Create Track From Key",
                                       width=20,
                        command=lambda: self.event_generate("<<create_edit>>"))
        self.createTrkBtn.grid(column=0, row=13, sticky="EW", **self.padArgs)
        self.clearEditBtn = ttk.Button(self, text="Clear Key Boxes",
                                       width=20,
                        command=lambda: self.event_generate("<<edit_clear>>"))
        self.clearEditBtn.grid(column=1, row=13, sticky="EW", **self.padArgs)
        self.saveBtn = tk.Button(self, text="Save JSON", width=20,
                                 background="lightgreen",
                        command=lambda: self.event_generate("<<save>>"))
        self.saveBtn.grid(column=0, row=15, sticky="EW", **self.padArgs)
        self.quitBtn = tk.Button(self, text="Quit", width=20,
                                 background="orange",
                        command=lambda: self.event_generate("<<quit>>"))
        self.quitBtn.grid(column=1, row=15, sticky="EW", **self.padArgs)
class CacheMonitor(tk.Frame):
    def __init__(self, parent,  bgColour=None, pad=3, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.padArgs = {"padx": pad,
                        "pady": pad}
        self.cacheLab = ttk.Label(self, text="Frame Cache Monitor:")
        self.cacheLab.grid(column=0, row=0, sticky="NSEW", **self.padArgs)
        self.pBar = ttk.Progressbar(self, orient="horizontal",
                                    length=300, mode="determinate")
        self.pBar.grid(column=1, row=0, sticky="nsew", **self.padArgs)
        self.count = tk.StringVar()
        self.countLab = ttk.Label(self, textvariable=self.count, width=30)
        self.countLab.grid(column=2, row=0, sticky="nsew", **self.padArgs)
        self.size = tk.StringVar()
        self.sizeLab = ttk.Label(self, textvariable=self.size, width=20,
                                 anchor="e")
        self.sizeLab.grid(column=3, row=0, sticky="nsew", **self.padArgs)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(0, weight=0)
    def _set(self, trackMan):
        self.trackMan = trackMan
        self.minVal = 0
        self.pBar["value"] = self.minVal
        self.maxVal = self.trackMan.Nfrm
        self.pBar["maximum"] = self.maxVal
    def _update_progress(self):
        nFrm, size_GB = self.trackMan._query_cache_state()
        self.pBar["value"] = nFrm
        self.count.set("{:11d} / {:d} Frames".format(nFrm, self.trackMan.Nfrm))
        self.size.set("{:.2f} GB".format(size_GB))
        root.after(1000, self._update_progress)  
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input video")
    ap.add_argument("-w", "--width", type=int, default=1300,
                    help="width of the image display, in pixels [1300].")
    ap.add_argument("-s", "--seek", type=int, default=100,
                    help="seeking window [100]. Set to '-1' if boxes offset")
    ap.add_argument("-c", "--cacheDir", nargs='?', default="",
                    const="./",
                    help="directory in which to make a cache file [./]")
    args = vars(ap.parse_args())
    root = tk.Tk()
    pad = 4
    app = App(parent=root,
              videoFile=args["input"],
              imageWidth=args["width"],
              seek=args["seek"],
              pad=pad,
              cacheDir=args["cacheDir"])
    for child in app.winfo_children():
        child.grid_configure(padx=pad, pady=pad)
    fontScale = int(args["width"] / 43)
    ttk.Style().configure('Treeview', rowheight=fontScale)
    app.pack(side="top", fill="both", expand=False)
    root.mainloop()
