#!/usr/bin/env python
try:
    import Tkinter as tk
    import ttk
    import tkFont
except Exception:
    import tkinter as tk
    from tkinter import ttk
    import tkinter.font as tkFont
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import copy
class ScrolledTreeTab(ttk.Frame):
    def __init__(self, parent, virtEvent="<<tab_row_selected>>", strPad=10,
                 selectMode="extended", *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)
        self.parent = parent
        self.rowSelected = None
        self.textSelected = None
        self.virtEvent = virtEvent
        self.strPad = strPad
        self.tree = ttk.Treeview(self, show="headings")
        vsb = ttk.Scrollbar(self, orient="vertical",
                            command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal",
                            command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky="NSWE")
        vsb.grid(column=1, row=0, sticky="NS")
        hsb.grid(column=0, row=1, sticky="WE")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.tree.configure(selectmode=selectMode)
        self.tree.bind("<<TreeviewSelect>>", self._on_row_select)
    def _on_row_select(self, event=None):
        item =  event.widget.selection()
        if not item == "" and len(item)==1:
            indx = event.widget.item(item, "text")
            self.rowSelected = int(indx)
            self.textSelected = event.widget.item(item, "value")
            self.event_generate(self.virtEvent)
    def _sortby(self, tree, col, descending):
        data = [(tree.set(child, col), child)
                for child in tree.get_children('')]
        data = self._change_numeric_onestep(data)
        data.sort(reverse=descending)
        for i, item in enumerate(data):
            tree.move(item[1], '', i)
        tree.heading(col, command=lambda col=col:                     self._sortby(tree, col, int(not descending)))
    def _change_numeric_onestep(self, data):
        newData = []
        try:
            for child, col in data:
                if child=="None":
                    child = "-inf"   
                newData.append((float(child), col))
            return newData
        except Exception:
            return data
    def name_columns(self, colNames):
        self.tree['columns'] = colNames
        for col in colNames:
            self.tree.heading(col, text=col, command=lambda c=col:
                              self._sortby(self.tree, c, 0))
            strWidth = tkFont.Font().measure(col.title())
            self.tree.column(col, width=strWidth + self.strPad)
            self.tree.column(col, minwidth=strWidth + self.strPad)
    def insert_rows(self, rows, colNames=None):
        if colNames is None:
            colNames = ["Row "+ str(x+1) for x in range(len(rows[0]))]
        if len(self.tree['columns'])==0:
            self.tree['columns'] = colNames
            for col in colNames:
                self.tree.heading(col, text=col, command=lambda c=col:
                                  self._sortby(self.tree, c, 0))
            strWidth = tkFont.Font().measure(col.title())
            self.tree.column(col, width=strWidth + self.strPad)
            self.tree.column(col, minwidth=strWidth + self.strPad)
        rowIndx = 0
        for row in rows:
            row = [str(x) for x in row]   
            self.tree.insert('', 'end', values=row, text=str(rowIndx))
            rowIndx += 1
            for i, val in enumerate(row):
                strWidth = tkFont.Font().measure(val.title())
                if self.tree.column(colNames[i], width=None)<                   (strWidth + self.strPad):
                    self.tree.column(colNames[i], width=strWidth +
                                     self.strPad)
                    self.tree.column(colNames[i], minwidth=strWidth +
                                     self.strPad)
    def get_indx_selected(self):
        if self.rowSelected is None:
            return None
        else:
            return int(self.rowSelected)
    def get_all_indx_selected(self):
        itemLst = self.tree.selection()
        rows = []
        for item in itemLst:
            rows.append(int(self.tree.item(item, "text")))
        return rows
    def set_row_selected(self, row):
        childID = self.tree.get_children()[row]
        self.tree.selection_set(childID)
    def get_text_selected(self):
        if self.textSelected is None:
            return None
        else:
            return self.textSelected
    def get_all_text(self):
        try:
            itemLst = self.tree.get_children()
            valueLst = []
            for item in itemLst:
                valueLst.append(self.tree.item(item, "value"))
            return valueLst
        except Exception:
            return None
    def insert_recarray(self, arr):
        colNames = arr.dtype.names
        self.insert_rows(arr, colNames)
    def clear_selected(self):
        try:
            idx = self.tree.selection()
            self.tree.delete(idx)
            self.rowSelected = None
            self.textSelected = None
            return idx
        except Exception:
            return None
    def clear_entries(self):
        try:
            x = self.tree.get_children()
            for entry in x:
                self.tree.delete(entry)
            self.rowSelected = None
        except Exception:
            pass
class ScrolledCanvasFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)
        self.parent = parent
        self.canvas = tk.Canvas(self, border=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical",
                            command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal",
                            command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.canvas.grid(column=0, row=0, sticky="NSEW")
        vsb.grid(column=1, row=0, sticky="NS")
        hsb.grid(column=0, row=1, sticky="WE")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.interior = ttk.Frame(self.canvas)
        self.winID = self.canvas.create_window((0,0), window=self.interior,
                                         anchor="nw", tags="self.interior")
        self.interior.bind('<Configure>', self._configure_interior)
        self.canvas.bind('<Configure>', self._configure_canvas)
    def _configure_interior(self, event):
        size = (self.interior.winfo_reqwidth(),
                self.interior.winfo_reqheight())
        self.canvas.config(scrollregion="0 0 %s %s" % size)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.config(width=self.interior.winfo_reqwidth())
    def _configure_canvas(self, event):
        if self.interior.winfo_reqwidth() != self.canvas.winfo_width():
            self.canvas.itemconfigure(self.winID,
                                      width=self.canvas.winfo_width())
class ScrolledListBox(ttk.Frame):
    def __init__(self, parent, selectmode="single",
                 virtEvent="<<list_row_selected>>", *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)
        self.parent = parent
        self.virtEvent = virtEvent
        self.rowSelected = None
        self.textSelected = None
        self.listBox = tk.Listbox(self, selectmode=selectmode)
        vsb = ttk.Scrollbar(self, orient="vertical",
                            command=self.listBox.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal",
                            command=self.listBox.xview)
        self.listBox.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.listBox.grid(column=0, row=0, sticky="NSWE")
        vsb.grid(column=1, row=0, sticky="NS")
        hsb.grid(column=0, row=1, sticky="WE")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.listBox.bind("<ButtonRelease-1>", self._on_row_select)
    def insert(self, item):
        self.listBox.insert(tk.END, str(item))
    def insert_list(self, items):
        for item in items:
            self.listBox.insert(tk.END, str(item))
    def clear(self):
        self.listBox.delete(0, tk.END)
    def clear_selected(self):
        try:
            index = self.listBox.curselection()[0]
            self.listBox.delete(index)
        except Exception:
            pass
    def get_text_selected(self):
        try:
            index = self.listBox.curselection()[0]
            return self.listBox.get(index)
        except Exception:
            return None
    def get_row_selected(self):
        try:
            return self.listBox.curselection()[0]
        except Exception:
            return None
    def get_all_text(self):
        try:
            return self.listBox.get(0, tk.END)
        except Exception:
            return None
    def _on_row_select(self, event=None):
        try:
            self.rowSelected = self.listBox.curselection()[0]
            self.textSelected = self.listBox.get(self.rowSelected)
            self.event_generate(self.virtEvent)
        except Exception:
            pass
class SingleFigFrame(ttk.Frame):
    def __init__(self, parent, *args, **kw):
        ttk.Frame.__init__(self, parent, *args, **kw)
        self.parent = parent
        self.fig = Figure(figsize=(4.5, 4.0))
        self.figCanvas = FigureCanvasTkAgg(self.fig, master=self)
        self.figCanvas.show()
        self.canvas = self.figCanvas.get_tk_widget()
        self.canvas.grid(column=0, row=0, padx=0, pady=0, sticky="NSEW")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.ax = None
    def add_axis(self):
        if self.ax is None:
            self.ax = self.fig.add_subplot(111)
        return self.ax
    def show(self):
        self.figCanvas.show()
class TrackScrubber(ttk.Frame):
    def __init__(self, parent, width=400, handlesize=8, from_=0, to=100,
                 barHeight=20, linewidth=3, yPad=20, xPad=20,  labFmt="{}",
                 widgetPadding=3, releaseEvent="<<select_frame>>", ):
        ttk.Frame.__init__(self, parent)
        self.parent = parent
        bgColour = ttk.Style().lookup("TFrame", "background")
        self.padArgs = {"padx": widgetPadding,
                        "pady": widgetPadding}
        self.releaseEvent = releaseEvent
        self.width = width
        self.yZero = yPad + barHeight * 2
        self.height = self.yZero + handlesize*3 + linewidth
        xPad = max(xPad, handlesize/2.0)
        self.canMin = xPad
        self.canMax = width-xPad
        self.lineWidth = linewidth
        self.barHeight = barHeight
        self.handleSize = handlesize
        self.labFmt = labFmt
        self.xMin = from_
        self.xMax = to
        self.runTime_s = None
        self.canRng = float(self.canMax - self.canMin)
        self.xRng = float(self.xMax - self.xMin)
        self.valueHandle = tk.DoubleVar()
        self.valueHandle.set(0)
        self.timeHandle = tk.StringVar()
        self.timeHandle.set("{:10.0f}m : {:05.2f}s".format(0, 0))
        self.limitLeft = None
        self.limitRight = None
        self.x = None
        self.xEdge = None
        self.selMode = False
        self.selLeft = None
        self.selRight = None
        self.editMode = False
        self.canvas = tk.Canvas(self, background=bgColour, width=self.width,
                                height=self.height, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=10, padx=0, pady=0)
        self._draw_ruler()
        h = self._draw_handle(self._world2canvas(0))
        self._create_bindings()
        self.selModeBtn = ttk.Button(self, text="Select Mode Off \u2610",
                                     width=15,
                                     command=self._toggle_selmode)
        self.selModeBtn.grid(row=1, column=0, sticky="E", **self.padArgs)
        self.clearSelBtn = ttk.Button(self, text="Clear Selection",
                                      width=15,
                                      command=self._clear_selection)
        self.clearSelBtn.grid(row=1, column=1, sticky="E", **self.padArgs)
        self.frmLab = ttk.Label(self, text="Frame Number: ",
                                anchor="e")
        self.frmLab.grid(row=1, column=2, sticky="EW", **self.padArgs)
        self.curLab = ttk.Label(self, textvariable=self.valueHandle,
                                anchor="w")
        self.curLab.grid(row=1, column=3, sticky="EW", **self.padArgs)
        self.timeLab = ttk.Label(self, text="Time: ", anchor="e")
        self.timeLab.grid(row=1, column=4, sticky="EW", **self.padArgs)
        self.timeValLab = ttk.Label(self, textvariable=self.timeHandle,
                                    anchor="w")
        self.timeValLab.grid(row=1, column=5, sticky="EW", **self.padArgs)
        self.prevBtn = ttk.Button(self, text="<",
                                  width=10,
                                  command=self._retard_frame)
        self.prevBtn.grid(row=1, column=6, sticky="E", **self.padArgs)
        self.skipLab = ttk.Label(self, text="Skip: ", anchor="e")
        self.skipLab.grid(row=1, column=7, sticky="E", **self.padArgs)
        frameSkipLst = ["1", "2", "3", "4", "5", "10", "20", "30", "50", "100"]
        self.frameSkip = tk.StringVar()
        self.frameSkipComb = ttk.Combobox(self,
                                          textvariable=self.frameSkip,
                                          values=frameSkipLst, width=3)
        self.frameSkipComb.current(0)
        self.frameSkipComb.grid(row=1, column=8, sticky="E", **self.padArgs)
        self.nextBtn = ttk.Button(self, text=">",
                                  width=10,
                                  command=self._advance_frame)
        self.nextBtn.grid(row=1, column=9, sticky="E", **self.padArgs)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)
    def _draw_ruler(self):
        self.canvas.delete("all")
        self.canvas.create_line(self.canMin, self.yZero,
                                self.canMax, self.yZero,
                                width=self.lineWidth)
        self.canvas.create_line(self.canMin,
                                self.yZero + np.round( self.lineWidth / 2),
                                self.canMin,
                                self.yZero - self.barHeight * 2,
                                width=self.lineWidth)
        self.canvas.create_line(self.canMax,
                                self.yZero + np.round( self.lineWidth / 2),
                                self.canMax,
                                self.yZero - self.barHeight * 2,
                                width=self.lineWidth)
        self.canvas.create_line(self.canMin,
                                self.yZero - self.barHeight * 2,
                                self.canMax,
                                self.yZero - self.barHeight * 2,
                                width=1)
    def _world2canvas(self, x):
        l = (x - self.xMin) * self.canRng / self.xRng + self.canMin
        return l
    def _canvas2world(self, l):
        x = (l-self.canMin) * self.xRng / self.canRng + self.xMin
        return x
    def _draw_handle(self, x):
        y = self.yZero - self.barHeight/2.0
        size = self.handleSize
        polygon = (x, y,
                   x, self.yZero - self.barHeight * 2,
                   x, y,
                   x+size, y+size,
                   x+size,  y+size*3.,
                   x-size,  y+size*3.,
                   x-size, y+size,
                   x, y)
        item = self.canvas.create_polygon(polygon, fill="lightblue",
                                          outline="black",
                                          width=2)
        self.canvas.itemconfigure(item, tag=('handle'))
        self._set_label_value(x)
        self.x = x
        return item
    def _create_bindings(self):
        self.canvas.tag_bind('handle', '<1>', self._sel_handle)
        self.canvas.bind('<B1-Motion>', self._drag_handle)
        self.canvas.bind('<Any-ButtonRelease-1>', self._release_handle)
    def _sel_handle(self, evt):
        self.x = self.canvas.canvasx(evt.x)
        self.canvas.addtag_withtag('active', 'current')
        self.canvas.itemconfigure('active',{'fill': 'red', 'stipple': ''})
    def _drag_handle(self, evt):
        if not self.canvas.find_withtag('active'):
            return
        cx = self.canvas.canvasx(evt.x)
        item = self.canvas.find_withtag('active')
        edgeX =self.canvas.coords(item)[0]
        dX = self.x - edgeX
        limLeft = self.canMin
        limRight = self.canMax
        if cx <= limLeft + dX:
            cx = limLeft + dX
        if cx >= limRight + dX:
            cx = limRight + dX
        self.canvas.move('active', cx - self.x, 0)
        self.x = cx
        self.xEdge = edgeX
        self._set_label_value(edgeX)
        if self.selMode:
            self.selRight = self.xEdge
            self._update_selrect()
    def _move_handle(self, deltaX):
        self.canvas.move('handle', deltaX, 0)
    def _release_handle(self, evt):
        if not self.canvas.find_withtag('active'):
            return
        self.canvas.itemconfigure('active',
                                  {'fill': 'lightblue', 'stipple': ''})
        self.canvas.dtag('active')
        self.event_generate(self.releaseEvent)
    def _set_label_value(self, value):
        frm = self._canvas2world(value)
        valFmt = "{:10.0f}".format(frm)
        self.valueHandle.set(valFmt)
        if self.runTime_s is not None:
            timeStamp_s = (frm / self.xMax) * self.runTime_s
            time_m, time_s = divmod(timeStamp_s, 60)
            timeFmt = "{:10.0f}m : {:05.2f}s".format(time_m, time_s)
            self.timeHandle.set(timeFmt)
    def _toggle_selmode(self):
        if self.selModeBtn.config("text")[-1] == u"Select Mode Off \u2610":
            self.selModeBtn.config(text=u"Select Mode On \u2611")
            self.selMode = True
            self.selRight = None
            self.selLeft = self.xEdge
        else:
            self.selModeBtn.config(text=u"Select Mode Off \u2610")
            self.selMode = False
    def _clear_selection(self):
        self.canvas.delete("selrect")
        self.selRight = None
        self.selLeft = self.xEdge
    def _retard_frame(self):
        curFrm = self.valueHandle.get()
        newFrm = max(curFrm - int(self.frameSkip.get()), self.xMin)
        curCan = self._world2canvas(curFrm)
        newCan = self._world2canvas(newFrm)
        skip = newCan - curCan
        self._move_handle(skip)
        self.x = newCan
        self._set_label_value(self.x)
        self.event_generate(self.releaseEvent)
    def _advance_frame(self):
        curFrm = self.valueHandle.get()
        newFrm = min(curFrm + int(self.frameSkip.get()), self.xMax)
        curCan = self._world2canvas(curFrm)
        newCan = self._world2canvas(newFrm)
        skip = newCan - curCan
        self._move_handle(skip)
        self.x = newCan
        self._set_label_value(self.x)
        self.event_generate(self.releaseEvent)
    def _update_selrect(self):
        self.canvas.delete("selrect")
        if not self.selLeft is None and not self.selRight is None:
            x1 = self.selLeft
            x2 = self.selRight
            y1 = self.yZero - self.barHeight
            y2 = self.yZero - self.barHeight * 2
            item = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                width=self.lineWidth,
                                                fill="spring green",
                                                outline="")
            self.canvas.itemconfigure(item, tag=("selrect"))
            self.canvas.tag_lower("selrect")
    def set_range(self, xMin, xMax, runTime_s=None):
        self.xMin = xMin
        self.xMax = xMax
        self.canRng = float(self.canMax - self.canMin)
        self.xRng = float(self.xMax - self.xMin)
        self.runTime_s = runTime_s
    def update_trackmask(self,  maskArr):
        self.canvas.delete("maskline")
        frames = np.nonzero(maskArr)[0].tolist()
        for frm in frames:
            item = self.canvas.create_line(
                self._world2canvas(frm),
                self.yZero - self.barHeight * 2 - int(self.lineWidth * 1.5),
                self._world2canvas(frm),
                self.yZero - self.barHeight * 2,
                width=self.lineWidth,
                fill="magenta")
            self.canvas.itemconfigure(item, tag=("maskline"))
            self.canvas.tag_raise("maskline")
    def update_tracklines(self, frameArr, maskArr):
        self.canvas.delete("trackline")
        frames = frameArr[np.nonzero(maskArr)].tolist()
        for frm in frames:
            item = self.canvas.create_line(self._world2canvas(frm),
                                    self.yZero + np.round( self.lineWidth / 2),
                                    self._world2canvas(frm),
                                    self.yZero - self.barHeight,
                                    width=self.lineWidth,
                                           fill="magenta")
            self.canvas.itemconfigure(item, tag=("trackline"))
            self.canvas.tag_lower("trackline")
    def update_editlines(self, frameArr, maskArr):
        self.canvas.delete("editline")
        frames = frameArr[np.nonzero(maskArr)].tolist()
        for frm in frames:
            item = self.canvas.create_line(self._world2canvas(frm),
                                           self.yZero - self.barHeight,
                                           self._world2canvas(frm),
                                           self.yZero - self.barHeight * 2,
                                           width=self.lineWidth,
                                           fill="blue")
            self.canvas.itemconfigure(item, tag=("editline"))
    def get_selrange(self):
        if self.selLeft is None or self.selRight is None:
            return 0, 0
        else:
            x1 = max(int(self._canvas2world(self.selLeft)), self.xMin)
            x2 = min(int(self._canvas2world(self.selRight)), self.xMax)
        xRng = [x1, x2]
        xRng.sort()
        return xRng
