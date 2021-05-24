#=============================================================================#
#                                                                             #
# MODIFIED: 14-Sep-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

from keras.callbacks import BaseLogger
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import os

#-----------------------------------------------------------------------------#
class TrainingMonitor(BaseLogger):
    """Class to log the loss/accuracy as a plot and to a JSON file."""

    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()   # Shortcut to BaseLogger init
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        """Load prior training history from a JSON file."""

        self.H = {}   # History dictionary

        # If the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # Trim entries beyond specified starting epoch
                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        """At the end of each epoch, dump history to disk and update plot."""

        # Parse the log items into the history dictionary
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # Dump the history dictionary to disk
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()

        # Create plot if at least two epochs have passed (zero-ref)
        if len(self.H["loss"]) > 1:
            mpl.rcParams["font.size"] = 12.0
            fig = plt.figure(figsize=(14., 6.))
            ax1 = fig.add_subplot(1,2,1)
            epoch = range(1, len(self.H["loss"])+1)
            ax1.step(epoch, self.H["loss"], where="mid", label="Train Loss")
            ax1.step(epoch, self.H["val_loss"], where="mid", label="Valid Loss")
            ax1.legend(loc="upper right", shadow=False, fontsize="medium")
            ax1.set_title("Model Loss [Epoch {:d}]".format(epoch[-1]))
            ax1.set_ylabel("Loss")
            ax1.set_xlabel("Epoch")
            ax2 = fig.add_subplot(1,2,2)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.step(epoch, self.H["acc"], where="mid", label="Train Accuracy")
            ax2.step(epoch, self.H["val_acc"], where="mid",
                     label="Valid Accuracy")
            ax2.legend(loc="lower right", shadow=False, fontsize="medium")
            ax2.set_title("Model Accuracy [Epoch {:d}]".format(epoch[-1]))
            ax2.set_ylabel("Accuracy")
            ax2.set_xlabel("Epoch")

            # Nice formatting
            ax1.tick_params(pad=7)
            for line in ax1.get_xticklines() + ax1.get_yticklines():
                line.set_markeredgewidth(1)
                ax2.tick_params(pad=7)
            for line in ax2.get_xticklines() + ax2.get_yticklines():
                line.set_markeredgewidth(1)
            plt.tight_layout()
            plt.savefig(self.figPath)
            plt.close()
