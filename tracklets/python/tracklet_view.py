import argparse
import numpy as np
import os
import sys
import re
from diditracklet import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore


def find_tracklets(
        directory,
        filter = None,
        pattern="tracklet_labels.xml"):
    diditracklets = []
    combined_filter = "(" + ")|(".join(filter) + "$)" if filter is not None else None
    if combined_filter is not None:
        combined_filter = combined_filter.replace("*", ".*")

    for root, dirs, files in os.walk(directory):
        for date in dirs: # 1 2 3
            for _root, drives, files in os.walk(os.path.join(root, date)): # ./1/ ./18/ ...
                for drive in drives:
                    if os.path.isfile(os.path.join(_root, drive, pattern)):
                        if filter is None or re.match(combined_filter, date + '/' + drive):
                            diditracklet = DidiTracklet(root, date, drive)
                            diditracklets.append(diditracklet)

    return diditracklets

if __name__ == '__main__':

    app = QtGui.QApplication([])

    ## Define a top-level widget to hold everything
    w = QtGui.QWidget()

    ## Create some widgets to be placed inside

    tv_button  = QtGui.QPushButton('Toggle View')
    im         = pg.image(title="Loading")

    ## Create a grid layout to manage the widgets size and position
    layout = QtGui.QGridLayout()
    w.setLayout(layout)

    ## Add widgets to the layout in their proper positions
    layout.addWidget(tv_button, 0, 0)  # button goes in upper-left
    layout.addWidget(im, 1, 0)  # plot goes on right side, spanning 3 rows

    def toggle_view():
        Thread.side_view = True if Thread.side_view == False else False

    def update(data):
        (tv, title) = data
        im.setImage(tv)
        im.win.setWindowTitle(title)

    class Thread(pg.QtCore.QThread):
        new_image = pg.QtCore.Signal(object)
        side_view = True

        def run(self):
            parser = argparse.ArgumentParser(description='View tracklets.')
            parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
                                help='Input folder where processed tracklet subdirectories are located')
            parser.add_argument('-f', '--filter', type=str, nargs='+', default=None,
                                help='Only include date/drive tracklet subdirectories, e.g. -f 1/21_f 2/24')
            args = parser.parse_args()

            diditracklets = find_tracklets(args.indir, args.filter)

            for tracklet in diditracklets:
                tvv = None

                for frame in tracklet.frames():

                    tv = tracklet.top_and_side_view(frame, with_boxes=True, zoom_to_box=True, SX=400)
                    if tvv is None:
                        tvv = np.expand_dims(tv, axis=0)
                    else:
                        tvv = np.concatenate((tvv, np.expand_dims(tv, axis=0)), axis=0)
                self.new_image.emit((tvv, tracklet.date + "/" + tracklet.drive + ".bag"))
            print("Finished!")


    w.show()
    tv_button.clicked.connect(toggle_view)

    thread = Thread()
    thread.new_image.connect(update)
    thread.start()





    import sys

    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()