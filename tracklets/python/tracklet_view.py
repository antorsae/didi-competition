import argparse
import numpy as np

from diditracklet import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

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
            parser.add_argument('-i', '--indir', type=str, default='../../../../didi-data/release2/Data-points-processed',
                                help='Input folder where processed tracklet subdirectories are located')
            parser.add_argument('-f', '--filter', type=str, nargs='+', default=None,
                                help='Only include date/drive tracklet subdirectories, e.g. -f 1/21_f 2/24')
            parser.add_argument('-y', '--yaw', type=float, default=0.,
                                help='Force initial yaw correction (e.g. -y 0.88)')
            parser.add_argument('-xi', '--xml-filename', type=str, default='tracklet_labels.xml',
                                help='tracklet xml filename (defaults to tracklet_labels.xml, TIP: use tracklet_labels_trainable.xml if available)')
            parser.add_argument('-z', '--zoom-to-box', action='store_true',
                                help='zoom view to bounding box')
            parser.add_argument('-ra', '--randomize', action='store_true',
                                help='random perturbation (augmentation)')
            parser.add_argument('-1', '--first', type=int, action='store',
                                help='View one frame only, e.g. -1 87 (views frame 87)')
            parser.add_argument('-m', '--many', type=int, action='store',
                                help='How many frames to view, e.g. -m 100 (views up to 100 frames)')
            parser.add_argument('-n', '--num-points', type=int, action='store',
                                help='Resample to number of points, e.g. -n 27000')
            parser.add_argument('-d', '--distance', default=50., type=float, action='store',
                                help='Distance ')
            parser.add_argument('-p', '--points-per-ring', default=None, type=int, action='store',
                                help='If specified, points per ring for linear interpolation')
            parser.add_argument('-r', '--rings', nargs='+', type=int, action='store', help='Only include rings, e.g. -or 10 11 12 13')
            parser.add_argument('-sw', '--scale-w', default=1., type=float, action='store', help='Scale bounding box width ')
            parser.add_argument('-sl', '--scale-l', default=1., type=float, action='store', help='Scale bounding box width ')
            parser.add_argument('-sh', '--scale-h', default=1., type=float, action='store', help='Scale bounding box width ')

            args = parser.parse_args()

            diditracklets = find_tracklets(args.indir, args.filter, args.yaw, args.xml_filename, False, (args.scale_h, args.scale_w, args.scale_l))

            for tracklet in diditracklets:
                tvv = None

                _frames = tracklet.frames()
                _first  = _frames[0] if args.first is None else args.first
                _many = None
                if args.first and (args.many is None):
                    _many = 1
                elif args.many is not None:
                    _many = args.many
                if _many is not None:
                    frames = [f for f in _frames if f in range(_first, _first+_many)]
                else:
                    frames = _frames

                print("Loading: " + str(len(frames)) + " / " + str(len(_frames)) + " frames")

                for frame in frames:
                    tv = tracklet.top_view(frame,
                                           with_boxes=True,
                                           zoom_to_box=args.zoom_to_box,
                                           SX=400,
                                           randomize=args.randomize,
                                           distance = args.distance,
                                           rings = range(args.rings[0], args.rings[1]) if args.rings else None,
                                           num_points = args.num_points,
                                           points_per_ring = args.points_per_ring)

                    #obs_points = tracklet.get_points_in_box(frame, ignore_z=False)
                    #print('frame ' + str(frame), obs_points)

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
