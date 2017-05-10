import numpy as np
from scipy.spatial.distance import cdist
import argparse
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import point_utils
from diditracklet import *

parser = argparse.ArgumentParser(description='Find point cloud translations')
parser.add_argument('-1', '--first', type=int, action='store', help='Do one frame only, e.g. -1 87 (does frame 87)')
#parser.add_argument('-s', '--search-yaw', action='store_true', help='Search for yaw')
parser.add_argument('-r', '--reference', type=str, action = 'store', help='First point cloud file name')
#parser.add_argument('-m', '--flip', action = 'store_true', help='Flip reference object for alignment')
parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
                    help='Input folder where processed tracklet subdirectories are located')
parser.add_argument('-f', '--filter', type=str, nargs='+', default=None,
                    help='Only include date/drive tracklet subdirectories, e.g. -f 1/21_f 2/24')
parser.add_argument('-y', '--yaw', type=float, nargs='?', default=0.,
                    help='Force initial yaw correction (e.g. -y 0.88)')

args = parser.parse_args()
search_yaw =  args.search_yaw

diditracklets = find_tracklets(args.indir, filter=args.filter, yaw_correction=args.yaw)

for tracklet in diditracklets:

    frames = tracklet.frames if args.first is None else [args.first]

    for frame in frames:
        print("Frame: " + str(frame) + " / " + str(len(frames)))
        tv = tracklet.refine_box(frame)
        print("")

'''
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

size=np.concatenate((
    0.01 * np.ones(reference.shape[0]),
    0.05 * np.ones(first_aligned.shape[0]),
    0.05 * np.ones(first.shape[0])), axis = 0)

print(size.shape)

sp1 = gl.GLScatterPlotItem(pos=np.concatenate((reference[:,0:3], first_aligned, first), axis=0),
                           size=size,
                           color=np.concatenate((
                               np.tile(np.array([0,0,1.,0.5]), (reference.shape[0],1)),
                               np.tile(np.array([1.,1.,1.,0.8]), (first_aligned.shape[0],1)),
                               np.tile(np.array([1., 0., 0., 0.8]), (first.shape[0], 1))
                           ), axis = 0),
                           pxMode=False)
sp1.translate(5,5,0)
w.addItem(sp1)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
'''
