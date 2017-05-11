import numpy as np
from scipy.spatial.distance import cdist
import argparse
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import point_utils
from diditracklet import *
from generate_tracklet import *

parser = argparse.ArgumentParser(description='Refine tracklets by finding pose of reference object or smoothing trajectory')
parser.add_argument('-1', '--first', type=int, action='store', help='Do one frame only, e.g. -1 87 (does frame 87)')
parser.add_argument('-s', '--start-refining-from', type=int, action='store', default=0, help='Start from frame (defaults to 0)')
parser.add_argument('-l', '--only-do-look-backs', action='store_true', help='Only search based on previous frame position (needs -s)')

#parser.add_argument('-s', '--search-yaw', action='store_true', help='Search for yaw')
parser.add_argument('-r', '--reference', type=str, action = 'store', help='First point cloud file name')
parser.add_argument('-m', '--flip', action = 'store_true', help='Flip reference object for alignment')
parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
                    help='Input folder where processed tracklet subdirectories are located')
parser.add_argument('-f', '--filter', type=str, nargs='+', default=None,
                    help='Only include date/drive tracklet subdirectories, e.g. -f 1/21_f 2/24')
parser.add_argument('-y', '--yaw', type=float, nargs='?', default=0.,
                    help='Force initial yaw correction (e.g. -y 0.88)')
parser.add_argument('-xi', '--input-xml-filename', type=str, nargs='?', default='tracklet_labels.xml',
                    help='input tracklet xml filename (defaults to tracklet_labels.xml)')
parser.add_argument('-xo', '--output-xml-filename', type=str, nargs='?', default='tracklet_labels_refined.xml',
                    help='output tracklet xml filename (defaults to tracklet_labels.xml)')
parser.add_argument('-d', '--dump', action='store_true', help='Print csv or x,y,z translations and does not ')
parser.add_argument('-n', '--no-refine', action='store_true', help='Do not attempt to fit reference vehicle')

args = parser.parse_args()
#search_yaw =  args.search_yaw

diditracklets = find_tracklets(args.indir,
                               filter=args.filter,
                               yaw_correction=args.yaw,
                               xml_filename=args.input_xml_filename,
                               flip=args.flip)

for tracklet in diditracklets:

    print("Refining " + tracklet.xml_path)
    print("")
    frames = tracklet.frames() if args.first is None else [args.first]

    t_boxes = []
    t_box = np.zeros(3)
    for frame in frames:
        print("Frame: " + str(frame) + " / " + str(len(frames)))
        if args.no_refine or frame < args.start_refining_from:
            t_box = np.zeros(3)
        else:
            if args.start_refining_from > 0 and args.only_do_look_backs:
                look_back_last_refined_centroid = T + t_box
            else:
                look_back_last_refined_centroid = None
            t_box = tracklet.refine_box(frame, look_back_last_refined_centroid = look_back_last_refined_centroid)
        t_boxes.append(t_box)
        print("")
        T, _ = tracklet.get_box_TR(frame)

    # WRITING TRACKLET

    collection = TrackletCollection()
    h, w, l = tracklet.get_box_size()
    obs_tracklet = Tracklet(object_type='Car', l=l,w=w,h=h, first_frame=frames[0])

    for frame, t_box in zip(frames, t_boxes):
        pose = tracklet.get_box_pose(frame)
        pose['tx'] += t_box[0]
        pose['ty'] += t_box[1]
        pose['tz'] += t_box[2]
        obs_tracklet.poses.append(pose)
        if args.dump:
            print(str(pose['tx']) + "," + str(pose['ty']) + ","+ str(pose['tz']))
    collection.tracklets.append(obs_tracklet)
        # end for obs_topic loop

    tracklet_path = os.path.join(tracklet.xml_path , args.output_xml_filename)
    collection.write_xml(tracklet_path)

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
