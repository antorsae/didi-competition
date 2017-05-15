import numpy as np
import os
from scipy.spatial.distance import cdist
import argparse
#from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph.opengl as gl

from diditracklet import *
from generate_tracklet import *
from sklearn import linear_model
from sklearn.svm import SVR
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Refine tracklets by finding pose of reference object or smoothing trajectory')
parser.add_argument('-1', '--first', type=int, action='store', help='Do one frame only, e.g. -1 87 (does frame 87)')
parser.add_argument('-s', '--start-refining-from', type=int, action='store', default=0, help='Start from frame (defaults to 0)')
parser.add_argument('-l', '--only-do-look-backs', action='store_true', help='Only search based on previous frame position (needs -s)')

#parser.add_argument('-s', '--search-yaw', action='store_true', help='Search for yaw')
#parser.add_argument('-r', '--reference', type=str, action = 'store', help='First point cloud file name')
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
parser.add_argument('-r', '--ransac', action='store_true', help='Use ransac for trajectory interpolation')

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
    if args.ransac:


        # Fit lines for each axis using all data
        y = []
        for frame in frames:
            y.append(list(tracklet.get_box_centroid(frame)[:3]))


        X = np.arange(0,len(y))
        x_axis = np.array(y)[:,0]
        y_axis = np.array(y)[:,1]
        z_axis = np.array(y)[:,2]


        X =  np.expand_dims(X, axis=1)
        model_x = linear_model.LinearRegression()
        model_x = SVR(kernel='rbf', C=1e3, gamma=0.1)
        model_x.fit(X, x_axis)

        model_y = linear_model.LinearRegression()
        model_y = SVR(kernel='rbf', C=1e3, gamma=0.1)
        model_y.fit(X, y_axis)

        model_z = linear_model.LinearRegression()
        model_z = SVR(kernel='rbf', C=1e3, gamma=0.1)
        model_z.fit(X, z_axis)

        # Robustly fit linear model with RANSAC algorithm
        #model_ransac = linear_model.RANSACRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1))
        #model_ransac.fit(X, x_axis)
        #inlier_mask = model_ransac.inlier_mask_
        #outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(0, len(y))
        x_pred = model_x.predict(line_X[:, np.newaxis])
        y_pred = model_y.predict(line_X[:, np.newaxis])
        z_pred = model_z.predict(line_X[:, np.newaxis])
        #line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])




        # Compare cloud with estimated points
        print("List cloud points differing from estimated points more than 1m ")
        #x_axis = np.expand_dims(x_axis, axis=1)
        x_axis_diff_points = np.where(abs(x_pred - x_axis) >= 1)
        y_axis_diff_points = np.where(abs(y_pred - y_axis) >= 1)
        z_axis_diff_points = np.where(abs(z_pred - z_axis) >= 1)
        print(x_axis_diff_points)
        print(y_axis_diff_points)
        print(z_axis_diff_points)



        lw = 2


        plt.scatter(X,x_axis, color='black', marker='x',
                 label='centroids x')
        plt.scatter(line_X[x_axis_diff_points],x_axis[x_axis_diff_points], color='red', marker='x')

        plt.plot(line_X, x_pred, color='green', linestyle='-', linewidth=lw,
                 label='SVM Regressor x. Outliers: ' +  str(x_axis_diff_points))

        plt.scatter(X,y_axis, color='black', marker='.',
                    label='centroids y')
        plt.scatter(line_X[y_axis_diff_points],y_axis[y_axis_diff_points], color='red', marker='.')

        plt.plot(line_X, y_pred, color='navy', linestyle='-', linewidth=lw,
                 label='SVM Regressor y. Outliers: ' +  str(y_axis_diff_points))

        plt.scatter(X,z_axis, color='black', marker='*',
                    label='centroids z')
        plt.scatter(line_X[z_axis_diff_points],z_axis[z_axis_diff_points], color='red', marker='*')

        plt.plot(line_X, z_pred, color='pink', linestyle='-', linewidth=lw,
                 label='SVM Regressor z. Outliers: ' +  str(z_axis_diff_points))


        plt.legend(loc=0, fontsize='xx-small')
        plt.savefig(tracklet.xml_path + ".png")
        plt.clf()


        #create new poses
        x_axis[x_axis_diff_points] = x_pred[x_axis_diff_points]
        y_axis[y_axis_diff_points] = y_pred[y_axis_diff_points]
        z_axis[z_axis_diff_points] = z_pred[z_axis_diff_points]

        t_boxes = zip(x_axis,y_axis,z_axis)



    else:
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
        if args.ransac:
            pose['tx'] = t_box[0]
            pose['ty'] = t_box[1]
            pose['tz'] = t_box[2]
        else:
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



