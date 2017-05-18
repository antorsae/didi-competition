import numpy as np
import os
from scipy.spatial.distance import cdist
import argparse
from diditracklet import *
from generate_tracklet import *
from sklearn import linear_model
from sklearn.svm import SVR

parser = argparse.ArgumentParser(description='Refine tracklets by finding pose of reference object or smoothing trajectory')
parser.add_argument('-1', '--first', type=int, action='store', help='Do one frame only, e.g. -1 87 (does frame 87)')
parser.add_argument('-s', '--start-refining-from', type=int, action='store', nargs=1, default=0, help='Start from frame (defaults to 0)')
parser.add_argument('-l', '--only-do-look-backs', action='store_true', help='Only search based on previous frame position (needs -s)')

parser.add_argument('-m', '--flip', action = 'store_true', help='Flip reference object for alignment')
parser.add_argument('-i', '--indir', type=str, default='../../../../release2/Data-points-processed',
                    help='Input folder where processed tracklet subdirectories are located')
parser.add_argument('-f', '--filter', type=str, nargs='+', default=None,
                    help='Only include date/drive tracklet subdirectories, e.g. -f 1/21_f 2/24')
parser.add_argument('-y', '--yaw', type=float, default=0.,
                    help='Force initial yaw correction (e.g. -y 0.88)')
parser.add_argument('-xi', '--input-xml-filename', type=str, default='tracklet_labels.xml',
                    help='input tracklet xml filename (defaults to tracklet_labels.xml)')
parser.add_argument('-xo', '--output-xml-filename', type=str,
                    help='output tracklet xml filename')
parser.add_argument('-d', '--dump', action='store_true', help='Print csv or x,y,z translations and does not ')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-n', '--no-refine', action='store_true', help='Do not attempt to fit reference vehicle')
group.add_argument('-r', '--ransac', action='store_true', help='Use ransac for trajectory interpolation')
group.add_argument('-a', '--align', action='store_true', help='Use 3d pose alignment')

parser.add_argument('-ap', '--align-percentage', type=float, action='store', default=0.6, help='Min percentage of lidar points for alignment')
parser.add_argument('-ad', '--align-distance', type=float, action='store', default=0.3, help='Threshold distance for a point to be considered inlier during alignment')
parser.add_argument('-as', '--align-yaw', action='store_true', help='Search for yaw during alignment')

parser.add_argument('-v', '--view', action='store_true', help='View in 3d')

args = parser.parse_args()

diditracklets = find_tracklets(args.indir,
                               filter=args.filter,
                               yaw_correction=args.yaw,
                               xml_filename=args.input_xml_filename,
                               flip=args.flip)

if args.output_xml_filename is None and args.no_refine is False:
    print("----------------------------------------------------------------------------------------")
    print("WARNING: no -xo or --output-xml-filename filename provided, tracklets will NOT be saved!")
    print("----------------------------------------------------------------------------------------")
    print("")

for tracklet in diditracklets:

    print("Refining " + tracklet.xml_path)
    print("")
    frames = tracklet.frames() if args.first is None else [args.first]

    t_boxes  = []
    t_states = np.ones(len(frames), dtype=np.int32)

    t_box = np.zeros(3)
    if args.ransac:

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        # Fit lines for each axis using all data
        y = []
        for frame in frames:
            y.append(list(tracklet.get_box_centroid(frame)[:3]))

        X = np.arange(0,len(y))
        x_axis = np.array(y)[:,0]
        y_axis = np.array(y)[:,1]
        z_axis = np.array(y)[:,2]
        x_d1 = np.diff(x_axis)
        x_d2 = np.diff(x_axis, n=2)

        X =  np.expand_dims(X, axis=1)
        model_x = linear_model.LinearRegression()
        model_x = SVR(kernel='rbf', C=1e4, gamma=0.01)
        model_x.fit(X, x_axis)


        model_y = linear_model.LinearRegression()
        model_y = SVR(kernel='rbf', C=1e4, gamma=0.01)
        model_y.fit(X, y_axis)

        model_z = linear_model.LinearRegression()
        model_z = SVR(kernel='rbf', C=1e4, gamma=0.01)
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
        print("List cloud points differing from estimated points more than 50cm ")
        #x_axis = np.expand_dims(x_axis, axis=1)
        x_axis_diff_points = np.where(abs(x_pred - x_axis) >= 0.5)
        y_axis_diff_points = np.where(abs(y_pred - y_axis) >= 0.5)
        z_axis_diff_points = np.where(abs(z_pred - z_axis) >= 0.5)
        print(x_axis_diff_points)
        print(y_axis_diff_points)
        print(z_axis_diff_points)

        lw = 2
        plt.scatter(X,x_axis, color='black', marker='x', label='centroids x')
        plt.scatter(line_X[x_axis_diff_points],x_axis[x_axis_diff_points], color='red', marker='x')

        plt.plot(line_X, x_pred, color='green', linestyle='-', linewidth=lw, label='SVM Regressor x. Outliers: ' +  str(x_axis_diff_points))


        plt.scatter(X,y_axis, color='black', marker='.', label='centroids y')
        plt.scatter(line_X[y_axis_diff_points],y_axis[y_axis_diff_points], color='red', marker='.')

        plt.plot(line_X, y_pred, color='navy', linestyle='-', linewidth=lw, label='SVM Regressor y. Outliers: ' +  str(y_axis_diff_points))

        plt.scatter(X,z_axis, color='black', marker='*', label='centroids z')
        plt.scatter(line_X[z_axis_diff_points],z_axis[z_axis_diff_points], color='red', marker='*')

        plt.plot(line_X, z_pred, color='pink', linestyle='-', linewidth=lw, label='SVM Regressor z. Outliers: ' +  str(z_axis_diff_points))

        plt.legend(loc=0, fontsize='xx-small')
        plt.savefig(os.path.join(tracklet.xml_path , "plot.png"))
        plt.clf()

        plt.scatter(X[1:],x_d1, color='black', marker='x', label='d x')
        plt.scatter(X[2:],x_d2, color='red', marker='*', label='d2 x')

        plt.savefig(os.path.join(tracklet.xml_path , "plotdiff.png"))
        plt.clf()

        #modify poses using predicted values --> not accurate
        #x_axis[x_axis_diff_points] = x_pred[x_axis_diff_points]
        #y_axis[y_axis_diff_points] = y_pred[y_axis_diff_points]
        #z_axis[z_axis_diff_points] = z_pred[z_axis_diff_points]

        #modify poses for outliers using the neighbours mean value in all axis
        x_axis[x_axis_diff_points] = (x_axis[np.array(x_axis_diff_points) -1 ] + x_axis[np.array(x_axis_diff_points) +1 ]) / 2
        y_axis[x_axis_diff_points] = (y_axis[np.array(x_axis_diff_points) -1 ] + y_axis[np.array(x_axis_diff_points) +1 ]) / 2
        z_axis[x_axis_diff_points] = (z_axis[np.array(x_axis_diff_points) -1 ] + z_axis[np.array(x_axis_diff_points) +1 ]) / 2

        y_axis[y_axis_diff_points] = (y_axis[np.array(y_axis_diff_points) -1 ] + y_axis[np.array(y_axis_diff_points) +1 ]) / 2
        x_axis[y_axis_diff_points] = (x_axis[np.array(y_axis_diff_points) -1 ] + x_axis[np.array(y_axis_diff_points) +1 ]) / 2
        z_axis[y_axis_diff_points] = (z_axis[np.array(y_axis_diff_points) -1 ] + z_axis[np.array(y_axis_diff_points) +1 ]) / 2

        #z_axis[z_axis_diff_points] = (z_axis[np.array(z_axis_diff_points) -1 ] + z_axis[np.array(z_axis_diff_points) +1 ]) / 2
        #y_axis[z_axis_diff_points] = (y_axis[np.array(z_axis_diff_points) -1 ] + y_axis[np.array(z_axis_diff_points) +1 ]) / 2
        #x_axis[z_axis_diff_points] = (x_axis[np.array(z_axis_diff_points) -1 ] + x_axis[np.array(z_axis_diff_points) +1 ]) / 2

        t_boxes = zip(x_axis - np.array(y)[:,0],y_axis - np.array(y)[:,1], z_axis - np.array(y)[:,2])

        t_states[np.array(x_axis_diff_points)] = 0
        t_states[np.array(y_axis_diff_points)] = 0

    elif args.align:

        for frame in frames:
            print("Frame: " + str(frame) + " / " + str(len(frames)))
            if args.no_refine or frame < args.start_refining_from:
                t_box = np.zeros(3)
            else:
                if (args.start_refining_from > 0) and args.only_do_look_backs:
                    look_back_last_refined_centroid = T + t_box
                else:
                    look_back_last_refined_centroid = None

            t_box, yaw_box, reference, first = tracklet.refine_box(frame,
                                                                   look_back_last_refined_centroid = look_back_last_refined_centroid,
                                                                   return_aligned_clouds=True,
                                                                   min_percent_first = args.align_percentage,
                                                                   threshold_distance = args.align_distance,
                                                                   search_yaw = args.align_yaw )

            yaw = tracklet.get_yaw(frame)

            t_boxes.append(t_box)
            print("")
            T, _ = tracklet.get_box_TR(frame)

    # WRITING TRACKLET
    if args.output_xml_filename is not None:

        collection = TrackletCollection()
        h, w, l = tracklet.get_box_size()
        obs_tracklet = Tracklet(object_type='Car', l=l,w=w,h=h, first_frame=frames[0])

        for frame, t_box, t_state in zip(frames, t_boxes, t_states):
            pose = tracklet.get_box_pose(frame)
            pose['tx'] += t_box[0]
            pose['ty'] += t_box[1]
            pose['tz'] += t_box[2]
            pose['status'] = t_state
            obs_tracklet.poses.append(pose)
            if args.dump:
                print(str(pose['tx']) + "," + str(pose['ty']) + ","+ str(pose['tz']))

        collection.tracklets.append(obs_tracklet)
        # end for obs_topic loop

        tracklet_path = os.path.join(tracklet.xml_path , args.output_xml_filename)
        collection.write_xml(tracklet_path)

if args.view:

    first_aligned = point_utils.rotZ(first, yaw_box) - point_utils.rotZ(np.array([t_box[0], t_box[1], 0.]), -yaw)

    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('Reference vehicle (blue) vs. original (red) vs. aligned (white) obstacle')

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
