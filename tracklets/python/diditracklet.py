#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import numpy as np
import os
import sys
import parse_tracklet as tracklets
import didipykitti as pykitti
from collections import defaultdict
import cv2
import point_utils
from scipy.linalg import expm3, norm

M = 10
MIN_HEIGHT = -2.  # from camera (i.e.  -2-1.65 =  3.65m above floor)
MAX_HEIGHT = 2.  # from camera (i.e.  +2-1.65 =  0.35m below floor)
M_HEIGHT = (MAX_HEIGHT - MIN_HEIGHT) / M
MIN_X = -40.
MAX_X = 40.
MIN_Z = 5.
MAX_Z = 70.
HEIGHT_F_RES = 0.1  # 0.1m for x,z slicing

C_W = 512
C_H = 64

CAMERA_FEATURES = (375, 1242, 3)
HEIGHT_FEATURES = (int((MAX_Z - MIN_Z) / HEIGHT_F_RES), int((MAX_X - MIN_X) / HEIGHT_F_RES), M + 2)
F_VIEW_FEATURES = (C_H, C_W, 3)


class DidiTracklet(object):
    kitti_cat_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc', 'Person (sitting)']
    kitti_cat_idxs = range(1, 1 + len(kitti_cat_names))

    LIDAR_ANGLE = np.pi / 6.

    def __init__(self, basedir, date, drive):
        self.basedir = basedir
        self.date    = date
        self.drive   = drive

        self.kitti_data = pykitti.raw(basedir, date, drive,
                                      range(0, 1))  # , range(start_frame, start_frame + total_frames))
        self.tracklet_data = tracklets.parse_xml(os.path.join(basedir, date, drive, "tracklet_labels.xml"))
        self.kitti_data.load_calib()  # Calibration data are accessible as named tuples

        # lidars is a dict indexed by frame: e.g. lidars[10] = np(N,4)
        self.lidars = {}

        # images is a dict indexed by frame: e.g. lidars[10] = np(SY,SX,3)
        self.images = {}
        self.im_dim = (1242, 375)  # by default

        # boxes is a dict indexed by frame:  e.g. boxes[10] = [box, box, ...]
        self._boxes = None  # defaultdict(list)

        reference_file = os.path.join(basedir, date, 'obs.txt')
        if os.path.isfile(reference_file):
            self.reference = self.__load_reference(reference_file)
        else:
            self.reference = None

    def __load_reference(self, reference_file):
        reference = np.genfromtxt(reference_file, dtype=np.float32, comments='/')
        reference = np.multiply(reference, np.array([0.0254, 0.0254, 0.0254, 1, 1, 1]))
        reference_min = np.amin(reference[:, 0:3], axis=0)
        reference_lwh = np.amax(reference[:, 0:3], axis=0) - reference_min

        reference[:, 0:3] -= (reference_min[0:3] + reference_lwh[0:3] / 2.)

        reference[:, 0:3] = point_utils.rotate(reference[:,0:3], np.array([1., 0., 0.]), np.pi / 2)

        reference[:, 2] = -reference[:, 2]
        reference[:, 2] -= (np.amin(reference[:, 2]))
        return reference

    def align(self, first):
        if self.reference is not None:
            T, _ = point_utils.icp(first, self.reference[:, 0:3])
            t    =  T[0:3, 3].T
        else:
            t    = np.zeros((3))
        return t

    # for DIDI -> don't filter anything
    # for KITTI ->
    # include tracklet IFF in image and not occluded
    # WARNING: There's a lot of tracklets with occs=-1 (255) which we need to fix
    def __include_tracklet(self, t, idx):
        return True # (t.truncs[idx] == tracklets.Truncation.IN_IMAGE) and (t.occs[idx, 0] == 0)

    def _get_yaw(self, frame):
        assert len(self.tracklet_data) == 1 # only one tracklet supported for now!
        for t in self.tracklet_data:
            assert frame in range(t.first_frame, t.first_frame + t.num_frames)
            idx = frame - t.first_frame
            yaw = t.rots[idx][2]
        return yaw


    # return list of frames with tracked objects of type only_with
    def frames(self, only_with=None):
        frames = []
        for t in self.tracklet_data:
            if (only_with is None) or (t.object_type in only_with):
                for frame_offset in range(t.first_frame, t.first_frame + t.num_frames):
                    if self.__include_tracklet(t, frame_offset - t.first_frame):
                        frames.append(frame_offset)
            else:
                print("UNTRACKED", t.object_type)
        self._init_boxes(only_with)
        return list(set(frames))  # remove duplicates

    def _read_lidar(self, frame):
        if frame not in self.kitti_data.frame_range:
            self.kitti_data = pykitti.raw(self.basedir, self.date, self.drive,
                                          range(frame, frame + 1))  # , range(start_frame, start_frame + total_frames))
            self.kitti_data.load_calib()
        assert frame in self.kitti_data.frame_range
        self.kitti_data.load_velo()  # Each scan is a Nx4 array of [x,y,z,reflectance]
        assert len(self.kitti_data.velo) == 1
        lidar = self.kitti_data.velo[0]
        ax = 0.0 * np.pi/180.
        ay = 0.0 * np.pi/180.
        from math import sin, cos
        rx  = np.array([[1., 0., 0.], [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]], dtype=np.float32)
        ry  = np.array([[cos(ay), 0., sin(ay)], [0, 1., 0.], [-sin(ay), 0, cos(ay)]], dtype=np.float32)

        rc = np.eye(lidar.shape[1])
        rc[0:3,0:3] = rx
        lidar = np.dot(lidar, rc)
        rc[0:3,0:3] = ry
        lidar = np.dot(lidar, rc)
        self.lidars[frame] = lidar
        return

    def _read_image(self, frame):
        if frame not in self.kitti_data.frame_range:
            self.kitti_data = pykitti.raw(self.basedir, self.date, self.drive,
                                          range(frame, frame + 1))  # , range(start_frame, start_frame + total_frames))
            self.kitti_data.load_calib()
        assert frame in self.kitti_data.frame_range
        self.kitti_data.load_rgb()
        self.images[frame] = self.kitti_data.rgb[0].left

        (sx, sy) = self.images[frame].shape[::-1][1:]

        if self.im_dim != (sx, sy):
            print("WARNING changing default dimensions to", (sx, sy))
            self.im_dim = (sx, sy)

        return

    # initialize self.boxes with a dict containing frame -> [box, box, ...]
    def _init_boxes(self, only_with):
        assert self._boxes is None
        self._boxes = defaultdict(list)
        for t in self.tracklet_data:
            if (only_with is None) or (t.object_type in only_with):
                for frame_offset in range(t.first_frame, t.first_frame + t.num_frames):
                    idx = frame_offset - t.first_frame
                    if self.__include_tracklet(t, idx):
                        h, w, l = t.size
                        assert (h > 0.) and (w > 0.) and (l > 0.)
                        # in velo:
                        # A       D
                        #
                        # B       C
                        trackletBox = np.array(
                            [  # in velodyne coordinates around zero point and without orientation yet\
                                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
                                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
                                [0.0, 0.0, 0.0, 0.0, h, h, h, h]])
                        yaw = t.rots[idx][2]  # other rotations are 0 in all xml files I checked

                        # IMPORTANT -> ignoring orientation so that we get ordered coordinates w/o orientation
                        #yaw = np.fmod(yaw, np.pi)
                        #if yaw >= np.pi / 2.:
                        #    yaw -= np.pi
                        #elif yaw <= -np.pi / 2.:
                        #    yaw += np.pi
                        #assert (yaw <= np.pi / 2) and (yaw >= -np.pi / 2)

                        assert np.abs(t.rots[idx][:2]).sum() == 0, 'object rotations other than yaw given!'
                        rotMat = np.array([
                            [np.cos(yaw), -np.sin(yaw), 0.0],
                            [np.sin(yaw), np.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0]])
                        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(t.trans[idx], (8, 1)).T
                        self._boxes[frame_offset].append(cornerPosInVelo)
        return

    # given lidar points, subsample POINTS by removing points from voxels with highest density
    def _lidar_subsample(self, lidar, POINTS):
        # X_RANGE = (  0., 70.)
        # Y_RANGE = (-40., 40.)
        # Z_RANGE = ( -2.,  2.)
        # RES = 0.2 (not needed)

        NX = 10
        NY = 10
        NZ = 4

        bins, edges = np.histogramdd(lidar[:, 0:3], bins=(NX, NY, NZ))

        bin_target = np.array(bins, dtype=np.int32)
        # inefficient but effective, TODO optimize (easy)
        for i in range(np.sum(bin_target) - POINTS):
            bin_target[np.unravel_index(bin_target.argmax(), bin_target.shape)] -= 1

        target_n = np.sum(bin_target)
        assert target_n >= POINTS

        subsampled = np.empty_like(lidar[:target_n, :])

        i = 0
        nx, ny, nz = bin_target.shape
        for (x, y, z), v in np.ndenumerate(bin_target):
            if v > 0:
                XX = edges[0][x:x + 2]
                YY = edges[1][y:y + 2]
                ZZ = edges[2][z:z + 2]
                # edge cases needed b/c histogramdd includes righest-most edge in bin
                if x < (nx - 1):
                    sublidar = lidar[(lidar[:, 0] >= XX[0]) & (lidar[:, 0] < XX[1])]
                else:
                    sublidar = lidar[(lidar[:, 0] >= XX[0]) & (lidar[:, 0] <= XX[1])]
                if y < (ny - 1):
                    sublidar = sublidar[(sublidar[:, 1] >= YY[0]) & (sublidar[:, 1] < YY[1])]
                else:
                    sublidar = sublidar[(sublidar[:, 1] >= YY[0]) & (sublidar[:, 1] <= YY[1])]
                if z < (nz - 1):
                    sublidar = sublidar[(sublidar[:, 2] >= ZZ[0]) & (sublidar[:, 2] < ZZ[1])]
                else:
                    sublidar = sublidar[(sublidar[:, 2] >= ZZ[0]) & (sublidar[:, 2] <= ZZ[1])]
                assert sublidar.shape[0] == bins[x, y, z]
                assert sublidar.shape[0] >= v
                subsampled[i:(i + v)] = sublidar[np.random.choice(range(sublidar.shape[0]), v, replace=False)]
                i += v
        return subsampled

    # For each detected object in frame, returns list of
    # - lidar_in_2d_bbox (MAX_POINTS, 3)
    # - label of object
    # - bbox 6 -> (Ax, Ay, Bx, By, l, h)
    def lidar_cone_of_detected_objects(self, frame, return_image=False):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]

        if return_image:
            if frame not in self.images:
                self._read_image(frame)
                assert frame in self.images
            image = self.images[frame]

        assert self._boxes is not None

        lidar_cones = []
        for box in self._boxes[frame]:
            lidar_in_2d_bbox = self.__lidar_in_2d_box(lidar, box)

            # we'll pass two lower points A and B, and l and h (that defines the bounding box in 3d)
            A = np.array([box[0, 0], box[1, 0]])
            B = np.array([box[0, 1], box[1, 1]])
            C = np.array([box[0, 3], box[1, 3]])
            l = np.linalg.norm(C - B)
            h = np.array([box[2, 4] - box[2, 0]])
            # import code
            # code.interact(local=locals())
            bbox = np.concatenate((A, B, [l], h), axis=0)
            lidar_cones.append((lidar_in_2d_bbox, bbox, Parser.kitti_cat_idxs[0]))  # TODO: deal with categories

            if return_image:
                bbox_l, bbox_h = self.__box_to_2d_box(box)
                _bbox_l = (int(bbox_l[0]), int(bbox_l[1]))  # todo ROUND result
                _bbox_h = (int(bbox_h[0]), int(bbox_h[1]))
                image = cv2.rectangle(image, _bbox_l, _bbox_h, (1., 1., 1.))
                print(lidar_in_2d_bbox.shape[0], "found in", bbox_l, bbox_h)

        if return_image:
            top_view = self.top_view(frame, with_boxes=True, SX=image.shape[1])
            image = np.concatenate((image, top_view), axis=0)

        if return_image:
            return lidar_cones, image
        return lidar_cones

    def side_view(self, frame, with_boxes = True):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]

        side_lidar = lidar[(lidar[1]) < 10. & (lidar[1] > -10.)]

        return

    def top_and_side_view(self, frame, with_boxes=True, lidar_override=None, SX=None, abl_overrides=None, zoom_to_box=False):
        tv = self.top_view(frame, with_boxes=with_boxes, lidar_override=lidar_override,
                           SX=SX, abl_overrides=abl_overrides, zoom_to_box=zoom_to_box,
                           remove_ground_plane=True, fine_tune_box = True, remove_points_below_plane = True)
        sv = self.top_view(frame, with_boxes=with_boxes, lidar_override=lidar_override,
                           SX=SX, abl_overrides=abl_overrides, zoom_to_box=zoom_to_box,
                           side_view=True, remove_points_below_plane = False)
        return np.concatenate((tv, sv), axis=0)

    # return a top view of the lidar image for frame
    # draw boxes for tracked objects if with_boxes is True
    #
    # SX are horizontal pixels of resulting image (vertical pixels maintain AR),
    # useful if you want to stack lidar below or above camera image
    #
    # if abl_override is provided it draws
    def top_view(self, frame, with_boxes=True, lidar_override=None, SX=None, abl_overrides=None,
                 zoom_to_box=False, side_view=False, fine_tune_box=False, remove_ground_plane = False, remove_points_below_plane =  False):

        print(frame)
        if with_boxes and zoom_to_box:
            assert self._boxes is not None
            box = self._boxes[frame][0] # first box for now
            cx = np.average(box[0,:])
            cy = np.average(box[1,:])
            X_SPAN  = 16.
            Y_SPAN  = 16.
        else:
            cx = 0.
            cy = 0.
            X_SPAN = 80.
            Y_SPAN = 80.

        X_RANGE = (cx - X_SPAN / 2., cx + X_SPAN / 2.)
        Y_RANGE = (cy - Y_SPAN / 2., cy + Y_SPAN / 2.)

        if SX is None:
            RES = 0.2
            Y_PIXELS = int(Y_SPAN / RES)
        else:
            Y_PIXELS = SX
            RES = Y_SPAN / SX
        X_PIXELS = int(X_SPAN / RES)

        top_view = np.zeros(shape=(X_PIXELS, Y_PIXELS, 3), dtype=np.float32)

        # convert from lidar x y to top view X Y
        def toY(y):
            return int((Y_PIXELS - 1) - (y - Y_RANGE[0]) // RES)

        def toX(x):
            return int((X_PIXELS - 1) - (x - X_RANGE[0]) // RES)

        def toXY(x, y):
            return (toY(y), toX(x))

        def inRange(x, y):
            return (x >= X_RANGE[0]) and (x < X_RANGE[1]) and (y >= Y_RANGE[0]) and (y < Y_RANGE[1])

        if lidar_override is not None:
            lidar = lidar_override
        else:
            if frame not in self.lidars:
                self._read_lidar(frame)
                assert frame in self.lidars
            lidar = self.lidars[frame]


        t_box = np.zeros((3))
        if remove_ground_plane:

            import pcl

            # get points close to the obstacle (7m) removing capture car (3m) just in case
            # this will be handy when we find the ground plane around the obstacle later
            lidar_close = lidar[( ((lidar[:, 0] - cx) ** 2 + (lidar[:, 1] - cy) ** 2) < 4 ** 2) & ((lidar[:, 0] ** 2 + lidar[:, 1] ** 2) >= 3**2)]

            # at a minimum we need 4 points (3 ground plane points plus 1 obstacle point)
            if (lidar_close.shape[0] >= 4):

                p = pcl.PointCloud(lidar_close[:,0:3].astype(np.float32))
                seg = p.make_segmenter()
                seg.set_optimize_coefficients(True)
                seg.set_model_type(pcl.SACMODEL_PLANE)
                seg.set_method_type(pcl.SAC_RANSAC)
                seg.set_distance_threshold(0.25)
                indices, model = seg.segment()
                gp = np.zeros((lidar_close.shape[0]), dtype=np.bool)
                gp[indices] = True
                lidar = lidar_close[~gp]

                a, b, c, d = model
                if remove_points_below_plane and  (len(lidar) > 1 ) and (len(model)== 4):

                    # see http://mathworld.wolfram.com/HessianNormalForm.html
                    # we can remove / dd because we're just interested in the sign
                    # dd = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                    lidar = lidar[( lidar[:, 0]* a + lidar[:,1] * b + lidar[:,2] * c  + d)  >= 0  ]

                ground_z = (-d - a * cx - b * cy) / c
                print(cx,cy,ground_z)

                if side_view is False and (lidar.shape[0] > 4):

                    # obs_isolated is just lidar points centered around 0,0 and sitting on ground 0 (z=0)
                    origin = np.array([cx,cy,ground_z])
                    obs_isolated = lidar[:,0:3]-origin
                    # rotate it so that it is aligned with our reference target
                    obs_isolated = point_utils.rotate(obs_isolated, np.array([0., 0., 1.]), self._get_yaw(frame))

                    np.save(str(frame), obs_isolated)
                    t_box = point_utils.rotate(self.align(obs_isolated), np.array([0., 0., 1.]), +self._get_yaw(frame))

                    #np.save(str(frame), lidar[:,0:3]-origin )



        if side_view:
            lidar = lidar[(lidar[:,1] >= -1.) & (lidar[:,1] <= 1.)]
            rot90X = np.array([[1, 0, 0, 0, 0], [0, 0, -1., 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]], dtype=np.float32)
            lidar  = np.dot(lidar, rot90X)


        # if we have camera calibration, use it to paint visible points in white, and not visible in gray
        # otherwise, just paint all white
        if self.kitti_data.calib is not None:
            in_img, outside_img = self.__project(lidar, return_projection=False, return_velo_in_img=True,
                                                 return_velo_outside_img=True)
        else:
            in_img      = lidar
            outside_img = []

        for point in in_img:
            x, y = point[0], point[1]
            if inRange(x, y):
                top_view[toXY(x, y)[::-1]] = (1., 1., 1.)

        for point in outside_img:
            x, y = point[0], point[1]
            if inRange(x, y):
                c = (0.2, 0.2, 0.2)
                if (self.LIDAR_ANGLE is not None) and (np.arctan2(x, np.absolute(y)) >= self.LIDAR_ANGLE):
                    c = (0.5, 0.5, 0.5)
                top_view[toXY(x, y)[::-1]] = c

        if with_boxes:
            assert self._boxes is not None
            boxes = self._boxes[frame]
            new_boxes = []
            if side_view:
                for box in boxes:
                    new_boxes.append(np.dot(box.T, rot90X[0:3, 0:3]).T)
            else:
                new_boxes = boxes
            if side_view:
                order = [0,2,6,4]
            else:
                order = [0,1,2,3]

            for box in new_boxes:
                # bounding box in image coords (x,y) defined by a->b->c->d
                a = np.array([toXY(box[0, order[0]], box[1, order[0]])])
                b = np.array([toXY(box[0, order[1]], box[1, order[1]])])
                c = np.array([toXY(box[0, order[2]], box[1, order[2]])])
                d = np.array([toXY(box[0, order[3]], box[1, order[3]])])

                cv2.polylines(top_view, [np.int32((a, b, c, d)).reshape((-1, 1, 2))], True, (1., 0., 0.), thickness=1)

                lidar_in_box  = self._lidar_in_box(frame, box)
                for point in lidar_in_box:
                    x, y = point[0], point[1]
                    if inRange(x, y):
                        top_view[toXY(x, y)[::-1]] = (0., 1., 1.)

                if fine_tune_box:

                    max_box = box - np.expand_dims(t_box, axis=1)

                    if max_box is not box:
                        a = np.array([toXY(max_box[0, order[0]], max_box[1, order[0]])])
                        b = np.array([toXY(max_box[0, order[1]], max_box[1, order[1]])])
                        c = np.array([toXY(max_box[0, order[2]], max_box[1, order[2]])])
                        d = np.array([toXY(max_box[0, order[3]], max_box[1, order[3]])])

                        cv2.polylines(top_view, [np.int32((a, b, c, d)).reshape((-1, 1, 2))], True, (0., 1., 0.), thickness=1)

        if abl_overrides is not None:
            color = np.array([1., 0., 0.])
            print("abl_overrides", abl_overrides)
            for i, abl_override in enumerate(abl_overrides):
                # A B are lidar bottom box corners
                A = np.array([abl_override[0], abl_override[1]])
                B = np.array([abl_override[2], abl_override[3]])
                l = abl_override[4]

                # given A,B and l, compute C and D by rotating -90 l * AB/|AB| and adding to B
                rot90 = np.array([[0, 1], [-1, 0]])
                AB = B - A
                ABn = np.linalg.norm(AB)
                C = B + l * np.dot(AB, rot90) / ABn
                D = C - AB

                c = toXY(C[0], C[1])
                d = toXY(D[0], D[1])
                a = toXY(abl_override[0], abl_override[1])
                b = toXY(abl_override[2], abl_override[3])

                cv2.polylines(top_view, [np.int32((a, b, c, d)).reshape((-1, 1, 2))], True, np.roll(color, i),
                              thickness=1)

        return top_view

    def __box_to_2d_box(self, box):
        box_in_img = self.__project(box.T, return_projection=True, dim_limit=None, return_velo_in_img=False,
                                    return_velo_outside_img=False)
        # some boxes are behind the viewpoint (eg. frame 70 @ drive 0036 ) and would return empty set of points
        # so we return an empty box
        if box_in_img.shape[0] != 8:
            return (0, 0), (0, 0)
        # print("lidar box", box.T,"in img", box_in_img)
        dim_limit = self.im_dim
        # clip 2d box corners within image
        box_in_img[:, 0] = np.clip(box_in_img[:, 0], 0, dim_limit[0])
        box_in_img[:, 1] = np.clip(box_in_img[:, 1], 0, dim_limit[1])
        # get 2d bbox
        bbox_l = (np.amin(box_in_img[:, 0]), np.amin(box_in_img[:, 1]))
        bbox_h = (np.amax(box_in_img[:, 0]), np.amax(box_in_img[:, 1]))
        return bbox_l, bbox_h

    def __lidar_in_2d_box(self, lidar, box):
        bbox_l, bbox_h = self.__box_to_2d_box(box)
        # print("2d clipping box", bbox_l, bbox_h, "filtering", lidar.shape)
        lidar_in_2d_box = self.__project(lidar,
                                         return_projection=False, dim_limit=bbox_h, dim_limit_zero=bbox_l,
                                         return_velo_in_img=True, return_velo_outside_img=False)
        # print("got", lidar_in_2d_box.shape, "in box")
        return lidar_in_2d_box

    # returns lidar points that are inside a given box, or just the indexes
    def _lidar_in_box(self, frame, box, ignore_z=False):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars

        lidar = self.lidars[frame]
        return self.__lidar_in_box(lidar, box, ignore_z=ignore_z)

    # returns lidar points that are inside a given box, or just the indexes
    def __lidar_in_box(self, lidar, box, return_idx_only=False, ignore_z=False):

        p = lidar[:, 0:3]

        # determine if points in M are inside a rectangle defined by AB AD (AB and AD are orthogonal)
        # tdlr: they are iff (0<AM⋅AB<AB⋅AB)∧(0<AM⋅AD<AD⋅AD)
        # http://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
        a = np.array([box[0, 0], box[1, 0]])
        b = np.array([box[0, 1], box[1, 1]])
        d = np.array([box[0, 3], box[1, 3]])
        ab = b - a
        ad = d - a
        abab = np.dot(ab, ab)
        adad = np.dot(ad, ad)

        amab = np.squeeze(np.dot(np.array([p[:, 0] - a[0], p[:, 1] - a[1]]).T, ab.reshape(-1, 2).T))
        amad = np.squeeze(np.dot(np.array([p[:, 0] - a[0], p[:, 1] - a[1]]).T, ad.reshape(-1, 2).T))

        if ignore_z:
            in_box_idx = np.where(
                (abab >= amab) & (amab >= 0.) & (amad >= 0.) & (adad >= amad))
        else:
            min_z = box[2, 0]
            max_z = box[2, 4]
            in_box_idx = np.where(
                (abab >= amab) & (amab >= 0.) & (amad >= 0.) & (adad >= amad) & (p[:, 2] >= min_z) & (p[:, 2] <= max_z))

        if return_idx_only:
            return in_box_idx

        points_in_box = np.squeeze(lidar[in_box_idx, :], axis=0)
        return points_in_box

    # given array of points with shape (N_points) and projection matrix w/ shape (3,4)
    # projects points onto a 2d plane
    # returns projected points (N_F_points,2) and
    # their LIDAR counterparts (N_F_points,3) (unless return_velo_in_img is set to False)
    #
    # N_F_points is the total number of resulting points after filtering (only_forward Z>0 by default)
    # and optionally filtering points projected into the image dimensions spec'd by dim_limit:
    #
    # Optionally providing dim_limit (sx,sy) limits projections that end up within (0-sx,0-sy)
    # only_forward to only get points with Z >= 0
    #
    def __project(self, points,
                  dim_limit=(-1, -1),
                  dim_limit_zero=(0, 0),
                  only_forward=True,
                  return_projection=True,
                  return_velo_in_img=True,
                  return_velo_outside_img=False,
                  return_append=None):

        if dim_limit == (-1, -1):
            dim_limit = self.im_dim

        assert return_projection or return_velo_in_img

        K = self.kitti_data.calib.K_cam0  # cam2 or cam0
        R = np.eye(4)
        R[0:3, 0:3] = K
        T = np.dot(R, self.kitti_data.calib.T_cam2_velo)[0:3]
        px = points

        if only_forward:
            only_forward_filter = px[:, 0] >= 0.
            px = px[only_forward_filter]
        if points.shape[1] < T.shape[1]:
            px = np.concatenate((px, np.ones(px.shape[0]).reshape(-1, 1)), axis=1)
        projection = np.dot(T, px.T).T

        norm = np.dot(projection[:, T.shape[0] - 1].reshape(-1, 1), np.ones((1, T.shape[0] - 1)))
        projection = projection[:, 0:T.shape[0] - 1] / norm

        if dim_limit is not None:
            x_limit, y_limit = dim_limit[0], dim_limit[1]
            x_limit_z, y_limit_z = dim_limit_zero[0], dim_limit_zero[1]
            only_in_img = (projection[:, 0] >= x_limit_z) & (projection[:, 0] < x_limit) & (
            projection[:, 1] >= y_limit_z) & (projection[:, 1] < y_limit)
            projection = projection[only_in_img]
            if return_velo_in_img:
                if return_velo_outside_img:
                    _px = px[~ only_in_img]
                px = px[only_in_img]
        if return_append is not None:
            appended = return_append[only_forward_filter][only_in_img]
            assert return_projection and return_velo_in_img
            return (projection, np.concatenate((px[:, 0:3], appended.reshape(-1, 1)), axis=1).T)
        if return_projection and return_velo_in_img:
            return (projection, px)
        elif (return_projection is False) and (return_velo_in_img):
            if return_velo_outside_img:
                return px, _px
            else:
                return px
        return projection

    def build_height_features(self, point_cam_in_img):
        assert False  # function not tested

        height_features = np.zeros(
            shape=(int((MAX_Z - MIN_Z) / HEIGHT_F_RES), int((MAX_X - MIN_X) / HEIGHT_F_RES), M + 2), dtype=np.float32)
        max_height_per_cell = np.zeros_like(height_features[:, :, 1])
        for p in point_cam_in_img.T:
            x = p[0]
            y = MAX_HEIGHT - np.clip(p[1], MIN_HEIGHT, MAX_HEIGHT)
            z = p[2]
            if (x >= MIN_X) and (x < MAX_X) and (z >= MIN_Z) and (z < MAX_Z):
                m = int(y // M_HEIGHT)
                xi = int((x + MIN_X) // HEIGHT_F_RES)
                zi = int((z - MIN_Z) // HEIGHT_F_RES)
                height_features[zi, xi, m] = max(y, height_features[zi, xi, m])
                if y >= max_height_per_cell[zi, xi]:
                    max_height_per_cell[zi, xi] = y
                    height_features[zi, xi, M] = p[3]  # intensity
                height_features[zi, xi, M + 1] += 1
        log64 = np.log(64)
        height_features[:, :, M + 1] = np.clip(np.log(1 + height_features[:, :, M + 1]) / log64, 0., 1.)
        return height_features


def build_front_view_features(point_cam_in_img):
    delta_theta = 0.08 / (180. / np.pi)  # horizontal resolution
    delta_phi = 0.4 / (180. / np.pi)  # vertical resolution as per http://velodynelidar.com/hdl-64e.html

    c_projection = np.empty((point_cam_in_img.shape[1], 5))  # -> c,r,height,distance,intensity
    points = point_cam_in_img.T
    # y in lidar is [0] in cam (x)
    # x in lidar is [2] in cam (z)
    # z in lidar is [1] in cam (y)
    c_range = (-40 / (180. / np.pi), 40 / (180. / np.pi))
    r_range = (-2.8 / (180. / np.pi), 15.3 / (180. / np.pi))

    c_projection[:, 0] = np.clip(
        np.arctan2(points[:, 0], points[:, 2]),
        c_range[0], c_range[1])  # c
    c_projection[:, 1] = np.clip(
        np.arctan2(points[:, 1], np.sqrt(points[:, 2] ** 2 + points[:, 0] ** 2)),
        r_range[0], r_range[1])  # r
    c_projection[:, 2] = MAX_HEIGHT - np.clip(points[:, 1], MIN_HEIGHT, MAX_HEIGHT)  # height
    c_projection[:, 3] = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)  # distance
    c_projection[:, 4] = points[:, 3]

    c_norm = np.zeros((C_H, C_W, 3))
    c_norm[np.int32((C_H - 1) * (c_projection[:, 1] - r_range[0]) // (r_range[1] - r_range[0])), np.int32(
        (C_W - 1) * (c_projection[:, 0] - c_range[0]) // (c_range[1] - c_range[0]))] = c_projection[:,
                                                                                       2:5]  # .reshape(-1,3)

    return c_norm