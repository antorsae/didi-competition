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
import pcl
import os
import sys
import re
import time
import scipy.interpolate

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

def find_tracklets(
        directory,
        filter=None,
        yaw_correction=0.,
        xml_filename="tracklet_labels_refined.xml",
        flip=False):
    diditracklets = []
    combined_filter = "(" + ")|(".join(filter) + "$)" if filter is not None else None
    if combined_filter is not None:
        combined_filter = combined_filter.replace("*", ".*")

    for root, dirs, files in os.walk(directory):
        for date in dirs:  # 1 2 3
            for _root, drives, files in os.walk(os.path.join(root, date)):  # ./1/ ./18/ ...
                for drive in drives:
                    if os.path.isfile(os.path.join(_root, drive, xml_filename)):
                        if filter is None or re.match(combined_filter, date + '/' + drive):
                            diditracklet = DidiTracklet(root, date, drive,
                                                        yaw_correction=yaw_correction,
                                                        xml_filename=xml_filename,
                                                        flip=flip)
                            diditracklets.append(diditracklet)

    return diditracklets


class DidiTracklet(object):
    kitti_cat_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Sitter', 'Cyclist', 'Tram', 'Misc', 'Person (sitting)']
    kitti_cat_idxs = range(1, 1 + len(kitti_cat_names))

    LIDAR_ANGLE = np.pi / 6.

    def __init__(self, basedir, date, drive, yaw_correction=0., xml_filename="tracklet_labels_refined.xml", flip=False):
        self.basedir = basedir
        self.date    = date
        self.drive   = drive

        self.kitti_data = pykitti.raw(basedir, date, drive,
                                      range(0, 1))  # , range(start_frame, start_frame + total_frames))
        self.xml_path = os.path.join(basedir, date, drive)
        self.tracklet_data = tracklets.parse_xml(os.path.join(self.xml_path, xml_filename))

        # correct yaw in all frames if yaw_correction provided
        if yaw_correction is not 0.:
            assert len(self.tracklet_data) == 1  # only one tracklet supported for now!
            for t in self.tracklet_data:
                for frame_offset in range(t.first_frame, t.first_frame + t.num_frames):
                    idx = frame_offset - t.first_frame
                    t.rots[idx][2] +=  yaw_correction

        self.kitti_data.load_calib()  # Calibration data are accessible as named tuples

        # lidars is a dict indexed by frame: e.g. lidars[10] = np(N,4)
        self.lidars = {}

        # images is a dict indexed by frame: e.g. lidars[10] = np(SY,SX,3)
        self.images = {}
        self.im_dim = (1242, 375)  # by default

        # boxes is a dict indexed by frame:  e.g. boxes[10] = [box, box, ...]
        self._boxes = None  # defaultdict(list)
        self._last_refined_box = None

        reference_file = os.path.join(basedir, date, 'obs.txt')

        if os.path.isfile(reference_file):
            if flip:
                print("Flipping")
            else:
                print("Not flipping")
            self.reference = self.__load_reference(reference_file, flip)
        else:
            self.reference = None

        self._init_boxes(only_with=None)


    def __load_reference(self, reference_file, flip=False):
        reference = np.genfromtxt(reference_file, dtype=np.float32, comments='/')

        # our reference model is in inches: convert to meters
        reference = np.multiply(reference[:, 0:3], np.array([0.0254, 0.0254, 0.0254]))

        reference_min = np.amin(reference[:, 0:3], axis=0)
        reference_lwh = np.amax(reference[:, 0:3], axis=0) - reference_min

        reference[:, 0:3] -= (reference_min[0:3] + reference_lwh[0:3] / 2.)

        # our reference model is rotated: align it correctly
        reference[:, 0:3] = point_utils.rotate(reference[:,0:3], np.array([1., 0., 0.]), np.pi / 2)

        # by default our reference model points to the opposite direction, so flip it accordingly
        if not flip:
            reference = point_utils.rotate(reference, np.array([0.,0.,1.]), np.pi)

        # flip it
        reference[:, 2] = -reference[:, 2]
        reference[:, 2] -= (np.amin(reference[:, 2]))

        # at this point our reference model is on lidar frame, centered around x=0,y=0 and sitting at z = 0
        return reference

    def _align(self, first, min_percent_first = 0.6, threshold_distance = 0.3, search_yaw=False):
        if self.reference is not None:

            model = point_utils.ICP(search_yaw=search_yaw)
            #first = point_utils.rotate(first, np.array([0., 0., 1.]), np.pi)

            t, _ = point_utils.ransac(first, self.reference[:, 0:3], model,
                                      min_percent_fist=min_percent_first,
                                      threshold= threshold_distance)

            if t is None:
                t =  np.zeros((3))

        else:
            print("No reference object, not aligning")
            t    = np.zeros((3))
        return t

    # for DIDI -> don't filter anything
    # for KITTI ->
    # include tracklet IFF in image and not occluded
    # WARNING: There's a lot of tracklets with occs=-1 (255) which we need to fix
    def __include_tracklet(self, t, idx):
        return True # (t.truncs[idx] == tracklets.Truncation.IN_IMAGE) and (t.occs[idx, 0] == 0)

    def get_yaw(self, frame):
        assert len(self.tracklet_data) == 1 # only one tracklet supported for now!
        for t in self.tracklet_data:
            assert frame in range(t.first_frame, t.first_frame + t.num_frames)
            idx = frame - t.first_frame
            yaw = t.rots[idx][2]
        return yaw

    def get_state(self, frame):
        assert len(self.tracklet_data) == 1  # only one tracklet supported for now!
        t = self.tracklet_data[0]
        return t.states[frame - t.first_frame]

    def get_box_first_frame(self, box=0):
        assert len(self.tracklet_data) == 1 # only one tracklet supported for now!
        return self.tracklet_data[box].first_frame

    def get_box_size(self, box=0):
        assert len(self.tracklet_data) == 1 # only one tracklet supported for now!
        return self.tracklet_data[box].size # h w l

    def get_box_TR(self, frame, box=0):
        assert len(self.tracklet_data) == 1 # only one tracklet supported for now!
        t = self.tracklet_data[box]
        assert frame in range(t.first_frame, t.first_frame + t.num_frames)
        idx = frame - t.first_frame
        T,R = t.trans[idx], t.rots[idx]
        return T,R

    def get_box_pose(self, frame, box=0):
        T,R = self.get_box_TR(frame, box=box)
        pose = {'tx': T[0], 'ty': T[1], 'tz': T[2], 'rx': R[0], 'ry': R[1], 'rz': R[2] }
        return pose

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
            self.kitti_data = pykitti.raw(self.basedir, self.date, self.drive, [frame])  # , range(start_frame, start_frame + total_frames))
            self.kitti_data.load_calib()
        assert frame in self.kitti_data.frame_range
        self.kitti_data.load_velo()
        if len(self.kitti_data.velo) != 1:
            print(frame, self, self.xml_path, self.kitti_data.velo)
            print(len(self.lidars))
        assert len(self.kitti_data.velo) == 1
        lidar = self.kitti_data.velo[0]
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
        #assert self._boxes is None
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
                                [-h/2., -h/2., -h/2., -h/2., h/2., h/2., h/2., h/2.]])
                                # CAREFUL: DIDI/UDACITY changed the semantics of a TZ!
                                #[0.0, 0.0, 0.0, 0.0, h, h, h, h]]) #
                        yaw = t.rots[idx][2]  # other rotations are 0 in all xml files I checked

                        assert np.abs(t.rots[idx][:2]).sum() == 0, 'object rotations other than yaw given!'
                        rotMat = np.array([
                            [np.cos(yaw), -np.sin(yaw), 0.0],
                            [np.sin(yaw), np.cos(yaw), 0.0],
                            [0.0, 0.0, 1.0]])
                        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(t.trans[idx], (8, 1)).T
                        self._boxes[frame_offset].append(cornerPosInVelo)
        return

    # given lidar points, subsample POINTS by removing points from voxels with highest density
    @staticmethod
    def _lidar_subsample(lidar, POINTS):
        # X_RANGE = (  0., 70.)
        # Y_RANGE = (-40., 40.)
        # Z_RANGE = ( -2.,  2.)
        # RES = 0.2 (not needed)

        NX = 10
        NY = 10
        NZ = 4

        bins, edges = np.histogramdd(lidar[:, 0:3], bins=(NX, NY, NZ))

        bin_target = np.array(bins, dtype=np.int32)
        subsample_time_start = time.time()

        bin_target_flat = bin_target.flatten()
        remaining = np.sum(bin_target) - POINTS

        bin_target_idx_sorted = np.argsort(bin_target_flat)
        maxi = bin_target_idx_sorted.shape[0] - 1
        i = maxi
        while (remaining > 0) and (i >= 0):
            maxt = bin_target_flat[bin_target_idx_sorted[maxi]]
            while bin_target_flat[bin_target_idx_sorted[i]] >= maxt:
                i -= 1
            available_to_substract = bin_target_flat[bin_target_idx_sorted[i+1:maxi+1]] - bin_target_flat[bin_target_idx_sorted[i]]
            total_available_to_substract = np.sum(available_to_substract)
            ii = i
            zz = 0
            if (total_available_to_substract < remaining):
                bin_target_flat[bin_target_idx_sorted[ii + 1:]] -= available_to_substract
                remaining -= total_available_to_substract
            else:
                while (total_available_to_substract > 0) and (remaining > 0):
                    to_substract = min(remaining, available_to_substract[zz])
                    bin_target_flat[bin_target_idx_sorted[ii+1]] -= to_substract
                    total_available_to_substract -= to_substract
                    remaining -= to_substract
                    ii += 1
                    zz += 1
                #print(bin_target_flat)
        #print(remaining)

        bin_target = bin_target_flat.reshape(bin_target.shape)
        #print("bin_target", bin_target)
        #print("_bin_target", _bin_target)

        subsample_time_end = time.time()
        #print 'Total subsample inside time: %0.3f ms' % ((subsample_time_end - subsample_time_start) * 1000.0)

        target_n = np.sum(bin_target)
        assert target_n == POINTS

        subsampled = np.empty((POINTS, lidar.shape[1]))

        i = 0
        j = maxi
        nx, ny, nz = bin_target.shape
#        for (x, y, z), v in np.ndenumerate(bin_target):
 #           if v > 0:
        while (bin_target_flat[bin_target_idx_sorted[j]] > 0):
            x,y,z = np.unravel_index(bin_target_idx_sorted[j], bin_target.shape)
            v = bin_target_flat[bin_target_idx_sorted[j]]
            XX = edges[0][x:x + 2]
            YY = edges[1][y:y + 2]
            ZZ = edges[2][z:z + 2]
            # edge cases needed b/c histogramdd includes righest-most edge in bin
            #if (x < (nx - 1)) & (y < (ny - 1)) & (z < (nz - 1)):
            #    sublidar = lidar[(lidar[:, 0] >= XX[0]) & (lidar[:, 0] < XX[1]) & (lidar[:, 1] >= YY[0]) & (lidar[:, 1] < YY[1]) & (lidar[:, 2] >= ZZ[0]) & (lidar[:, 2] < ZZ[1])]
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
            #subsampled[i:(i + v)] = sublidar[:v]

            i += v
            j -= 1
        return subsampled

    @staticmethod
    def _remove_capture_vehicle(lidar):
        return lidar[~ ((np.abs(lidar[:, 0]) < 2.6) & (np.abs(lidar[:, 1]) < 1.))]

    @staticmethod
    def resample_lidar(lidar, num_points):
        lidar_size = lidar.shape[0]
        if num_points > lidar_size:
            upsample_time_start = time.time()

            #lidar = np.concatenate((lidar, lidar[np.random.choice(lidar.shape[0], size=num_points - lidar_size, replace=True)]), axis = 0)
            if True:
                # tile existing array and pad it with missing slice
                reps = num_points // lidar_size - 1
                if reps > 0:
                    lidar = np.tile(lidar, (reps + 1, 1))
                missing = num_points - lidar.shape[0]
                lidar = np.concatenate((lidar, lidar[:missing]), axis=0)
                upsample_time_end = time.time()
                #print 'Total upsample time: %0.3f ms' % ((upsample_time_end - upsample_time_start) * 1000.0)

        elif num_points < lidar_size:
            subsample_time_start = time.time()
            lidar = DidiTracklet._lidar_subsample(lidar, num_points)
            subsample_time_end = time.time()
            #print 'Total subsample time: %0.3f ms' % ((subsample_time_end - subsample_time_start) * 1000.0)

        return lidar

    @staticmethod
    def filter_lidar(lidar, num_points = None, remove_capture_vehicle=True, max_distance = None):
        if remove_capture_vehicle:
            lidar = DidiTracklet._remove_capture_vehicle(lidar)

        if max_distance is not None:
            lidar = lidar[(lidar[:,0] ** 2 + lidar[:,1] ** 2) <= (max_distance **2)]

        if num_points is not None:
            lidar = DidiTracklet.resample_lidar(lidar, num_points)

        return lidar

    ''' Returns array len(rings), points_per_ring, 2 
    '''
    def get_lidar_rings(self, frame, rings, points_per_ring, clip=None, rotate=0., flipX = False, flipY=False, jitter=False):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar  = self.lidars[frame]
        if rotate != 0.:
            lidar = point_utils.rotZ(lidar, rotate)

        if flipX:
            lidar[:,0] = -lidar[:,0]
        if flipY:
            lidar[:,1] = -lidar[:,1]

        lidar_d_i = np.empty((len(rings), points_per_ring, 2), dtype=np.float32)
        for i in rings:
            l  = lidar[lidar[:,4] == i]
            lp = l.shape[0]
            if lp < 2:
                _int_d = _int_i = np.zeros((points_per_ring), dtype=np.float32)
            else:
                _r  = np.arctan2(l[:, 1], l[:, 0])  # y/x
                _d  = np.linalg.norm(l[:, :3], axis=1)  # total distance
                _i  = l[:,3]

                __d = scipy.interpolate.interp1d(_r, _d, fill_value='extrapolate', kind='nearest')
                __i = scipy.interpolate.interp1d(_r, _i, fill_value='extrapolate', kind='nearest')

                _int_d = __d(np.linspace(-np.pi, (points_per_ring - 1) * np.pi / points_per_ring, num=points_per_ring))
                if clip is not None:
                    np.clip(_int_d, clip[0], clip[1], out=_int_d)
                _int_i = __i(np.linspace(-np.pi, (points_per_ring - 1) * np.pi / points_per_ring, num=points_per_ring))

            lidar_d_i[i-rings[0]] = np.vstack((_int_d, _int_i)).T

        return lidar_d_i

    def _alias_lidar_rings(self, frame, rings, points_per_ring):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]
        lidar_d_i    = np.empty((rings, points_per_ring, 2))
        lidar_r_p    = np.empty((rings, points_per_ring, 2))

        lidar_points = np.empty((rings, points_per_ring, 3))
        for i in range(rings):
            l  = lidar[lidar[:,4] == i]
            lp = l.shape[0]
            #assert  lp<= (points_per_ring + PAD *2)
            _d = np.linalg.norm(l[:,:3], axis=1) # total distance
            _r = np.arctan2(l[:,1], l[:,0]) # y/x
            _p = np.arctan2(l[:,2], np.linalg.norm(l[:,:2], axis=1))     # z/sqrt(x**2+y**2)
            _i = l[:,3]
            _int_d = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = _d)
            _int_i = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = _i)
            _int_r = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = _r)
            _int_p = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = _p)

            lidar_d_i[i]    = np.vstack((_int_d, _int_i)).T
            lidar_r_p[i]    = np.vstack((_int_r, _int_p)).T

            _int_x = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = l[:,0])
            _int_y = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = l[:,1])
            _int_z = np.interp(np.linspace(0,1.,num=points_per_ring), xp = np.linspace(0,1.,num=lp), fp = l[:,2])

            lidar_points[i]  = np.vstack((_int_x, _int_y, _int_z)).T

        aliased_lidar = np.empty((rings * points_per_ring, 4))

        for i in range(rings):
            print( np.sin(lidar_r_p[i,:,1]).shape)
            z = np.expand_dims(lidar_d_i[i,:,0]  * np.sin(lidar_r_p[i,:,1]), axis=-1)
            print(z.shape)
            x = np.expand_dims(lidar_d_i[i,:,0] * np.cos(lidar_r_p[i,:,1])  * np.cos(lidar_r_p[i,:,0]), axis=-1)
            y = np.expand_dims(lidar_d_i[i,:,0] * np.cos(lidar_r_p[i,:,1])  * np.sin(lidar_r_p[i,:,0]), axis=-1)

            aliased_lidar[i*points_per_ring:(i+1) * points_per_ring] = np.concatenate((x,y,z, lidar_d_i[i,:,1:2]), axis=1)

        return aliased_lidar


    def get_lidar(self, frame, num_points = None, remove_capture_vehicle=True, max_distance = None):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]
        return self.filter_lidar(lidar, num_points = num_points, remove_capture_vehicle=remove_capture_vehicle, max_distance = max_distance)

    def get_box(self, frame):
        assert self._boxes is not None
        assert len(self._boxes[frame]) == 1
        box = self._boxes[frame][0] # first box for now
        return box

    def get_box_centroid(self, frame):
        assert self._boxes is not None
        assert len(self._boxes[frame]) == 1
        box = self._boxes[frame][0] # first box for now
        return np.average(box, axis=1)

    def get_number_of_points_in_box(self, frame, ignore_z=True):
        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        assert self._boxes is not None
        assert len(self._boxes[frame]) == 1
        box = self._boxes[frame][0] # first box for now
        lidar = self.lidars[frame]
        return len(self.__lidar_in_box(lidar, box, ignore_z=ignore_z))

    def top_and_side_view(self, frame, with_boxes=True, lidar_override=None, SX=None, abl_overrides=None, zoom_to_box=False, distance=50.):
        tv = self.top_view(frame, with_boxes=with_boxes, lidar_override=lidar_override,
                           SX=SX, zoom_to_box=zoom_to_box, distance=distance)
        sv = self.top_view(frame, with_boxes=with_boxes, lidar_override=lidar_override,
                           SX=SX, zoom_to_box=zoom_to_box, distance=distance,
                           side_view=True)
        return np.concatenate((tv, sv), axis=0)

    def refine_box(self,
                   frame,
                   remove_points_below_plane =  True,
                   search_ground_plane_radius = 20.,
                   search_centroid_radius = 4.,
                   look_back_last_refined_centroid=None,
                   return_aligned_clouds=False,
                   min_percent_first = 0.6,
                   threshold_distance = 0.3,
                   search_yaw=False):

        if look_back_last_refined_centroid is None:
            assert self._boxes is not None
            box = self._boxes[frame][0]  # first box for now
            cx = np.average(box[0, :])
            cy = np.average(box[1, :])
            cz = np.average(box[2, :])
        else:
            cx,cy,cz = look_back_last_refined_centroid
            print("Using last refined centroid", cx,cy,cz)

        T, _ = self.get_box_TR(frame)
        print("averaged centroid", cx,cy,cz, " vs ",T[0], T[1], T[2])

        t_box   = np.zeros((3))
        yaw_box = 0.

        if frame not in self.lidars:
            self._read_lidar(frame)
            assert frame in self.lidars
        lidar = self.lidars[frame]

        # get points close to the obstacle (d_range meters) removing capture car (2.6m x, 1m y) just in case
        # this will be handy when we find the ground plane around the obstacle later
        lidar_without_capture = DidiTracklet._remove_capture_vehicle(lidar)
        lidar_close = lidar_without_capture[( ((lidar_without_capture[:, 0] - cx) ** 2 + (lidar_without_capture[:, 1] - cy) ** 2) < search_ground_plane_radius ** 2) ]

        obs_isolated = []
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
            if remove_points_below_plane and (len(lidar) > 1 ) and (len(model)== 4):

                # see http://mathworld.wolfram.com/HessianNormalForm.html
                # we can remove / dd because we're just interested in the sign
                # dd = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                lidar = lidar[( lidar[:, 0]* a + lidar[:,1] * b + lidar[:,2] * c  + d)  >= 0  ]

            ground_z = (-d - a * cx - b * cy) / c
            print("Original centroid @ " + str((cx,cy,cz)) + " ground_z estimated @ " + str(ground_z)  )

            origin = np.array([cx, cy, ground_z])
            if lidar.shape[0] > 4:

                # obs_isolated is just lidar points centered around 0,0 and sitting on ground 0 (z=0)
                obs_isolated = lidar[:,0:3]-origin

                dd = np.sqrt(a ** 2 + b ** 2 + c ** 2)
                nx = a / dd
                ny = b / dd
                nz = c / dd
                print("Hessian normal", nx,ny,nz)
                roll   = np.arctan2(nx, nz)
                pitch  = np.arctan2(ny, nz)
                print("ground roll | pitch " + str(roll  * 180. / np.pi) + " | " + str(pitch * 180. / np.pi))

                # rotate it so that it is aligned with our reference target
                obs_isolated = point_utils.rotZ(obs_isolated, self.get_yaw(frame))

                # correct ground pitch and roll
                print("z min before correction", np.amin(obs_isolated[:,2]))

                obs_isolated = point_utils.rotate(obs_isolated, np.array([0., 1., 0.]), -roll)  # along Y axis
                obs_isolated = point_utils.rotate(obs_isolated, np.array([1., 0., 0.]), -pitch) # along X axis
                print("z min after correction", np.amin(obs_isolated[:,2]))

                # remove stuff beyond search_centroid_radius meters of the current centroid
                obs_cx = 0 #np.mean(obs_isolated[:,0])
                obs_cy = 0 #np.mean(obs_isolated[:,1])

                obs_isolated = obs_isolated[(((obs_isolated[:, 0] - obs_cx)** 2) + (obs_isolated[:, 1] - obs_cy) ** 2) <= search_centroid_radius ** 2]
                print("Isolated", obs_isolated.shape)
                if (obs_isolated.shape[0] > 0):
                    _t_box = self._align(
                        obs_isolated,
                        min_percent_first = min_percent_first,
                        threshold_distance = threshold_distance,
                        search_yaw=search_yaw)
                    yaw_box = _t_box[3] if search_yaw else 0.
                    _t_box[2] = 0
                    t_box = -point_utils.rotZ(_t_box[:3], -self.get_yaw(frame))

            # if we didn't find it in the first place, check if we found it in the last frame and attempt to find it from there
            if (t_box[0] == 0.) and (t_box[1] == 0.) and (look_back_last_refined_centroid is None) and (self._last_refined_box is not None):
                print("Looking back")
                t_box, _ = self.refine_box(frame,
                                           look_back_last_refined_centroid=self._last_refined_box,
                                           min_percent_first=min_percent_first,
                                           threshold_distance=threshold_distance,
                                           )
                t_box = -t_box

            new_ground_z = (-d - a * (cx+t_box[0]) - b * (cy+t_box[1])) / c
            print("original z centroid", T[2], "new ground_z", new_ground_z)
            t_box[2] = new_ground_z + self.tracklet_data[0].size[0]/2. - T[2]

            if look_back_last_refined_centroid is not None:
                t_box[:2] = t_box[:2] + origin[:2] - T[:2]

            if (t_box[0] != 0.) or (t_box[1] != 0.):
                self._last_refined_box = T + t_box
            else:
                self._last_refined_box = None

        print(t_box)
        print(yaw_box)
        if return_aligned_clouds:
            return t_box, yaw_box, self.reference[:, 0:3], obs_isolated

        return t_box, yaw_box


    # return a top view of the lidar image for frame
    # draw boxes for tracked objects if with_boxes is True
    #
    # SX are horizontal pixels of resulting image (vertical pixels maintain AR),
    # useful if you want to stack lidar below or above camera image
    #
    def top_view(self, frame, with_boxes=True, lidar_override=None, SX=None,
                 zoom_to_box=False, side_view=False, randomize=False, distance=50., rings=None, num_points=None):

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
            X_SPAN = distance*2.
            Y_SPAN = distance*2.

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
            if rings is not None:
                lidar = lidar[np.in1d(lidar[:,4],np.array(rings, dtype=np.float32))]
            if num_points is not None:
                lidar = DidiTracklet.resample_lidar(lidar, num_points)

        if randomize:
            centroid = self.get_box_centroid(frame)

            perturbation = (np.random.random_sample((lidar.shape[0], 5)) * 2. - np.array([1., 1., 1., 1., 1.])) * \
                           np.expand_dims(np.clip(
                               np.sqrt(((lidar[:, 0] - centroid[0]) ** 2) + ((lidar[:, 1] - centroid[1]) ** 2)) - 5.,
                               0., 20.), axis=1) * np.array([[2. / 20., 2. / 20., 0.1 / 20., 4. / 20., 0.]])
            lidar += perturbation

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
                if point[4] in range(10,25):
                    top_view[toXY(x, y)[::-1]] = np.ones(3)# * point[4] / 32.

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

                assert len(self.tracklet_data) == 1  # only one tracklet supported for now!
                t = self.tracklet_data[0]
                if t.states[frame - t.first_frame] == tracklets.STATE_UNSET:
                    box_color = (0., 0., 1.)
                else:
                    box_color = (1., 0., 0.)

                cv2.polylines(top_view, [np.int32((a, b, c, d)).reshape((-1, 1, 2))], True, box_color, thickness=1)

                lidar_in_box  = self._lidar_in_box(frame, box)
                for point in lidar_in_box:
                    x, y = point[0], point[1]
                    if inRange(x, y):
                        top_view[toXY(x, y)[::-1]] = (0., 1., 1.)

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