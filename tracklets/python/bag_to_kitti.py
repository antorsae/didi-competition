#! /usr/bin/python
""" Udacity Self-Driving Car Challenge Bag Processing
"""

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import sys
import cv2
import math
import imghdr
import argparse
import functools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyKDL as kd
import sensor_msgs.point_cloud2 as pc2

from bag_topic_def import *
from bag_utils import *
from generate_tracklet import *


# Bag message timestamp source
TS_SRC_PUB = 0
TS_SRC_REC = 1
TS_SRC_OBS_REC = 2


# Correction method
CORRECT_NONE = 0
CORRECT_PLANE = 1

CAP_RTK_FRONT_Z = .3323 + 1.2192
CAP_RTK_REAR_Z  = .3323 +  .8636

# From capture vehicle 'GPS FRONT' - 'LIDAR' in
# https://github.com/udacity/didi-competition/blob/master/mkz-description/mkz.urdf.xacro
FRONT_TO_LIDAR = [-1.0922, 0, -0.0508]

# For pedestrian capture, a different TF from mkz.urdf was used in capture. This must match
# so using that value here.
BASE_LINK_TO_LIDAR_PED = [1.9, 0., 1.6]

CAMERA_COLS = ["timestamp", "width", "height", "frame_id", "filename"]
LIDAR_COLS = ["timestamp", "points", "frame_id", "filename"]

GPS_COLS = ["timestamp", "lat", "long", "alt"]
POS_COLS = ["timestamp", "tx", "ty", "tz", "rx", "ry", "rz"]


def obs_name_from_topic(topic):
    return topic.split('/')[2]


def obs_prefix_from_topic(topic):
    words = topic.split('/')
    prefix = '_'.join(words[1:4])
    name = words[2]
    return prefix, name

def get_outdir(base_dir, name=''):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def check_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def write_image(bridge, outdir, msg, fmt='png'):
    results = {}
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        if hasattr(msg, 'format') and 'compressed' in msg.format:
            buf = np.ndarray(shape=(1, len(msg.data)), dtype=np.uint8, buffer=msg.data)
            cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
            if cv_image.shape[2] != 3:
                print("Invalid image %s" % image_filename)
                return results
            results['height'] = cv_image.shape[0]
            results['width'] = cv_image.shape[1]
            # Avoid re-encoding if we don't have to
            if check_format(msg.data) == fmt:
                buf.tofile(image_filename)
            else:
                cv2.imwrite(image_filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    results['filename'] = image_filename
    return results


def camera2dict(timestamp, msg, write_results, camera_dict):
    camera_dict["timestamp"].append(timestamp)
    if write_results:
        camera_dict["width"].append(write_results['width'] if 'width' in write_results else msg.width)
        camera_dict['height'].append(write_results['height'] if 'height' in write_results else msg.height)
        camera_dict["frame_id"].append(msg.header.frame_id)
        camera_dict["filename"].append(write_results['filename'])

def write_lidar(outdir, msg):
    results = {}
    lidar_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()))
    cloud_gen = pc2.read_points(msg)
    cloud = []
    point_count = 0
    for x, y, z, intensity, ring in cloud_gen:
        cloud.append([x, y, z, intensity, ring])
        point_count += 1
    results['points'] = point_count
    np.save(lidar_filename, cloud)
    results['filename'] = lidar_filename
    return results


def lidar2dict(timestamp, msg, write_results, lidar_dict):
    lidar_dict["timestamp"].append(timestamp)
    if write_results:
        lidar_dict["points"].append(write_results['points'] if 'points' in write_results else msg.width)
        lidar_dict["frame_id"].append(msg.header.frame_id)
        lidar_dict["filename"].append(write_results['filename'])

def gps2dict(timestamp, msg, gps_dict):
    gps_dict["timestamp"].append(timestamp)
    gps_dict["lat"].append(msg.latitude)
    gps_dict["long"].append(msg.longitude)
    gps_dict["alt"].append(msg.altitude)


def pose2dict(timestamp, msg, pose_dict):
    pose_dict["timestamp"].append(timestamp)
    pose_dict["tx"].append(msg.pose.position.x)
    pose_dict["ty"].append(msg.pose.position.y)
    pose_dict["tz"].append(msg.pose.position.z)
    rotq = kd.Rotation.Quaternion(
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w)
    rot_xyz = rotq.GetRPY()
    pose_dict["rx"].append(rot_xyz[0])
    pose_dict["ry"].append(rot_xyz[1])
    pose_dict["rz"].append(rot_xyz[2])


def tf2dict(timestamp, tf, tf_dict):
    tf_dict["timestamp"].append(timestamp)
    tf_dict["tx"].append(tf.translation.x)
    tf_dict["ty"].append(tf.translation.y)
    tf_dict["tz"].append(tf.translation.z)
    rotq = kd.Rotation.Quaternion(
        tf.rotation.x,
        tf.rotation.y,
        tf.rotation.z,
        tf.rotation.w)
    rot_xyz = rotq.GetRPY()
    tf_dict["rx"].append(rot_xyz[0])
    tf_dict["ry"].append(rot_xyz[1])
    tf_dict["rz"].append(rot_xyz[2])


def imu2dict(timestamp, msg, imu_dict):
    imu_dict["timestamp"].append(timestamp)
    imu_dict["ax"].append(msg.linear_acceleration.x)
    imu_dict["ay"].append(msg.linear_acceleration.y)
    imu_dict["az"].append(msg.linear_acceleration.z)


def get_yaw(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])


def dict_to_vect(di):
    return kd.Vector(di['tx'], di['ty'], di['tz'])


def list_to_vect(li):
    return kd.Vector(li[0], li[1], li[2])


def vect_to_dict3(v):
    return dict(tx=v[0], ty=v[1], tz=v[2])


def vect_to_dict6(v):
    if len(v) == 6:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=v[3], ry=v[4], rz=v[5])
    else:
        return dict(tx=v[0], ty=v[1], tz=v[2], rx=0, ry=0, rz=0)


def frame_to_dict(frame, yaw_only=False):
    r, p, y = frame.M.GetRPY()
    if yaw_only:
        return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=0., ry=0., rz=y)
    return dict(tx=frame.p[0], ty=frame.p[1], tz=frame.p[2], rx=r, ry=p, rz=y)


def dict_to_frame(di):
    return kd.Frame(
        kd.Rotation.RPY(di['rx'], di['ry'], di['rz']),
        kd.Vector(di['tx'], di['ty'], di['tz']))


def init_df(data_dict, cols, filename, outdir=''):
    df = pd.DataFrame(data=data_dict, columns=cols)
    if len(df.index) and filename:
        df.to_csv(os.path.join(outdir, filename), index=False)
    return df

def interpolate_df(input_dfs, index_df, filter_cols=[], filename='', outdir=''):
    if not isinstance(input_dfs, list):
        input_dfs = [input_dfs]
    if not isinstance(index_df.index, pd.DatetimeIndex):
        print('Error: Camera/lidar dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for i in input_dfs:
        if len(i.index) == 0:
            print('Warning: Empty dataframe passed to interpolate, skipping.')
            return pd.DataFrame()
        i['timestamp'] = pd.to_datetime(i['timestamp'])
        i.set_index(['timestamp'], inplace=True)
        i.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [index_df] + input_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[index_df.index]  # back to only index' rows
    filtered.fillna(0.0, inplace=True)
    filtered['timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    if len(filtered.index) and filename:
        filtered.to_csv(os.path.join(outdir, filename), header=True)
    return filtered

def obstacle_rtk_to_pose(
        cap_front,
        cap_rear,
        obs_front,
        obs_rear,
        obs_gps_to_centroid,
        front_to_velodyne,
        cap_yaw_err=0.,
        cap_pitch_err=0.):

    # calculate capture yaw in ENU frame and setup correction rotation
    cap_front_v = dict_to_vect(cap_front)
    cap_rear_v = dict_to_vect(cap_rear)
    cap_yaw = get_yaw(cap_front_v, cap_rear_v)
    cap_yaw += cap_yaw_err
    rot_cap = kd.Rotation.EulerZYX(-cap_yaw, -cap_pitch_err, 0)

    obs_rear_v = dict_to_vect(obs_rear)
    if obs_front:
        obs_front_v = dict_to_vect(obs_front)
        obs_yaw = get_yaw(obs_front_v, obs_rear_v)
        # use the front gps as the obstacle reference point if it exists as it's closer
        # to the centroid and mounting metadata seems more reliable
        cap_to_obs = obs_front_v - cap_front_v
    else:
        cap_to_obs = obs_rear_v - cap_front_v

    # transform capture car to obstacle vector into capture car velodyne lidar frame
    res = rot_cap * cap_to_obs
    res += list_to_vect(front_to_velodyne)

    # obs_gps_to_centroid is offset for front gps if it exists, otherwise rear
    obs_gps_to_centroid_v = list_to_vect(obs_gps_to_centroid)
    if obs_front:
        # if we have both front + rear RTK calculate an obstacle yaw and use it for centroid offset
        obs_rot_z = kd.Rotation.RotZ(obs_yaw - cap_yaw)
        centroid_offset = obs_rot_z * obs_gps_to_centroid_v
    else:
        # if no obstacle yaw calculation possible, treat rear RTK as centroid and offset in Z only
        obs_rot_z = kd.Rotation()
        centroid_offset = kd.Vector(0, 0, obs_gps_to_centroid_v[2])
    res += centroid_offset
    return frame_to_dict(kd.Frame(obs_rot_z, res), yaw_only=True)

def old_get_obstacle_pos(
        front,
        rear,
        obstacle,
        obstacle_yaw,
        velodyne_to_front,
        gps_to_centroid):
    front_v = dict_to_vect(front)
    rear_v = dict_to_vect(rear)
    obs_v = dict_to_vect(obstacle)

    yaw = get_yaw(front_v, rear_v)
    rot_z = kd.Rotation.RotZ(-yaw)

    diff = obs_v - front_v
    res = rot_z * diff
    res += list_to_vect(velodyne_to_front)

    # FIXME the gps_to_centroid offset of the obstacle should be rotated by
    # the obstacle's yaw. Unfortunately the obstacle's pose is unknown at this
    # point so we will assume obstacle is axis aligned with capture vehicle
    # for now.

    centroid = kd.Rotation.RotZ( obstacle_yaw - yaw ) * list_to_vect(gps_to_centroid)

    res += centroid # list_to_vect(gps_to_centroid)

    return frame_to_dict(kd.Frame(kd.Rotation.RotZ(obstacle_yaw - yaw), res))


def interpolate_to_target(target_df, other_dfs, filter_cols=[]):
    if not isinstance(other_dfs, list):
        other_dfs = [other_dfs]
    if not isinstance(target_df.index, pd.DatetimeIndex):
        print('Error: Camera dataframe needs to be indexed by timestamp for interpolation')
        return pd.DataFrame()

    for o in other_dfs:
        o['timestamp'] = pd.to_datetime(o['timestamp'])
        o.set_index(['timestamp'], inplace=True)
        o.index.rename('index', inplace=True)

    merged = functools.reduce(lambda left, right: pd.merge(
        left, right, how='outer', left_index=True, right_index=True), [target_df] + other_dfs)
    merged.interpolate(method='time', inplace=True, limit=100, limit_direction='both')

    filtered = merged.loc[target_df.index].copy()  # back to only camera rows
    filtered.fillna(0.0, inplace=True)
    filtered.loc[:,'timestamp'] = filtered.index.astype('int')  # add back original timestamp integer col
    if filter_cols:
        if not 'timestamp' in filter_cols:
            filter_cols += ['timestamp']
        filtered = filtered[filter_cols]

    return filtered


def estimate_obstacle_poses(
    cap_front_rtk,
    #cap_front_gps_offset,
    cap_rear_rtk,
    #cap_rear_gps_offset,
    obs_rear_rtk,
    obs_rear_gps_offset,  # offset along [l, w, h] dim of car, in obstacle relative coords
    obs_yaw
):
    # offsets are all [l, w, h] lists (or tuples)
    assert(len(obs_rear_gps_offset) == 3)
    # all coordinate records should be interpolated to same sample base at this point
    assert len(cap_front_rtk) == len(cap_rear_rtk) == len(obs_rear_rtk)

    velo_to_front = [-1.0922, 0, -0.0508]
    rtk_coords = zip(cap_front_rtk, cap_rear_rtk, obs_rear_rtk, obs_yaw)
    output_poses = [
        get_obstacle_pos(c[0], c[1], c[2], c[3], velo_to_front, obs_rear_gps_offset) for c in rtk_coords]

    return output_poses


def check_oneof_topics_present(topic_map, name, topics):
    if not isinstance(topics, list):
        topics = [topics]
    if not any(t in topic_map for t in topics):
        print('Error: One of %s must exist in bag, skipping bag %s.' % (topics, name))
        return False
    return True

def extract_metadata(md, obs_name):
    md = next(x for x in md if x['obstacle_name'] == obs_name)
    if 'gps_l' in md:
        # make old rear RTK only obstacle metadata compatible with new
        md['rear_gps_l'] = md['gps_l']
        md['rear_gps_w'] = md['gps_w']
        md['rear_gps_h'] = md['gps_h']
    return md

def process_rtk_data(
        bagset,
        cap_data,
        obs_data,
        index_df,
        outdir,
        correct=CORRECT_NONE,
        yaw_err=0.,
        pitch_err=0.
):
    tracklets = []
    cap_rear_gps_df = init_df(cap_data['rear_gps'], GPS_COLS, 'cap_rear_gps.csv', outdir)
    cap_front_gps_df = init_df(cap_data['front_gps'], GPS_COLS, 'cap_front_gps.csv', outdir)
    cap_rear_rtk_df = init_df(cap_data['rear_rtk'], POS_COLS, 'cap_rear_rtk.csv', outdir)
    cap_front_rtk_df = init_df(cap_data['front_rtk'], POS_COLS, 'cap_front_rtk.csv', outdir)
    if not len(cap_rear_rtk_df.index):
        print('Error: No capture vehicle rear RTK entries exist.'
              ' Skipping bag %s.' % bagset.name)
        return tracklets
    if not len(cap_rear_rtk_df.index):
        print('Error: No capture vehicle front RTK entries exist.'
              ' Skipping bag %s.' % bagset.name)
        return tracklets

    rtk_z_offsets = [np.array([0., 0., CAP_RTK_FRONT_Z]), np.array([0., 0., CAP_RTK_REAR_Z])]
    if correct > 0:
        # Correction algorithm attempts to fit plane to rtk measurements across both capture rtk
        # units and all obstacles. We will subtract known RTK unit mounting heights first.
        cap_front_points = cap_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[0]
        cap_rear_points = cap_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - rtk_z_offsets[1]
        point_arrays = [cap_front_points, cap_rear_points]
        filtered_point_arrays = [filter_outlier_points(cap_front_points), filter_outlier_points(cap_rear_points)]

    obs_rtk_dfs = {}
    for obs_name, obs_rtk_dict in obs_data.items():
        obs_front_rtk_df = init_df(obs_rtk_dict['front_rtk'], POS_COLS, '%s_front_rtk.csv' % obs_name, outdir)
        obs_rear_rtk_df = init_df(obs_rtk_dict['rear_rtk'], POS_COLS, '%s_rear_rtk.csv' % obs_name, outdir)
        if not len(obs_rear_rtk_df.index):
            print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bagset.name))
            continue
        obs_rtk_dfs[obs_name] = {'rear': obs_rear_rtk_df}
        if len(obs_front_rtk_df.index):
            obs_rtk_dfs[obs_name]['front'] = obs_front_rtk_df
        if correct > 0:
            # Use obstacle metadata to determine rtk mounting height and subtract that height
            # from obstacle readings
            md = extract_metadata(bagset.metadata, obs_name)
            if not md:
                print('Error: No metadata found for %s, skipping obstacle.' % obs_name)
                continue
            if len(obs_front_rtk_df.index):
                obs_z_offset = np.array([0., 0., md['front_gps_h']])
                rtk_z_offsets.append(obs_z_offset)
                obs_front_points = obs_front_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
                point_arrays.append(obs_front_points)
                filtered_point_arrays.append(filter_outlier_points(obs_front_points))
            obs_z_offset = np.array([0., 0., md['rear_gps_h']])
            rtk_z_offsets.append(obs_z_offset)
            obs_rear_points = obs_rear_rtk_df.as_matrix(columns=['tx', 'ty', 'tz']) - obs_z_offset
            point_arrays.append(obs_rear_points)
            filtered_point_arrays.append(filter_outlier_points(obs_rear_points))

    if correct == CORRECT_PLANE:
        points = np.array(np.concatenate(filtered_point_arrays))
        centroid, normal, rotation = fit_plane(
            points, do_plot=True, dataset_outdir=outdir, name=bagset.name)

        def apply_correction(p, z):
            p -= centroid
            p = np.dot(rotation, p.T).T
            c = np.concatenate([centroid[0:2], z[2:]])
            p += c
            return p

        corrected_points = [apply_correction(pa, z) for pa, z in zip(point_arrays, rtk_z_offsets)]
        cap_front_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[0]
        cap_rear_rtk_df.loc[:, ['tx', 'ty', 'tz']] = corrected_points[1]
        pts_idx = 2
        for obs_name in obs_rtk_dfs.keys():
            if 'front' in obs_rtk_dfs[obs_name]:
                obs_rtk_dfs[obs_name]['front'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
                pts_idx += 1
            obs_rtk_dfs[obs_name]['rear'].loc[:, ['tx', 'ty', 'tz']] = corrected_points[pts_idx]
            pts_idx += 1

    interpolate_df(
        cap_front_gps_df, index_df, GPS_COLS, 'cap_front_gps_interp.csv', outdir)
    interpolate_df(
        cap_rear_gps_df, index_df, GPS_COLS, 'cap_rear_gps_interp.csv', outdir)
    cap_front_rtk_interp = interpolate_df(
        cap_front_rtk_df, index_df, POS_COLS, 'cap_front_rtk_interp.csv', outdir)
    cap_rear_rtk_interp = interpolate_df(
        cap_rear_rtk_df, index_df, POS_COLS, 'cap_rear_rtk_interp.csv', outdir)

    if not obs_rtk_dfs:
        print('Warning: No obstacles or obstacle RTK data present. '
              'Skipping Tracklet generation for %s.' % bagset.name)
        return tracklets
    if not bagset.metadata:
        print('Error: No metadata found, metadata.csv file should be with .bag files.'
              'Skipping tracklet generation.')
        return tracklets

    cap_front_rtk_rec = cap_front_rtk_interp.to_dict(orient='records')
    cap_rear_rtk_rec = cap_rear_rtk_interp.to_dict(orient='records')
    for obs_name in obs_rtk_dfs.keys():
        obs_front_rec = {}
        if 'front' in obs_rtk_dfs[obs_name]:
            obs_front_interp = interpolate_df(
                obs_rtk_dfs[obs_name]['front'], index_df, POS_COLS, '%s_front_rtk_interpolated.csv' % obs_name, outdir)
            obs_front_rec = obs_front_interp.to_dict(orient='records')
        obs_rear_interp = interpolate_df(
            obs_rtk_dfs[obs_name]['rear'], index_df, POS_COLS, '%s_rear_rtk_interpolated.csv' % obs_name, outdir)
        obs_rear_rec = obs_rear_interp.to_dict(orient='records')

        # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
        fig = plt.figure()
        plt.plot(
            cap_front_rtk_interp['tx'].tolist(),
            cap_front_rtk_interp['ty'].tolist(),
            cap_rear_rtk_interp['tx'].tolist(),
            cap_rear_rtk_interp['ty'].tolist(),
            obs_rear_interp['tx'].tolist(),
            obs_rear_interp['ty'].tolist())
        if 'front' in obs_rtk_dfs[obs_name]:
            plt.plot(
                obs_front_interp['tx'].tolist(),
                obs_front_interp['ty'].tolist())
        fig.savefig(os.path.join(outdir, '%s-%s-plot.png' % (bagset.name, obs_name)))
        plt.close(fig)

        # Extract lwh and object type from CSV metadata mapping file
        md = extract_metadata(bagset.metadata, obs_name)

        obs_tracklet = Tracklet(
            object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

        # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
        # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
        # along the respective length, width, height axis away from that point. They are converted to
        # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
        lrg_to_centroid = [md['l'] / 2., -md['w'] / 2., md['h'] / 2.]
        if 'front' in obs_rtk_dfs[obs_name]:
            lrg_to_front_gps = [md['front_gps_l'], -md['front_gps_w'], md['front_gps_h']]
            gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_front_gps)
        else:
            lrg_to_rear_gps = [md['rear_gps_l'], -md['rear_gps_w'], md['rear_gps_h']]
            gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_rear_gps)

        # Convert ENU RTK coords of obstacle to capture vehicle body frame relative coordinates
        if obs_front_rec:
            rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_front_rec, obs_rear_rec)
            obs_tracklet.poses = [obstacle_rtk_to_pose(
                c[0], c[1], c[2], c[3],
                gps_to_centroid, FRONT_TO_LIDAR, yaw_err, pitch_err) for c in rtk_coords]
        else:
            rtk_coords = zip(cap_front_rtk_rec, cap_rear_rtk_rec, obs_rear_rec)
            obs_tracklet.poses = [obstacle_rtk_to_pose(
                c[0], c[1], {}, c[2],
                gps_to_centroid, FRONT_TO_LIDAR, yaw_err, pitch_err) for c in rtk_coords]

        tracklets.append(obs_tracklet)
    return tracklets


def process_pose_data(
        bagset,
        cap_data,
        obs_data,
        index_df,
        outdir,
):
    tracklets = []
    cap_pose_df = init_df(cap_data['base_link_pose'], POS_COLS, 'cap_pose.csv', outdir)
    cap_pose_interp = interpolate_df(
        cap_pose_df, index_df, POS_COLS, 'cap_pose_interp.csv', outdir)
    cap_pose_rec = cap_pose_interp.to_dict(orient='records')

    for obs_name, obs_pose_dict in obs_data.items():
        obs_pose_df = init_df(obs_pose_dict['pose'], POS_COLS, 'obs_pose.csv', outdir)
        obs_pose_interp = interpolate_df(
            obs_pose_df, index_df, POS_COLS, 'obs_pose_interp.csv', outdir)
        obs_pose_rec = obs_pose_interp.to_dict(orient='records')

        # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
        fig = plt.figure()
        plt.plot(
            obs_pose_interp['tx'].tolist(),
            obs_pose_interp['ty'].tolist(),
            cap_pose_interp['tx'].tolist(),
            cap_pose_interp['ty'].tolist())
        fig.savefig(os.path.join(outdir, '%s-%s-plot.png' % (bagset.name, obs_name)))
        plt.close(fig)

        # FIXME hard coded metadata, only Pedestrians currently using pose capture and there is only one person
        md = {'object_type': 'Pedestrian', 'l': 0.8, 'w': 0.8, 'h': 1.708}
        base_link_to_lidar = BASE_LINK_TO_LIDAR_PED

        obs_tracklet = Tracklet(
            object_type=md['object_type'], l=md['l'], w=md['w'], h=md['h'], first_frame=0)

        def _calc_cap_to_obs(cap, obs):
            cap_frame = dict_to_frame(cap)
            obs_frame = dict_to_frame(obs)
            cap_to_obs = cap_frame.Inverse() * obs_frame
            cap_to_obs.p -= list_to_vect(base_link_to_lidar)
            cap_to_obs.p -= kd.Vector(0, 0, md['h'] / 2)
            return frame_to_dict(cap_to_obs, yaw_only=True)

        obs_tracklet.poses = [_calc_cap_to_obs(c[0], c[1]) for c in zip(cap_pose_rec, obs_pose_rec)]
        tracklets.append(obs_tracklet)
    return tracklets



def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data',
        help='Input folder where bagfiles are located')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-t', '--ts_src', type=str, nargs='?', default='pub',
        help="""Timestamp source. 'pub'=capture node publish time, 'rec'=receiver bag record time,
        'obs_rec'=record time for obstacles topics only, pub for others. Default='pub'""")
    parser.add_argument('-m', dest='msg_only', action='store_true', help='Messages only, no images')
    parser.add_argument('-l', dest='include_lidar', action='store_true', help='Include lidar')
    parser.add_argument('-L', dest='index_by_lidar', action='store_true', help='Index tracklets by lidar frames instead of camera frames')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.add_argument('-u', dest='unique_paths', action='store_true', help='Unique bag output paths')
    parser.set_defaults(msg_only=False)
    parser.set_defaults(unique_paths=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    indir = args.indir
    ts_src = TS_SRC_PUB
    if args.ts_src == 'rec':
        ts_src = TS_SRC_REC
    elif args.ts_src == 'obs_rec':
        ts_src = TS_SRC_OBS_REC
    msg_only = args.msg_only
    debug_print = args.debug
    unique_paths = args.unique_paths
    
    bridge = CvBridge()

    include_images = False if msg_only else True
    include_lidar  = args.include_lidar
    index_by_lidar = args.index_by_lidar

    filter_topics = CAMERA_TOPICS + CAP_FRONT_RTK_TOPICS + CAP_REAR_RTK_TOPICS \
        + CAP_FRONT_GPS_TOPICS + CAP_REAR_GPS_TOPICS + LIDAR_TOPICS

    # FIXME hard coded obstacles
    # The original intent was to scan bag info for obstacles and populate dynamically in combination
    # with metadata.csv. Since obstacle names were very static, and the obstacle topic root was not consistent
    # between data releases, that didn't happen.
    obstacle_topics = []

    # For obstacles tracked via RTK messages
    OBS_RTK_NAMES = ['obs1']
    OBS_FRONT_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/front/gps/rtkfix' for x in OBS_RTK_NAMES]
    OBS_REAR_RTK_TOPICS = [OBJECTS_TOPIC_ROOT + '/' + x + '/rear/gps/rtkfix' for x in OBS_RTK_NAMES]
    obstacle_topics += OBS_FRONT_RTK_TOPICS
    obstacle_topics += OBS_REAR_RTK_TOPICS

    # For obstacles tracked via TF + pose messages
    OBS_POSE_TOPICS = ['/obstacle/ped/pose']  # not under same root as other obstacles for some reason
    obstacle_topics += OBS_POSE_TOPICS
    filter_topics += [TF_TOPIC]  # pose based obstacles rely on TF

    filter_topics += obstacle_topics

    bagsets = find_bagsets(indir, filter_topics=filter_topics, set_per_file=True, metadata_filename='metadata.csv')
    if not bagsets:
        print("No bags found in %s" % indir)
        exit(-1)

    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        cap_data = defaultdict(lambda: defaultdict(list))
        obs_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        outdir = os.path.join(base_outdir, bs.get_name(unique_paths))
        print( bs.get_name(unique_paths))
        get_outdir(outdir)

        dataset_outdir = os.path.join(base_outdir, "%s" % bs.name)
        print(dataset_outdir)

        get_outdir(dataset_outdir)
        if include_images:
            camera_outdir = get_outdir(dataset_outdir, "camera")
        if include_lidar:
            lidar_outdir  = get_outdir(dataset_outdir, "lidar")

        bs.write_infos(dataset_outdir)
        readers = bs.get_readers()
        stats_acc = defaultdict(int)

        def _process_msg(topic, msg, ts_recorded, stats):
            if topic == '/tf':
                timestamp = msg.transforms[0].header.stamp.to_nsec()
            else:
                timestamp = msg.header.stamp.to_nsec()  # default to publish timestamp in message header
            if ts_src == TS_SRC_REC:
                timestamp = ts_recorded.to_nsec()
            elif ts_src == TS_SRC_OBS_REC and topic in obstacle_topics:
                timestamp = ts_recorded.to_nsec()

            if topic in CAMERA_TOPICS:
                if debug_print:
                    print("%s_camera %d" % (topic[1], timestamp))

                write_results = {}
                if include_images:
                    write_results = write_image(bridge, camera_outdir, msg, fmt=img_format)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                camera2dict(timestamp, msg, write_results, cap_data['camera'])
                stats['img_count'] += 1
                stats['msg_count'] += 1

            elif topic in LIDAR_TOPICS:
                if debug_print:
                    print("%s_lidar %d" % (topic[1], timestamp))

                write_results = {}
                if include_lidar:
                    write_results = write_lidar(lidar_outdir, msg)
                    write_results['filename'] = os.path.relpath(write_results['filename'], dataset_outdir)
                lidar2dict(timestamp, msg, write_results, cap_data['lidar'])
                stats['lidar_count'] += 1
                stats['msg_count'] += 1


            elif topic in CAP_REAR_RTK_TOPICS:
                pose2dict(timestamp, msg.pose, cap_data['rear_rtk'])
                stats['msg_count'] += 1


            elif topic in CAP_FRONT_RTK_TOPICS:
                pose2dict(timestamp, msg.pose, cap_data['front_rtk'])
                stats['msg_count'] += 1


            elif topic in CAP_REAR_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_data['rear_gps'])
                stats['msg_count'] += 1


            elif topic in CAP_FRONT_GPS_TOPICS:
                gps2dict(timestamp, msg, cap_data['front_gps'])
                stats['msg_count'] += 1


            elif topic in OBS_REAR_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg.pose, obs_data[name]['rear_rtk'])
                stats['msg_count'] += 1


            elif topic in OBS_FRONT_RTK_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg.pose, obs_data[name]['front_rtk'])
                stats['msg_count'] += 1


            elif topic == TF_TOPIC:
                for t in msg.transforms:
                    if t.child_frame_id == '/base_link':
                        tf2dict(timestamp, t.transform, cap_data['base_link_pose'])


            elif topic in OBS_POSE_TOPICS:
                name = obs_name_from_topic(topic)
                pose2dict(timestamp, msg, obs_data[name]['pose'])
                stats['msg_count'] += 1

            else:
                pass

        for reader in readers:
            last_img_log = 0
            last_msg_log = 0
            for result in reader.read_messages():
                _process_msg(*result, stats=stats_acc)
                if last_img_log != stats_acc['img_count'] and stats_acc['img_count'] % 1000 == 0:
                    print("%d images, processed..." % stats_acc['img_count'])
                    last_img_log = stats_acc['img_count']
                    sys.stdout.flush()
                if last_msg_log != stats_acc['msg_count'] and stats_acc['msg_count'] % 10000 == 0:
                    print("%d messages processed..." % stats_acc['msg_count'])
                    last_msg_log = stats_acc['msg_count']
                    sys.stdout.flush()

        print("Writing done. %d images, %d lidar frames, %d messages processed." %
              (stats_acc['img_count'], stats_acc['lidar_count'], stats_acc['msg_count']))
        sys.stdout.flush()
        camera_df = pd.DataFrame(data=cap_data['camera'], columns=CAMERA_COLS)
        lidar_df  = pd.DataFrame(data=cap_data['lidar'],  columns=LIDAR_COLS)

        if include_images:
            camera_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_camera.csv'), index=False)

        if include_lidar:
            lidar_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_lidar.csv'), index=False)

        if index_by_lidar:
            target_df = lidar_df
        else:
            target_df = camera_df

        if len(target_df['timestamp']):
            # Interpolate samples from all used sensors to camera/lidar frame timestamps
            target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
            target_df.set_index(['timestamp'], inplace=True)
            target_df.index.rename('index', inplace=True)
            target_index_df = pd.DataFrame(index=target_df.index)

            collection = TrackletCollection()

            if 'front_rtk' in cap_data and 'rear_rtk' in cap_data:
                tracklets = process_rtk_data(
                    bs, cap_data, obs_data, target_index_df, outdir)
                collection.tracklets += tracklets

            if 'base_link_pose' in cap_data:
                tracklets = process_pose_data(
                    bs, cap_data, obs_data, target_index_df, outdir)
                collection.tracklets += tracklets

            if collection.tracklets:
                tracklet_path = os.path.join(outdir, 'tracklet_labels.xml')
                collection.write_xml(tracklet_path)
        else:
            print('Warning: No camera image times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')

        '''

        cap_rear_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_gps.csv'), index=False)
        cap_front_gps_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_gps.csv'), index=False)
        cap_rear_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk.csv'), index=False)
        cap_front_rtk_df.to_csv(os.path.join(dataset_outdir, 'capture_vehicle_front_rtk.csv'), index=False)

        obs_rtk_df_dict = {}
        for obs_topic, obs_rtk_dict in obstacle_rtk_dicts.items():
            obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
            obs_rtk_df = pd.DataFrame(data=obs_rtk_dict, columns=rtk_cols)
            if not len(obs_rtk_df.index):
                print('Warning: No entries for obstacle %s in %s. Skipping.' % (obs_name, bs.name))
                continue
            obs_rtk_df.to_csv(os.path.join(dataset_outdir, '%s_rtk.csv' % obs_prefix), index=False)
            obs_rtk_df_dict[obs_topic] = obs_rtk_df

        if index_by_lidar:
            target_dict = lidar_dict
            target_df   = lidar_df
        else:
            target_dict = camera_dict
            target_df   = camera_df

        if len(target_dict['timestamp']):
            # Interpolate samples from all used sensors to index frame timestamps
            target_df['timestamp'] = pd.to_datetime(target_df['timestamp'])
            target_df.set_index(['timestamp'], inplace=True)
            target_df.index.rename('index', inplace=True)

            target_index_df = pd.DataFrame(index=target_df.index)

            cap_rear_gps_interp = interpolate_to_target(target_index_df, cap_rear_gps_df, filter_cols=gps_cols)
            cap_rear_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_gps_interp.csv'), header=True)

            cap_front_gps_interp = interpolate_to_target(target_index_df, cap_front_gps_df, filter_cols=gps_cols)
            cap_front_gps_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_gps_interp.csv'), header=True)

            cap_rear_rtk_interp = interpolate_to_target(target_index_df, cap_rear_rtk_df, filter_cols=rtk_cols)
            cap_rear_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_rear_rtk_interp.csv'), header=True)
            cap_rear_rtk_interp_rec = cap_rear_rtk_interp.to_dict(orient='records')

            cap_front_rtk_interp = interpolate_to_target(target_index_df, cap_front_rtk_df, filter_cols=rtk_cols)
            cap_front_rtk_interp.to_csv(
                os.path.join(dataset_outdir, 'capture_vehicle_front_rtk_interp.csv'), header=True)
            cap_front_rtk_interp_rec = cap_front_rtk_interp.to_dict(orient='records')

            if not obs_rtk_df_dict:
                print('Warning: No obstacles or obstacle RTK data present. '
                      'Skipping Tracklet generation for %s.' % bs.name)
                continue

            collection = TrackletCollection()
            for obs_topic in obstacle_rtk_dicts.keys():
                obs_rtk_df = obs_rtk_df_dict[obs_topic]
                obs_interp = interpolate_to_target(target_index_df, obs_rtk_df, filter_cols=rtk_cols)
                obs_prefix, obs_name = obs_prefix_from_topic(obs_topic)
                obs_interp.to_csv(
                    os.path.join(dataset_outdir, '%s_rtk_interpolated.csv' % obs_prefix), header=True)

                # Plot obstacle and front/rear rtk paths in absolute RTK ENU coords
                fig = plt.figure()
                plt.plot(
                    obs_interp['tx'].tolist(),
                    obs_interp['ty'].tolist(),
                    cap_front_rtk_interp['tx'].tolist(),
                    cap_front_rtk_interp['ty'].tolist(),
                    cap_rear_rtk_interp['tx'].tolist(),
                    cap_rear_rtk_interp['ty'].tolist())
                fig.savefig(os.path.join(dataset_outdir, '%s-%s-plot.png' % (bs.name, obs_name)))
                plt.close(fig)

                # Extract lwh and object type from CSV metadata mapping file
                md = bs.metadata if bs.metadata else default_metadata
                if not bs.metadata:
                    print('Warning: Default metadata used, metadata.csv file should be with .bag files.')
                for x in md:
                    if x['obstacle_name'] == obs_name:
                        mdr = x

                obs_tracklet = Tracklet(
                    object_type=mdr['object_type'], l=mdr['l'], w=mdr['w'], h=mdr['h'], first_frame=0)

                # NOTE these calculations are done in obstacle oriented coordinates. The LWH offsets from
                # metadata specify offsets from lower left, rear, ground corner of the vehicle. Where +ve is
                # along the respective length, width, height axis away from that point. They are converted to
                # velodyne/ROS compatible X,Y,Z where X +ve is forward, Y +ve is left, and Z +ve is up.
                lrg_to_gps = [mdr['gps_l'], -mdr['gps_w'], mdr['gps_h']]
                lrg_to_centroid = [mdr['l'] / 2., -mdr['w'] / 2., mdr['h'] / 2.]
                gps_to_centroid = np.subtract(lrg_to_centroid, lrg_to_gps)

                # compute obstacle yaw based on movement, and fill out missing gaps where obstacle moves too little
                obs_rear_rtk_diff  = obs_interp.diff()
                obs_moving  = (obs_rear_rtk_diff.tx ** 2 + obs_rear_rtk_diff.ty ** 2) >= (0.1 ** 2)
                obs_rear_rtk_diff.loc[~obs_moving, 'tx':'ty'] = None
                obs_yaw_computed = np.arctan2(obs_rear_rtk_diff['ty'], obs_rear_rtk_diff['tx'])
                obs_yaw_computed = obs_yaw_computed.fillna(method='bfill').fillna(method='ffill').fillna(value=0.)

                # Convert NED RTK coords of obstacle to capture vehicle body frame relative coordinates
                obs_tracklet.poses = estimate_obstacle_poses(
                    cap_front_rtk=cap_front_rtk_interp_rec,
                    #cap_front_gps_offset=[0.0, 0.0, 0.0],
                    cap_rear_rtk=cap_rear_rtk_interp_rec,
                    #cap_rear_gps_offset=[0.0, 0.0, 0.0],
                    obs_rear_rtk=obs_interp.to_dict(orient='records'),
                    obs_rear_gps_offset=gps_to_centroid,
                    obs_yaw = obs_yaw_computed
                )

                collection.tracklets.append(obs_tracklet)
                # end for obs_topic loop

            tracklet_path = os.path.join(dataset_outdir, 'tracklet_labels.xml')
            collection.write_xml(tracklet_path)
        else:
            print('Warning: No camera/lidar times were found. '
                  'Skipping sensor interpolation and Tracklet generation.')
                          '''



if __name__ == '__main__':
    main()
