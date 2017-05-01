"""Provides 'raw', which loads and parses raw KITTI data."""

import datetime as dt
import glob
import os
from collections import namedtuple

import numpy as np

#import pykitti.utils as utils


class raw:
    """Load and parse raw data into a usable format."""

    def __init__(self, base_path, date, drive, frame_range=None):
        """Set the path."""
        self.path = os.path.join(base_path, date, drive)
        self.frame_range = frame_range
        self.velo = []

    def _load_calib_rigid(self, filename):
        assert False

    def _load_calib_cam_to_cam(self, velo_to_cam_file, cam_to_cam_file):
        assert False

    def load_calib(self):
        self.calib = None
        return # TODO

    def load_timestamps(self):
        assert False

    def _poses_from_oxts(self, oxts_packets):
        assert False

    def load_oxts(self):
        assert False

    def load_gray(self, **kwargs):
        assert False

    def load_rgb(self, **kwargs):
        assert False

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_path = os.path.join(self.path, 'lidar', '*.npy')
        velo_files = sorted(glob.glob(velo_path))

        # Subselect the chosen range of frames, if any
        if self.frame_range:
            velo_files = [velo_files[i] for i in self.frame_range]

        for velo_file in velo_files:
            self.velo.append(np.load(velo_file))