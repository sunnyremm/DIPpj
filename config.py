# coding: utf-8
from os import path as osp

root = osp.dirname(osp.abspath(__file__))

# # TestA
# TRAIN_A_ROOT = os.path.join(root, 'TrainA')
# TEST_A_ROOT = os.path.join(root, 'TestA')
# TEST_B_ROOT = os.path.join(root, 'nature')

# O-Haze
OHAZE_ROOT = osp.abspath(osp.join(root, '../data', 'O-Haze'))

# RESIDE
TRAIN_ITS_ROOT = osp.abspath(osp.join(root, '../data', 'RESIDE', 'ITS_v2'))  # ITS
TEST_SOTS_ROOT = osp.abspath(osp.join(root, '../data', 'RESIDE', 'SOTS', 'nyuhaze500'))  # SOTS indoor
TEST_MINE_ROOT = osp.abspath(osp.join(root, '../data', 'Collected'))
# TEST_SOTS_ROOT = os.path.join(root, 'SOTS', 'outdoor')  # SOTS outdoor
# TEST_HSTS_ROOT = os.path.join(root, 'HSTS', 'synthetic')  # HSTS

# HazeRD
TEST_HAZERD_ROOT = osp.abspath(osp.join(root, '../data', 'HazeRD'))
