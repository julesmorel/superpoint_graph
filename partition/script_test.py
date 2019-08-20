import os.path
import numpy as np
import argparse
import sys

sys.path.append("/home/jules/Project/superpoint_graph/partition")
from provider import *
spg_file='/home/jules/Project/superpoint_graph/test/superpoint_graphs/test/full5_0_5.000_1.5.h5'
ply_file='/home/jules/Project/superpoint_graph/test/superpoint_graphs/test/full5_0_5.000_1.5.5'
fea_file='/home/jules/Project/superpoint_graph/test/features_supervision/test/full5_0_5.000_1.5.h5'
geof, xyz, graph_nn, labels = read_features(fea_file)
graph_spg, components, in_component = read_spg(spg_file)
spg2ply(ply_file + "_spg.ply", graph_spg)
partition2ply(ply_file + "_partition.ply", xyz, components)
