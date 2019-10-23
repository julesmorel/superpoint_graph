# This scripts aims at applying model on new data

import os
import sys
import numpy as np
import laspy
import glob
import numpy as np
import h5py
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import transforms3d
import math
import igraph
import argparse
from timeit import default_timer as timer
import torchnet as tnt
import functools
import argparse
from sklearn.linear_model import RANSACRegressor
import pandas as pd


# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = '/home/boissieu/git/superpoint_graph/trees'
sys.path.insert(0, os.path.join(DIR_PATH, '..'))
sys.path.append(os.path.join(DIR_PATH,"../partition/cut-pursuit/build/src"))

from partition.ply_c import libply_c
from supervized_partition.graph_processing import compute_graph_nn_2, write_structure, graph_loader, graph_collate, read_spg
from partition.provider import read_las, pcfeatures2ascii, write_spg, partition2ply, reduced_labels2full
from supervized_partition import supervized_partition as sp
from learning import main as seg
from learning.pointnet import LocalCloudEmbedder
# from learning.pointnet import PointNet
from supervized_partition.losses import compute_dist, compute_partition
from partition.graphs import compute_sp_graph


parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')

# parser.add_argument('--ROOT_PATH', default='/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf')
# parser.add_argument('--dataset', default='trunkbranchleaf')
# parameters
parser.add_argument('--compute_geof', default=1, type=int, help='compute hand-crafted features of the local geometry')
parser.add_argument('--k_nn_local', default=20, type=int, help='number of neighbors to describe the local geometry')
parser.add_argument('--k_nn_adj', default=5, type=int, help='number of neighbors for the adjacency graph')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--plane_model', default=1, type=int, help='uses a simple plane model to derive elevation')
parser.add_argument('--use_voronoi', default=0.0, type=float,
                    help='uses the Voronoi graph in combination to knn to build the adjacency graph, useful for sparse aquisitions. If 0., do not use voronoi. If >0, then is the upper length limit for an edge to be kept. ')
parser.add_argument('--spg_model_path', default='/home/boissieu/git/superpoint_graph/results_partition/trunkBranchLeaf_sub/best/1/model.pth.tar')
parser.add_argument('--seg_model_path', default='/home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar')
parser.add_argument('--input_las', default='/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf/real_data/tree_006/ScanPos005.laz')
parser.add_argument('--ver_batch', default=5000000, type=int, help='batch size for reading large files')
args = parser.parse_args()

with open(os.path.join(os.path.dirname(args.input_las), 'cmdline.txt'), 'w') as f:
    f.write(" ".join(["'" + a + "'" if (len(a) == 0 or a[0] != '-') else a for a in sys.argv]))

data_file = os.path.realpath(args.input_las)
str_file = data_file[:-4]+'_str.h5'
xyz = read_las(data_file)
# model trained in feet instead of meters and with coordinates xzy instead of xyz
# xyz_copy = np.copy(xyz)
# xyz[:,1] = xyz_copy[:,2]
# xyz[:,2] = xyz_copy[:,1]
# xyz = xyz*3


spg_file = data_file[:-4]+'_spg.h5'

n_ver = xyz.shape[0]

### Voronoi graph
print("computing NN structure")

graph_nn, local_neighbors = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_local, voronoi=args.use_voronoi)



if (args.compute_geof):
    geof = libply_c.compute_geof(xyz, local_neighbors, args.k_nn_local).astype('float32')
    geof[:, 3] = 2. * geof[:, 3]
else:
    geof = 0

if args.plane_model:  # use a simple palne model to the compute elevation
    low_points = ((xyz[:, 2] - xyz[:, 2].min() < 0.5)).nonzero()[0]
    reg = RANSACRegressor(random_state=0).fit(xyz[low_points, :2], xyz[low_points, 2])
    elevation = xyz[:, 2] - reg.predict(xyz[:, :2])
else:
    elevation = xyz[:, 2] - xyz[:, 2].min()


ma, mi = np.max(xyz[:,:2],axis=0,keepdims=True), np.min(xyz[:,:2],axis=0,keepdims=True)
xyn = (xyz[:,:2] - mi) / (ma - mi + 1e-8) #global position


is_transition = labels = objects = np.zeros(n_ver)
pcfeatures2ascii(str_file[:-3] + "_objects.xyz", xyz, [], [], [], elevation, geof)
rgb=[]
write_structure(str_file, xyz, rgb, graph_nn, local_neighbors.reshape([n_ver, args.k_nn_local]), \
                    is_transition, labels, objects, geof, elevation, xyn)


### apply superpoint_graph model
checkpoint = torch.load(args.spg_model_path)
model = sp.create_model(checkpoint['args'])  # use original arguments, architecture can't change

model.eval()

args_sp= checkpoint['args']

test_dataset = tnt.dataset.ListDataset([str_file],
                        functools.partial(graph_loader, train=False, args=args_sp, db_path='',
                                          preloaded=False))

ptnCloudEmbedder = LocalCloudEmbedder(checkpoint['args'])
dbinfo = sp.dataset(args_sp)[0]
with torch.no_grad():
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=graph_collate,
                                         num_workers=0)
    for bidx, (fname, edg_source, edg_target, is_transition, labels, objects, clouds_data, xyz) in enumerate(loader):
        clouds, clouds_global, nei = clouds_data
        clouds_data = (clouds.to('cuda', non_blocking=True), clouds_global.to('cuda', non_blocking=True), nei)
        embeddings = ptnCloudEmbedder.run_batch(model, *clouds_data, xyz)
        diff = compute_dist(embeddings, edg_source, edg_target, args_sp.dist_type)

        pred_components, pred_in_component = compute_partition(args_sp, embeddings, edg_source, edg_target, diff, xyz)
        graph_sp = compute_sp_graph(xyz, 100, pred_in_component, pred_components, labels, dbinfo["classes"])
        write_spg(spg_file, graph_sp, pred_components, pred_in_component)
        partition2ply(spg_file[:-3] + "_partition.ply", xyz, pred_components)

### parse spg file
from learning.trunkbranchleaf_dataset import parse_spg
parsed_file = data_file[:-4]+'_parsed.h5'

parse_spg(str_file, spg_file, parsed_file, supervized_partition=1)


### apply segmentation model
seg_model_path= args.seg_model_path
# seg_model_path = '/home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv3/model.pth.tar'
checkpoint = torch.load(seg_model_path)
args_seg = checkpoint['args']
from learning import spg
from learning import trunkbranchleaf_dataset
from learning import pointnet

seg_file = data_file[:-4]+'_seg.xyz'
dbinfo = trunkbranchleaf_dataset.get_info(args_seg)

testlist = [spg.spg_reader(args_seg, spg_file, True)]
test_dataset = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                       functools.partial(trunkbranchleaf_dataset.loader, train=False, args=args_seg, parsed_file=parsed_file,
                                                         test_seed_offset=0))
model = seg.create_model(args_seg, dbinfo)
ptnCloudEmbedder = pointnet.CloudEmbedder(args_seg)

model.eval()

loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=spg.eccpc_collate,
                                     num_workers=args_seg.nworkers)
for bidx, (targets, GIs, clouds_data) in enumerate(loader):
    model.ecc.set_info(GIs, args_seg.cuda)

    embeddings = ptnCloudEmbedder.run(model, *clouds_data)
    outputs = model.ecc(embeddings)

    pred_red = np.argmax(outputs.data.cpu().numpy(), 1)


    graph_spg, components, in_component = read_spg(spg_file)
    pred_full = reduced_labels2full(pred_red, components, len(xyz))
    data = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
    data['obj']=in_component
    data['pred']=pred_full
    print('Writing segmentation results in: '+seg_file)
    data.to_csv(seg_file, sep='\t', index=False)


