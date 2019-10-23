#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:16:14 2018

@author: landrieuloic
""""""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Template file for processing custome datasets
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg


def get_datasets(args, test_seed_offset=0):
    """build training and testing set"""

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    valid_names = [] # no validation dataset for the moment
    # if args.db_test_name == 'test' then the test set is the evaluation set
    # otherwise it serves as valdiation set to select the best epoch
    # train set
    for n in range(1, 6):
        if n != args.cvfold:
            path = '{}/superpoint_graphs/0{:d}/'.format(args.CUSTOM_SET_PATH, n)
            for fname in sorted(os.listdir(path)):
                if fname.endswith(".h5") and not (args.use_val_set and fname in valid_names):
                    # training set
                    if args.CUSTOM_SET_PATH == '/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub.old':
                        G = spg.spg_reader_debug(args, path + fname, True)
                    else:
                        G = spg.spg_reader(args, path + fname, True)
                    trainlist.append(G)
    # test set
    path = '{}/superpoint_graphs/0{:d}/'.format(args.CUSTOM_SET_PATH, args.cvfold)
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            if args.CUSTOM_SET_PATH == '/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub.old':
                testlist.append(spg.spg_reader_debug(args, path + fname, True))
            else:
                testlist.append(spg.spg_reader(args, path + fname, True))



     # Normalize edge features
    if args.spg_attribs01:
       trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist)

    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                   functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH,
                                                     test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset)) ,\
            scaler

def get_info(args):
    inv_class_map = {0:'trunk', 1:'branch', 2:'leaf'} #etc...
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    if args.loss_weights == 'none':
        weights = np.ones((len(inv_class_map),),dtype='f4')
    else:
        weights = h5py.File(args.S3DIS_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(5) if i != args.cvfold-1]].sum(1)
        weights = weights.mean()/weights
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)

    return {
        'node_feats': 8 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'class_weights': weights,
        'classes': len(inv_class_map),
        'inv_class_map': inv_class_map,
    }

def preprocess_pointclouds(SEMA3D_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['01', '02', '03', '04', '05']:
        pathP = '{}/parsed/{}/'.format(SEMA3D_PATH, n)
        pathD = '{}/features_supervision/{}/'.format(SEMA3D_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(SEMA3D_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(0)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                # rgb = f['rgb'][:].astype(np.float)
                if not args.supervized_partition:
                    lpsv = f['geof'][:]
                    lpsv -= 0.5 #normalize
                else:
                    lpsv = np.stack([f["geof"][:] ]).squeeze()
                # rescale to [-0.5,0.5]; keep xyz

                # rescale to [-0.5,0.5]; keep xyz
                #warning - to use the trained model, make sure the elevation is comparable
                #to the set they were trained on
                #i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
                # and -0.5 for floor and 0.5 for ceiling for s3dis
                e = f['elevation'][:]/30 -0.5# (rough guess) #adapt

                # Expected P columns are xyzrgbelpsvXYZd in this order. They should all be present (see spg.load_superpoint)
                rgb=np.zeros(shape=(xyz.shape[0], 3))
                rgb[:]=np.nan
                # XYZd=np.zeros(shape=(xyz.shape[0], 4))
                # XYZd[:]=np.nan
                P = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
def parse_spg(feature_file, spg_file, parsed_file, supervized_partition=1):
    f = h5py.File(feature_file, 'r')
    xyz = f['xyz'][:]
    # rgb = f['rgb'][:].astype(np.float)
    if not supervized_partition:
        lpsv = f['geof'][:]
        lpsv -= 0.5  # normalize
    else:
        lpsv = np.stack([f["geof"][:]]).squeeze()
    # rescale to [-0.5,0.5]; keep xyz

    # rescale to [-0.5,0.5]; keep xyz
    # warning - to use the trained model, make sure the elevation is comparable
    # to the set they were trained on
    # i.e. ~0 for roads and ~0.2-0.3 for builings for sema3d
    # and -0.5 for floor and 0.5 for ceiling for s3dis
    e = f['elevation'][:] / 30 - 0.5  # (rough guess) #adapt

    # Expected P columns are xyzrgbelpsvXYZd in this order. They should all be present (see spg.load_superpoint)
    rgb = np.zeros(shape=(xyz.shape[0], 3))
    rgb[:] = np.nan
    # XYZd=np.zeros(shape=(xyz.shape[0], 4))
    # XYZd[:]=np.nan
    P = np.concatenate([xyz, rgb, e[:, np.newaxis], lpsv], axis=1)

    f = h5py.File(spg_file, 'r')
    numc = len(f['components'].keys())

    with h5py.File(parsed_file, 'w') as hf:
        for c in range(numc):
            idx = f['components/{:d}'.format(c)][:].flatten()
            if idx.size > 10000:  # trim extra large segments, just for speed-up of loading time
                ii = random.sample(range(idx.size), k=10000)
                idx = idx[ii]

            hf.create_dataset(name='{:d}'.format(c), data=P[idx, ...])

def loader(entry, train, args, parsed_file, test_seed_offset=0):
    """ Prepares a superpoint graph (potentially subsampled in training) and associated superpoints. """
    G, fname = entry
    # 1) subset (neighborhood) selection of (permuted) superpoint graph
    if train:
        if 0 < args.spg_augm_hardcutoff < G.vcount():
            perm = list(range(G.vcount())); random.shuffle(perm)
            G = G.permute_vertices(perm)

        if 0 < args.spg_augm_nneigh < G.vcount():
            G = spg.random_neighborhoods(G, args.spg_augm_nneigh, args.spg_augm_order)

        if 0 < args.spg_augm_hardcutoff < G.vcount():
            G = spg.k_big_enough(G, args.ptn_minpts, args.spg_augm_hardcutoff)

    # 2) loading clouds for chosen superpoint graph nodes
    clouds_meta, clouds_flag = [], [] # meta: textual id of the superpoint; flag: 0/-1 if no cloud because too small
    clouds, clouds_global = [], [] # clouds: point cloud arrays; clouds_global: diameters before scaling

    for s in range(G.vcount()):
        cloud, diam = spg.load_superpoint(args, parsed_file, G.vs[s]['v'], train, test_seed_offset)
        if cloud is not None:
            clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v'])); clouds_flag.append(0)
            clouds.append(cloud.T)
            clouds_global.append(diam)
        else:
            clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v'])); clouds_flag.append(-1)

    clouds_flag = np.array(clouds_flag)
    clouds = np.stack(clouds)
    clouds_global = np.concatenate(clouds_global)

    return np.array(G.vs['t']), G, clouds_meta, clouds_flag, clouds, clouds_global


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_SET_PATH', default='/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf')
    parser.add_argument('--supervized_partition', type=int, default=1)
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)


