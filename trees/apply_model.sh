#!/usr/bin/env bash

python trees/apply_model.py --input_las /home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub/real_data/Trees/tree_006/ScanPos005.laz \
    --spg_model_path /home/boissieu/git/superpoint_graph/results_partition/trunkBranchLeaf_sub/best/1/model.pth.tar \
    --seg_model_path /home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar

python trees/apply_model.py --input_las /home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub/real_data/Trees/tree_006/ScanPos006.laz \
    --spg_model_path /home/boissieu/git/superpoint_graph/results_partition/trunkBranchLeaf_sub/best/1/model.pth.tar \
    --seg_model_path /home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar

python trees/apply_model.py --input_las /home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub/real_data/Trees/tree_022/ScanPos003.laz \
    --spg_model_path /home/boissieu/git/superpoint_graph/results_partition/trunkBranchLeaf_sub/best/1/model.pth.tar \
    --seg_model_path /home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar

python trees/apply_model.py --input_las /home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub/real_data/Trees/tree_022/ScanPos004.laz \
    --spg_model_path /home/boissieu/git/superpoint_graph/results_partition/trunkBranchLeaf_sub/best/1/model.pth.tar \
    --seg_model_path /home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar

