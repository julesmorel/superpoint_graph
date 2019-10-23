#!/usr/bin/env bash
export TBL_DIR=/home/boissieu/git/superpoint_graph/data/trunkBranchLeaf_sub

echo "Building the point cloud graph..."
python ./supervized_partition/graph_processing.py --ROOT_PATH $TBL_DIR

for FOLD in 1 2 3 4 5; do \
    echo "Computing the supervised partition for fold ${FOLD}..."
    python ./supervized_partition/supervized_partition.py --ROOT_PATH $TBL_DIR  --cvfold $FOLD \
    --batch_size 4 --nworkers 4 \
    --odir results_partition/trunkBranchLeaf_sub/best; \
done

echo "Parsing dataset graph..."
python learning/trunkbranchleaf_dataset.py --CUSTOM_SET_PATH $TBL_DIR --supervized_partition 1

for FOLD in 1 2 3 4 5; do \
    echo "Learning for fold ${FOLD}..."
    # batch_size must be a divider of number of training files: e.g. 4 for 32 training files will result in 8 iterations, if 5 only 30 training files would be taken into account to make 6 iterations
	python ./learning/main.py --dataset trunkbranchleaf --CUSTOM_SET_PATH $TBL_DIR --batch_size 4 \
  --cvfold $FOLD --epochs 250 --lr_steps '[150,200]' --model_config "gru_10_0,f_3" --ptn_nfeat_stn 8 \
  --nworkers 4 --spg_augm_order 5 --pc_attribs xyzelpsv --spg_augm_hardcutoff 768 --ptn_minpts 10 \
  --use_val_set 0 --odir results/trunkBranchLeaf_sub/best/cv$FOLD; \
    done;


#for FOLD in 1; do \
#    echo "Rerun fold ${FOLD} that failed previously..."
#    # batch_size must be a divider of number of training files: e.g. 4 for 32 training files will result in 8 iterations, if 5 only 30 training files would be taken into account to make 6 iterations
#	python ./learning/main.py --dataset trunkbranchleaf --CUSTOM_SET_PATH $TBL_DIR --batch_size 4 \
#  --cvfold $FOLD --epochs 115 --lr_steps '[150,200]' --model_config "gru_10_0,f_3" --ptn_nfeat_stn 8 \
#  --nworkers 8 --spg_augm_order 5 --pc_attribs xyzelpsv --spg_augm_hardcutoff 768 --ptn_minpts 10 \
#  --use_val_set 0 --odir results/trunkBranchLeaf_sub/best/cv$FOLD \
#  --resume '/home/boissieu/git/superpoint_graph/results/trunkBranchLeaf_sub/best/cv1/model.pth.tar';
#    done;

