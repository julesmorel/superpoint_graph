# This script writes the segmented point clouds
# The output file is an .xyz file with columns:
# - x, y, z,
# - labels: original class
# - objects: object number defined by graph cut computed by graph_processing
# - elevation, linearity, planarity, scattering, verticality computed by graph_processing
# - pred: prediction


import glob
import os
import numpy as np
import pandas as pd
from partition.provider import *

ROOT_PATH='../data/trunkBranchLeaf_sub'
ROOT_PATH = os.path.realpath(ROOT_PATH)


fea_dir = os.path.join(ROOT_PATH, 'features_supervision')
spg_dir = os.path.join(ROOT_PATH, 'superpoint_graphs')
res_dir = os.path.realpath('../results/trunkBranchLeaf_sub')
out_dir = os.path.join(ROOT_PATH, 'clouds')
file_list = glob.glob(spg_dir+'/**/*.h5', recursive=True)

for f in file_list:
    dname = os.path.basename(os.path.dirname(f))
    fname = os.path.basename(f)[:-3]
    fold = int(dname)
    fea_file = os.path.join(fea_dir, dname, fname+'.h5')
    obj_file = os.path.join(fea_dir, dname, fname + '_objects.xyz')
    spg_file = os.path.join(spg_dir, dname, fname+'.h5')
    res_file = os.path.join(res_dir, 'best', 'cv'+str(fold), 'predictions_test.h5')
    out_file = os.path.join(out_dir, dname, fname+'.xyz')
    if os.path.isfile(fea_file) and os.path.isfile(spg_file) and os.path.isfile(res_file):
        data = pd.read_csv(obj_file, sep='\t')
        geof, xyz, rgb, graph_nn, labels = read_features(fea_file)
        graph_spg, components, in_component = read_spg(spg_file)
        try:
            res = h5py.File(res_file, 'r')
            pred_red = np.array(res.get(dname+'/'+fname))
            if (len(pred_red) != len(components)):
                raise ValueError("It looks like the spg is not adapted to the result file")
            pred_full = reduced_labels2full(pred_red, components, len(xyz))
        except OSError:
            raise ValueError("%s does not exist in %s" % (folder + file_name, res_file))


        data['pred']=pred_full
        if not os.path.exists(os.path.dirname(out_file)):
            os.mkdir(os.path.dirname(out_file))
        print('Writing file: '+out_file)
        data.to_csv(out_file, sep='\t', index=False)
