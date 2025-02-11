import warnings
warnings.filterwarnings("ignore")

import argparse
import uproot
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
import random

parser = argparse.ArgumentParser("Creating labels and seeds")
parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load entire dataset for all features in one go
print("Loading files...")
with uproot.open(args.data) as f:
    datatree = f['tree']

num_evts = datatree.num_entries

# Preload arrays for all events for faster access in loops
trk_data = {feat: datatree[feat].array() for feat in trk_features}
sig_ind_array = datatree['sig_ind'].array()
sig_flag_array = datatree['sig_flag'].array()
bkg_flag_array = datatree['bkg_flag'].array()
bkg_ind_array = datatree['bkg_ind'].array()
seed_array = datatree['seed_ind'].array()
SV_ind_array = datatree['SVtrk_ind'].array()
had_pt_array = datatree['had_pt'].array()

def create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, 
                   seed_array, SV_ind_array, had_pt_array, trk_features, nevts=3):

    had_objects = []

    if (int(args.end) == -1): end = num_evts
    else: end = int(args.end)

    for evt in range(int(args.start), end):
        print(evt)
        evt_features = {f: trk_data[f][evt] for f in trk_features}

        #bins = [10, 20, 30, 40, 50]
        #had_weights = [4, 3, 2, 1] #10to20, 20to30, 30to40, 40to50

        for had in np.unique(sig_flag_array[evt]):
            sig_inds = sig_ind_array[evt][sig_flag_array[evt] == had]
            bkg_inds = list(set(bkg_ind_array[evt][bkg_flag_array[evt] == had]) - set(sig_inds))
            comb_inds = list(sig_inds) + bkg_inds
            feature_matrix = np.vstack([np.array([evt_features[f][int(ind)] for ind in comb_inds]) for f in trk_features]).T

            hadron_pt = had_pt_array[evt][had]

            had_nan_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[had_nan_mask]
            val_comb_inds = np.array(comb_inds)[had_nan_mask]

            sig_inds = [i for i, ind in enumerate(val_comb_inds) if ind in sig_inds]

            if(len(sig_inds) < 3): continue

            bkg_inds = [i for i, ind in enumerate(val_comb_inds) if ind in bkg_inds]
            labels = np.zeros(len(sig_inds) + len(bkg_inds))
            labels[:len(sig_inds)] = 1

            shuffled_inds = np.random.permutation(len(labels))
            feature_matrix = feature_matrix[shuffled_inds]
            labels = labels[shuffled_inds]
            #sig_inds = [shuffled_inds.tolist().index(i) for i in sig_inds]
            #bkg_inds = [shuffled_inds.tolist().index(i) for i in bkg_inds]

            num_nodes = feature_matrix.shape[0]

            max_edges = num_nodes *(num_nodes -1) // 2

            num_edges = max_edges // 2

            source = np.random.randint(0, num_nodes, num_edges)
            target = np.random.randint(0, num_nodes, num_edges)

            edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.int32)

            hadron_weight = 1  # Default weight

            #for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
             #   if lower <= hadron_pt < upper:
             #       hadron_weight = had_weights[i]
              #      break

            had_data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float),
                y=torch.tensor(labels, dtype=torch.float),
                edge_index=edge_index,
                had_weight=torch.tensor([hadron_weight], dtype=torch.float)
            )
            had_objects.append(had_data)

    return had_objects

print("Creating hadron training objects...")
had_data = create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array,
                                    seed_array, SV_ind_array, had_pt_array, trk_features)
print(f"Saving had_data to haddata_{args.save_tag}.pkl...")
with open("haddata_" + args.save_tag + ".pkl", 'wb') as f:
    pickle.dump(had_data, f)

