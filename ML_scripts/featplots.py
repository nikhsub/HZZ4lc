import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch_geometric
from torch_geometric.data import Data, DataLoader
import xgboost as xgb
import pickle
from tqdm import tqdm
import shap

parser = argparse.ArgumentParser("Generate per feature plots")
parser.add_argument("-f", "--file", required=True, help="Testing data file")
parser.add_argument("-m", "--model", required=True, help="Path to model")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

def analyze_with_shap(graphs, model, trk_features):
    print("Computing SHAP values...")

    # Extract feature data from graphs
    all_feats = []
    for data in tqdm(graphs):
        all_feats.append(data.x.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)  # Shape: (N_total, num_features)

    # Create SHAP explainer
    explainer = shap.Explainer(model)  # Automatically detects XGBoost model
    shap_values = explainer(all_feats)  # Compute SHAP values

    # Plot SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=all_feats, feature_names=trk_features, show=False)
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot Feature Dependence Plots
    for i, feat in enumerate(tqdm(trk_features)):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(i, shap_values.values, all_feats, feature_names=trk_features, show=False)
        plt.savefig(f"shap_dependence_{feat}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("SHAP analysis completed!")

def plot_feature_vs_xgb_score(graphs, model, track_features, bins_x=50, bins_y=50):
    all_feats = []
    all_scores = []

    for data in tqdm(graphs):
        preds = model.predict_proba(data.x.cpu().numpy())[:, 1]  # XGBoost score
        all_scores.append(preds)
        all_feats.append(data.x.cpu().numpy())
    
    all_feats = np.concatenate(all_feats, axis=0)  # Shape: (N_total, num_features)
    all_scores = np.concatenate(all_scores)        # Shape: (N_total,)

    print("Plotting...")

    feat_range = { "trk_eta" : (-2.5, 2.5),
                  "trk_phi" : (-3, 3),
                  "trk_ip2d": (-10, 10),
                  "trk_ip3d": (-10, 10),
                  "trk_ip2dsig": (-10, 10),
                  "trk_ip3dsig": (-10, 10),
                  "trk_p": (0, 200),
                  "trk_pt": (0, 100),
                  "trk_nValid": (0, 30),
                  "trk_nValidPixel": (0, 12),
                  "trk_nValidStrip": (0, 30),
                  "trk_charge": (-1, 1)
                }

    for i, feat in enumerate(tqdm(trk_features)):
        plt.figure(figsize=(8, 6))

        # Define bins for the XGBoost score (x-axis)
        xgb_bins = np.linspace(0, 1, bins_x + 1)

        feat_bins = np.linspace(feat_range[feat][0], feat_range[feat][1], bins_y + 1)

        # Use 'auto' binning for the feature (y-axis)
        hist = plt.hist2d(all_feats[:, i], all_scores, bins=[feat_bins, xgb_bins], cmap='winter', cmin=1)
        
        plt.ylabel("XGBoost Score")
        plt.xlabel(feat)
        plt.title(f"{feat} vs. XGBoost Score")

        # Colorbar
        plt.colorbar(hist[3], label="Counts")

        # Save the plot
        plt.savefig(f"{feat}_{args.savetag}.png", dpi=300, bbox_inches='tight')
        plt.close()

with open(args.model, "rb") as f:
    model = pickle.load(f)

print(f"Loading data from {args.file}...")
with open(args.file, 'rb') as f:
    graphs = pickle.load(f)

print("Plotting per feature plot...")

analyze_with_shap(graphs, model, trk_features)

#plot_feature_vs_xgb_score(graphs, model, trk_features)

#print("Plotting feature importance plot...")
#
#fig, ax = plt.subplots(figsize=(8, 6))
#xgb.plot_importance(model, importance_type='gain', ax=ax)  # Options: 'gain', 'weight', 'cover'
#
## Save the figure
#plt.savefig(f"feat_importance_{args.savetag}.png")

