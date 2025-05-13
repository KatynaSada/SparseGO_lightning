"""
 This script computes the correlation between the real AUDRC and the predicted AUDRC for each drug on an individual basis.
 In the case of multiple models due to k-fold cross-validation, an average correlation is derived.
 It also computes the density plot of all models and its metrics. 
"""
import ai4clinic
from ai4clinic.graphs import best2worst2
from ai4clinic.graphs import drugs2waterfall
from ai4clinic.graphs import preds2scatter

import argparse
import sys
import torch
import torch.nn as nn
import torch.utils.data as du
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import statistics
import wandb
import os 
sys.path.append("../code")
import utils
from utils import *
from train_cv import SparseGODataModule, SparseGO

from lightning.pytorch import Trainer

def get_compound_names(file_name):
    compounds = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            compounds.append([tokens[1], tokens[2]])

    return compounds

if __name__ == "__main__":
    mac = "/Users/katyna/Library/CloudStorage/OneDrive----/"
    windows = "C:/CodeTFG/"
    manitou = "/manitou/pmg/users/ks4237/" 
    computer = mac # CHANGE

    parser = argparse.ArgumentParser(description='SparseGO metrics')

    parser.add_argument('-project', help='W&B project name', type=str, default="TFG_Pruebas")
    parser.add_argument('-entity', help='W&B entity', type=str, default="elinarestov-universidad-de-navarra")
    parser.add_argument('-tags', help='Tags of type of data or/and model we are testing', type=list_of_strings, default=['ChEMBL500_1_5'])
    parser.add_argument('-job_type', help='Job type', type=str, default="job-test")
    parser.add_argument('-sweep_name', help='Job type', type=str, default="final metrics drugs")

    parser.add_argument('-input_type', help='Type of omics data used', type=str, default="mutations")
    parser.add_argument('-split_type', help='Type of data split used (e.g., cell-blind)', type=str, default="cell-blind")
    parser.add_argument('-input_folder', help='Directory containing the input data folders.', type=str, default=computer + "data/")
    parser.add_argument('-output_folder', help='Directory containing the folders that have the resulting models', type=str, default=computer + "results/")
    parser.add_argument('-samples_folders', help='Folders to analyze', type=list_of_strings, default=["samples1","samples2","samples3","samples4","samples4"])
    
    parser.add_argument('-predictions_name', help='Which results to use for the density plot', type=str, default="d_test_predictions.txt")
    parser.add_argument('-labels_name', help='Which results to use for the density plot', type=str, default="sparseGO_test.txt")
    parser.add_argument('-drugs_names', help='Drugs names and SMILEs', type=str, default="compound_names.txt")
    parser.add_argument('-drugs_fake_lelo', help='Drugs used in fake LELO (only if needed)', required=False, type=list_of_strings)
    parser.add_argument('-log_artifact', help='Log artifact', type=bool, default=True)
    
    opt, unknown = parser.parse_known_args()

    # Resulting figures are uploaded to w&b
    log_artifact = opt.log_artifact
    
    if opt.drugs_fake_lelo:
        hyperparameters = {
            "input_type": opt.input_type,    # Adding input_type as a hyperparameter
            "drugs_fake_lelo": opt.drugs_fake_lelo
            }
    else:
        hyperparameters = {
        "input_type": opt.input_type    # Adding input_type as a hyperparameter
        }
    
    run = wandb.init(project=opt.project, entity=opt.entity, name=opt.sweep_name,tags=opt.tags, job_type=opt.job_type, config=hyperparameters)

    input_type = opt.input_type
    split_type = opt.split_type
    input_folder = opt.input_folder
    output_folder = opt.output_folder
    samples_folders = opt.samples_folders

    labels_name = opt.labels_name
    predictions_name = opt.predictions_name

    drugs_names = opt.drugs_names

    compound_names = get_compound_names(input_folder + "allsamples/" + drugs_names)
    compound_names.pop(0)

   
    # Iterate over each fold in the cross-validation
    for fold in samples_folders:  # e.g., "samples1", "samples2", etc.
        input_path = input_folder + fold + "/"
        output_path = output_folder + fold + "/"
        print(f"Processing fold: {fold}")
        
        fold_data = pd.read_csv(input_path + labels_name, delimiter="\t", header=None)
        fold_predictions = np.loadtxt(output_path + predictions_name, delimiter="\t")
        fold_labels = fold_data[2].values  # Convert to NumPy array
        fold_drugs = fold_data[1].values   # Convert to NumPy array
        fold_cells = fold_data[0].values   # Convert to NumPy array
        
        # Create tensor of fold names repeated to match predictions length
        fold_names = torch.full((len(fold_predictions),), int(fold.replace('samples', '')), dtype=torch.int)
        
        # If this is the first fold, initialize the all_folds lists
        if fold == samples_folders[0]:
            all_predictions = fold_predictions
            all_labels = fold_labels
            all_drugs = fold_drugs
            all_cells = fold_cells
            all_fold_names = fold_names
        else:
            # Concatenate using numpy for arrays and torch for tensors
            all_predictions = np.concatenate([all_predictions, fold_predictions])
            all_labels = np.concatenate([all_labels, fold_labels])
            all_drugs = np.concatenate([all_drugs, fold_drugs])
            all_cells = np.concatenate([all_cells, fold_cells])
            all_fold_names = torch.cat([all_fold_names, fold_names], dim=0)

    # Convert the compound_names list to a dictionary
    smiles_to_name = {smiles: name for smiles, name in compound_names}
    # Convert all_drugs SMILES to names
    all_drug_names = [smiles_to_name.get(smiles, "Unknown Compound") for smiles in all_drugs]

    scatter_metrics = preds2scatter(all_predictions, all_labels, all_cells, all_fold_names,
                    output_path = f"{output_folder}figures/density_{input_type}_{split_type}.png",
                    density_bins=(90, 90),
                    cmap='turbo',
                    marker='.',
                    marker_size=6,
                    best_fit_line=True,
                    title='', title_fontsize=20, # Density Scatter Plot: Random Cross-Validation
                    xlabel='Real Response (AUDRC)', xlabel_fontsize=20,
                    ylabel='Predicted Response (AUDRC)', ylabel_fontsize=20,
                    display_plot=False,
                    verbose=True,
                    show_legend=True,
                    legend_position=(0.46, 0.25),  # default outside to the right
                    annotation_fontsize=17,
                    transparent_bg=True,
                    plot_size=(7, 5),
                    x_range=(-0.01,1.1), y_range=(0,0.7),
                    xtick_fontsize=17, ytick_fontsize=17
                    )
    run.log({"Average pearson cor:": scatter_metrics['average_pearson']})
    run.log({"Average spearman cor:": scatter_metrics['average_spearman']})
    run.log({"Overall pearson cor:": scatter_metrics['overall_pearson']})
    run.log({"Overall spearman cor:": scatter_metrics['overall_spearman']})
    
    artifact = wandb.Artifact("preds2scatter",type="graphic")
    artifact.add_file(f"{output_folder}figures/density_{input_type}_{split_type}.png")
    if log_artifact: run.log_artifact(artifact)

    metrics = best2worst2(all_predictions, all_labels, all_drug_names, all_cells, all_fold_names,
                    plot_size=(11, 8),
                    corr_metric='spearman',
                    num_select=2,
                    output_path=f"{output_folder}figures/{input_type}_{split_type}.png",
                    marker='.',
                    marker_size=7,
                    best_fit_line=True,
                    title='', # Scatter Plot with Drugs: Random Cross-Validation
                    title_fontsize=40, 
                    xlabel='Real Response (AUDRC)',
                    xlabel_fontsize=30,
                    ylabel='Predicted Response (AUDRC)',
                    ylabel_fontsize=30,
                    annotation_fontsize=30,
                    worst_color_hex=None,
                    best_color_hex=None,
                    display_plot=False,
                    verbose=True,
                    show_metrics=False,
                    show_legend=True,
                    legend_position=(2, 0.5),  # Moved further right to place outside
                    x_range=(-0.01,1.1),
                    y_range=(0,0.7),
                    xtick_fontsize=25,
                    ytick_fontsize=25,
                    transparent_bg=True)
    artifact = wandb.Artifact("best2worst2",type="graphic")
    artifact.add_file(f"{output_folder}figures/{input_type}_{split_type}.png")
    if log_artifact: run.log_artifact(artifact)

    mean_corr, drug_corrs = drugs2waterfall(all_predictions, all_labels, all_drug_names, all_cells, all_fold_names, 
                        output_path=f"{output_folder}figures/waterfall_{input_type}_{split_type}.png",
                        corr_metric='spearman', num_select=10, 
                        mark_threshold=0.5, color=None, display_plot=False,
                        percentage_position=(0.17, 0.5), percentage_fontsize=40, # (0.3, 0.5)
                        ylabel='Spearman Correlation', ylabel_fontsize=25, 
                        xlabel='', xlabel_fontsize=20, 
                        title='', title_fontsize=20,
                        ax2_title="",
                        ytick_fontsize=22, transparent_bg=True,
                        bar_annotation_fontsize=26, drug_name_fontsize=24, # bar_annotation_fontsize=23, drug_name_fontsize=16
                        plot_size=(13, 7),
                        ax2_ylim=(0.03,0.9), # (-0.13,0.9)
                        legend_position=(1,1),
                        legend_fontsize=15,
                        legend=False,
                        )
    run.log({"Average spearman correlations of all drugs:": mean_corr})
        
    artifact = wandb.Artifact("drugs2waterfall",type="graphic")
    artifact.add_file(f"{output_folder}figures/waterfall_{input_type}_{split_type}.png")
    if log_artifact: run.log_artifact(artifact)
    
    run.finish()