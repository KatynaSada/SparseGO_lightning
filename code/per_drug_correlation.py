"""
 This script computes the correlation between the real AUDRC and the predicted AUDRC for each drug on an individual basis.
 In the case of multiple models due to k-fold cross-validation, an average correlation is derived.
 It also computes the density plot of all models and its metrics. 
"""

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
from pytorch_lightning import Trainer
from train_cv import SparseGO 


def load_mapping(mapping_file):
    """
    Opens a txt file with two columns and saves the second column as the key of the dictionary and the first column as a value.

        Parameters
        ----------
        mapping_file: str, path to txt file

        Output
        ------
        mapping: dic

        Notes: used to read gene2ind.txt, drug2ind.txt

    """
    mapping = {} # dictionary of values on required txt

    file_handle = open(mapping_file) # function opens a file, and returns it as a file object.

    for line in file_handle:
        line = line.rstrip().split() # quitar espacios al final del string y luego separar cada elemento (en gene2ind hay dos elementos 3007	ZMYND8, los pone en una lista ['3007', 'ZMYND8'] )
        mapping[line[1]] = int(line[0]) # en gene2ind el nombre del gen es el key del dictionary y el indice el valor del diccionario

    file_handle.close()

    return mapping

def build_input_vector(inputdata, cell_features, drug_features):
    # For training
    genedim = len(cell_features[0,:])
    drugdim = len(drug_features[0,:])
    feature = np.zeros((inputdata.size()[0], (genedim+drugdim)))

    for i in range(inputdata.size()[0]):
        feature[i] = np.concatenate((cell_features[int(inputdata[i,0])], drug_features[int(inputdata[i,1])]), axis=None)

    feature = torch.from_numpy(feature).float()
    return feature

def predict_short(predict_data, model, batch_size, cell_features, drug_features, device):
    """
    Predict the output labels using the provided model on the given predict_data.

    Parameters:
    predict_data (tuple): A tuple containing the features and labels for prediction.
    model (torch.nn.Module): The model used for prediction.
    batch_size (int): Batch size for prediction.
    cell_features (list): List of cell features.
    drug_features (list): List of drug features.
    device (string): Device for GPU/CPU computation.

    Returns:
    tuple: A tuple containing the Pearson correlation coefficient and Spearman correlation coefficient.
    """

    # Unpack predict_data tuple into features and labels
    predict_feature, predict_label = predict_data

    # Move predict_label to the specified device
    predict_label_device = predict_label.to(device, non_blocking=True).detach()

    # Set model to evaluation mode
    model.eval()

    # Create a DataLoader for prediction
    test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

    # Initialize test predictions
    test_predict = torch.zeros(0, 1, device=device)

    # Perform prediction
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(test_loader):
            # Build input feature vector
            features = build_input_vector(inputdata, cell_features, drug_features)
            features = features.to(device)

            # Make predictions for test data
            out = model(features)
            test_predict = torch.cat([test_predict, out])

    # Calculate Pearson and Spearman correlation coefficients
    test_corr = pearson_corr(test_predict, predict_label_device)
    test_corr_spearman = spearman_corr(test_predict.cpu().detach().numpy(), predict_label_device.cpu())

    return test_corr, test_corr_spearman

def load_selected_drug_data(file_name, cell2id, drug2id, selected_drug):
    """
    Load selected samples of a chosen drug from a specified file.

    Parameters:
    file_name (str): The name of the file to load data from.
    cell2id (dict): A dictionary mapping cell names to IDs.
    drug2id (dict): A dictionary mapping drug names to IDs.
    selected_drug (str): The name of the drug for which to select samples.

    Returns:
    tuple: A tuple containing the selected features and labels.
    """

    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            if tokens[1] == selected_drug:
                feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
                label.append([float(tokens[2])])

    return feature, label

def create_density_plot(all_predictions, list_models_pearsons, list_models_spearmans, output_path):
    """
    Create and save a density plot visualizing the relationship between real and predicted AUDRC values, and its correlations. 

    Parameters:
    all_predictions (DataFrame): A pandas DataFrame containing the predictions with columns:
                                  - "Class": The class of the prediction (e.g., "SparseGO").
                                  - "Real AUDRC": The actual AUDRC values.
                                  - "Predicted AUDRC": The predicted AUDRC values.
    output_path (str): The file path where the resulting plot will be saved (including the filename).

    Returns:
    None: This function saves the plot to the specified output path.
    
    Example usage:
    create_density_plot(all_predictions, sp_overall, pe_overall, loss_overall, sp_average, pe_average, output_folder + 'density_plot.png')

    """
    # Calculate overall correlations and loss
    # sp_overall (float): Overall Spearman correlation.
    # pe_overall (float): Overall Pearson correlation.
    # loss_overall (float): Overall mean squared error (MSE) loss.
    # sp_average (float): Average Spearman correlation.
    # pe_average (float): Average Pearson correlation.
    
    criterion=nn.MSELoss()

    pe_overall = pearson_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    pe_overall = np.around(pe_overall,4)

    sp_overall = spearman_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()).detach().numpy(),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    sp_overall = np.around(sp_overall,4)

    loss_overall = criterion(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    loss_overall = np.around(loss_overall,4)
        
    # Calculate average correlations
    pe_average = np.around(statistics.mean(list(list_models_pearsons.values())),4)
    sp_average =  np.around(statistics.mean(list(list_models_spearmans.values())),4)

    # Set figure size
    plt.rcParams['figure.figsize'] = (10, 7)

    # Extract real and predicted AUDRC values for SparseGO
    x = all_predictions.loc[all_predictions["Class"] == "SparseGO", "Real AUDRC"].to_numpy()
    y = all_predictions.loc[all_predictions["Class"] == "SparseGO", "Predicted AUDRC"].to_numpy()

    # Define histogram parameters
    bins = [1000, 1000]  # Number of bins

    # Create a 2D histogram of the data
    hh, locx, locy = np.histogram2d(x, y, bins=bins)

    # Sort the points by density for plotting
    z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(x, y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]

    # Create the scatter plot
    fig, ax = plt.subplots()
    plt.scatter(x2, y2, c=z2, cmap='turbo', marker='.', s=4)

    # Fit a line to the data
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='#333333')  # Line of best fit

    # Customize the axes
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')

    # Remove ticks
    ax.tick_params(bottom=False, left=False)

    # Add horizontal grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add labels and title
    ax.set_xlabel('Real response (AUDRC)', labelpad=18, color='#333333', fontsize=25)
    ax.set_ylabel('Predicted response (AUDRC)', labelpad=18, color='#333333', fontsize=25)
    ax.set_title('Density Plot', color='#000000', weight='bold', fontsize=30)

    # Add correlation and loss information as text annotations
    plt.text(0.45, 0.18, f"Overall Spearman corr. = {sp_overall:.3f}", fontsize=13, color='#333333', weight='bold')
    plt.text(0.45, 0.14, f"Overall Pearson corr. = {pe_overall:.3f}", fontsize=13, color='#333333', weight='bold')
    plt.text(0.45, 0.10, f"Overall MSE loss = {loss_overall:.3f}", fontsize=13, color='#333333', weight='bold')
    plt.text(0.45, 0.06, f"Average Spearman corr. = {sp_average:.3f}", fontsize=13, color='#333333', weight='bold')
    plt.text(0.45, 0.02, f"Average Pearson corr. = {pe_average:.3f}", fontsize=13, color='#333333', weight='bold')

    # Print summary statistics to console
    print(f"Overall Spearman corr. = {sp_overall:.3f}")
    print(f"Overall Pearson corr. = {pe_overall:.3f}")
    print(f"Overall MSE loss = {loss_overall:.3f}")
    print(f"Average Spearman corr. = {sp_average:.3f}")
    print(f"Average Pearson corr. = {pe_average:.3f}")

    # Make the chart fill out the figure better
    fig.tight_layout()

    # Save the plot to the specified output path
    fig.savefig(output_path, transparent=True)

def create_linear_models_plot(all_predictions, list_models_pearsons, list_models_spearmans, all_df_values, output_path):
    
    criterion=nn.MSELoss()

    pe_overall = pearson_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    pe_overall = np.around(pe_overall,4)

    sp_overall = spearman_corr(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()).detach().numpy(),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    sp_overall = np.around(sp_overall,4)

    loss_overall = criterion(torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(all_predictions.loc[all_predictions.loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy()
    loss_overall = np.around(loss_overall,4)
        
    # Calculate average correlations
    pe_average = np.around(statistics.mean(list(list_models_pearsons.values())),4)
    sp_average =  np.around(statistics.mean(list(list_models_spearmans.values())),4)
    
    all_df_values = all_df_values.dropna()
    
    # Determine selected drugs
    if len(all_df_values) >= 4:
        # Find top 2 and worst 2 drugs
        selected_drugs = all_df_values.head(2)['Name'].tolist() + all_df_values.tail(2)['Name'].tolist()

        # Define colors for selected drugs
        color_map = {
            selected_drugs[0]: 'darkgreen',
            selected_drugs[1]: 'lightgreen',
            selected_drugs[2]: 'lightcoral',
            selected_drugs[3]: 'darkred'
        }
    elif len(all_df_values) >= 2:
        # Find top 1 and worst 1 drugs
        selected_drugs = all_df_values.head(1)['Name'].tolist() + all_df_values.tail(1)['Name'].tolist()

        # Define colors for selected drugs
        color_map = {
            selected_drugs[0]: 'darkgreen',
            selected_drugs[1]: 'darkred'
        }    
    else:
        selected_drugs = []
        color_map = {}

    # Extract real and predicted AUDRC values for SparseGO
    x = all_predictions.loc[all_predictions["Class"] == "SparseGO", "Real AUDRC"].to_numpy()
    y = all_predictions.loc[all_predictions["Class"] == "SparseGO", "Predicted AUDRC"].to_numpy()
    # drug_colors = all_predictions.loc[all_predictions["Class"] == "SparseGO", "Drug"].map(color_map).to_numpy()

    # Create the scatter plot
    fig, ax = plt.subplots()
    # plt.scatter(x, y, c=drug_colors, marker='.', s=50)

    # Fit a line to the data
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color='#333333')  # Line of best fit

    for drug in all_predictions['Drug'].unique():
        # Extract data for each drug
        drug_data = all_predictions[all_predictions['Drug'] == drug]
        x = drug_data['Real AUDRC'].to_numpy()
        y = drug_data['Predicted AUDRC'].to_numpy()
        
        if drug in selected_drugs:
            plt.scatter(x, y, c=[color_map[drug]], label=drug, marker='.', s=50)
            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m * x + b, color=color_map[drug], linewidth=2)
                # Annotate with the mean value
                mean_value = all_df_values.loc[all_df_values['Name'] == drug, 'Mean'].values[0]
                plt.text(x.mean(), y.mean(), f'{mean_value:.2f}', fontsize=12, color='#000000', weight='bold', ha='right', va='bottom')
        else:
            plt.scatter(x, y, c='gray', marker='.', s=50, alpha=0.5)

    # Customize the axes
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')

    # Remove ticks
    ax.tick_params(bottom=False, left=False)

    # Add horizontal grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add labels and title
    ax.set_xlabel('Real response (AUDRC)', labelpad=18, color='#333333', fontsize=25)
    ax.set_ylabel('Predicted response (AUDRC)', labelpad=18, color='#333333', fontsize=25)
    ax.set_title('', color='#000000', weight='bold', fontsize=30)
    
    # Add correlation and loss information as text annotations
    plt.text(0.7, 0.30, f"Overall Spearman corr. = {sp_overall:.3f}", fontsize=11, color='#333333', weight='bold')
    plt.text(0.7, 0.26, f"Overall Pearson corr. = {pe_overall:.3f}", fontsize=11, color='#333333', weight='bold')
    plt.text(0.7, 0.22, f"Overall MSE loss = {loss_overall:.3f}", fontsize=11, color='#333333', weight='bold')
    plt.text(0.7, 0.18, f"Average Spearman corr. = {sp_average:.3f}", fontsize=11, color='#333333', weight='bold')
    plt.text(0.7, 0.14, f"Average Pearson corr. = {pe_average:.3f}", fontsize=11, color='#333333', weight='bold')

    # Print summary statistics to console
    print(f"Overall Spearman corr. = {sp_overall:.3f}")
    print(f"Overall Pearson corr. = {pe_overall:.3f}")
    print(f"Overall MSE loss = {loss_overall:.3f}")
    print(f"Average Spearman corr. = {sp_average:.3f}")
    print(f"Average Pearson corr. = {pe_average:.3f}")

    # Add legend outside the plot
    plt.legend(title='Drug', bbox_to_anchor=(1, 1), loc='upper left', fontsize=11)

    # Make the chart fill out the figure better
    # fig.tight_layout()

    # Display the plot
    plt.show()

    # Save the plot to the specified output path
    fig.savefig(output_path, transparent=True)

def create_waterfall_plot_correlation(all_df_values, output_path, correlation_type='pearson'):
    """
    Create and save a waterfall plot visualizing correlation values for drugs.

    Parameters:
    all_df_values (DataFrame): A pandas DataFrame containing drug names, mean correlation values, and standard deviations.
                               It should have the following columns:
                               - "Name": Names of the drugs.
                               - "Mean": Mean correlation values.
                               - "Std": Standard deviations of the correlation values.
    output_path (str): The file path where the resulting plot will be saved (including the filename).
    correlation_type (str): The type of correlation to plot ('pearson' or 'spearman'). Default is 'pearson'.

    Returns:
    None: This function saves the plot to the specified output path.
    
    Example usage:
    create_waterfall_plot_correlation(all_df_values, output_folder + 'WaterfallDrugsSparseGO_correlation.png', correlation_type='spearman')

    """

    # Set figure size
    plt.rcParams['figure.figsize'] = (12, 9)

    # Extract data from DataFrame
    drugs = all_df_values["Name"]
    rhos = all_df_values["Mean"]
    error = all_df_values["Std"]

    # Calculate the percentage of drugs with a correlation greater than 0.5
    percentage = round((sum(rhos > 50) / len(rhos)))

    # Create a bar plot
    fig, ax = plt.subplots()
    colors = ['#C9C9C9' if (x < 50) else '#B4D04F' for x in rhos]
    
    ax.bar(
        x=drugs,
        height=rhos,
        edgecolor=colors,
        color=colors,
        linewidth=1
    )
    
    # Customize the axes
    plt.xticks([])
    plt.yticks(fontsize=28)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove ticks
    ax.tick_params(bottom=False, left=False)

    # Add horizontal grid (keeping vertical grid hidden)
    ax.set_axisbelow(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    # Add labels and title
    ax.set_xlabel('Drugs', labelpad=-30, color='#333333', fontsize=50)
    ax.set_ylabel(f'{correlation_type.capitalize()} correlation', labelpad=15, color='#333333', fontsize=50)
    ax.set_title('', color='#333333', weight='bold')

    # Add percentage text
    plt.text(10, 25, str(percentage) + "%", fontsize=60, color='#000000')

    # Set y-axis limits
    plt.ylim((-10, 90))

    # Make the chart fill out the figure better
    fig.tight_layout()

    # Save the figure
    fig.savefig(output_path, transparent=True)
    
def create_top_drugs_bar_chart(all_df_values, output_path, correlation_type='pearson'):
    """
    Create and save a bar chart visualizing the top 10 drugs based on correlation values.

    Parameters:
    all_df_values (DataFrame): A pandas DataFrame containing drug names, mean correlation values, and standard deviations.
                               It should have the following columns:
                               - "Name": Names of the drugs.
                               - "Mean": Mean correlation values.
                               - "Std": Standard deviations of the correlation values.
    output_path (str): The file path where the resulting plot will be saved (including the filename).
    correlation_type (str): The type of correlation to plot ('pearson' or 'spearman'). Default is 'pearson'.

    Returns:
    None: This function saves the plot to the specified output path.
    
    Example usage:
    create_top_drugs_bar_chart(all_df_values, output_folder + 'top10sparse_spearman.png', correlation_type='spearman')

    """

    # Set figure size
    plt.rcParams['figure.figsize'] = (16, 22)
    fig, ax = plt.subplots()

    # Extract top 10 correlation values and drug names
    rhos = all_df_values["Mean"]
    drugs = all_df_values["Name"]
    rhos_top = rhos[0:10]
    drugs_top = drugs[0:10].copy()

    # Set colors based on correlation values
    colors = ['lightseagreen' if (x < 50) else '#B4D04F' for x in rhos_top]
    
    # Create the bar chart
    bars = ax.bar(
        x=drugs_top,
        height=rhos_top,
        edgecolor="none",
        linewidth=1,
        color=colors,
        width=0.9,
    )
    
    # Customize the axes
    plt.yticks([])
    plt.xticks([])

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')

    # Add a horizontal grid (keeping vertical grid hidden)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

    # Add text annotations to the top of the bars
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            round(bar.get_height(), 3),
            horizontalalignment='center',
            color='#000000',
            weight='bold',
            fontsize=80,
            rotation="vertical"
        )

    # Add drug names below the bars
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.01,
            drugs_top.iloc[i],
            horizontalalignment='center',
            color='#000000',
            fontsize=60,
            rotation="vertical",
        )

    ax.tick_params(bottom=True, left=False, axis='x', which='major', pad=-1)

    # Add labels and title
    ax.set_xlabel('', labelpad=15, color='#333333')
    
    # Make the chart fill out the figure better
    fig.tight_layout()

    # Save the plot to the specified output path
    fig.savefig(output_path, transparent=True)

def get_correlations_per_fold(input_folder,output_folder,druginput_name,drug2id,genomics_name,cell2id,gene2id,labels_name,drugs_names,model_name,cuda, folder):
    """
    Process a single sample folder to load data, make predictions, and return correlation results.

    Result:
    Saves in each output folder two DataFrames:
        - predictions_dataframe_sort_pearson: Sorted DataFrame of Pearson correlation results.
        - predictions_dataframe_sort_spearman: Sorted DataFrame of Spearman correlation results.
    """
    
    input_dir = input_folder + folder + "/"  # CHANGE
    results_dir = output_folder + folder + "/"  # CHANGE

    # Load input data
    drug2fingerprint_fold = input_dir + druginput_name
    drug2id_fold = input_dir + drug2id
    genotype_fold  = input_dir + genomics_name
    cell2id_fold  = input_dir + cell2id
    gene2id_fold  = input_dir + gene2id
    test_fold  = input_dir + labels_name
    
    # Load features
    cell_features = np.genfromtxt(genotype_fold, delimiter=',')
    drug_features = np.genfromtxt(drug2fingerprint_fold, delimiter=',')
    
    # Load mappings
    cell2id_mapping = load_mapping(cell2id_fold)
    drug2id_mapping = load_mapping(drug2id_fold)
    
    # Get compound names
    names = get_compound_names(input_dir + drugs_names)
    names.pop(0)
        
    # Set device and load model
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    # model = torch.load(results_dir + model_name, map_location=device)
    model = SparseGO.load_from_checkpoint(results_dir + model_name)
    

    batchsize = 10000
    
    # If the pre-trained model was wrapped with DataParallel, extract the underlying model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(device)
    
    predictions_pearson = {}
    predictions_spearman = {}
    
    for selected_drug_data in names:
        selected_drug = selected_drug_data[0]
        features, labels = load_selected_drug_data(test_fold, cell2id_mapping, drug2id_mapping, selected_drug)
        predict_data = (torch.Tensor(features), torch.FloatTensor(labels))
        pearson, spearman = predict_short(predict_data, model, batchsize, cell_features, drug_features, device)
        predictions_pearson[selected_drug_data[1]] = pearson.item()
        predictions_spearman[selected_drug_data[1]] = spearman.item()
    
    # Create and sort DataFrames for Pearson results
    predictions_dataframe_pearson = pd.DataFrame(predictions_pearson.items(), columns=['Name', folder + ' Pearson'])
    predictions_dataframe_sort_pearson = predictions_dataframe_pearson.sort_values(by=[folder + ' Pearson'], ascending=False)
    
    # Save Pearson results
    with open(results_dir + "drug_predictions_dataframe_sort_test_pearson.pkl", 'wb') as file:
        pickle.dump(predictions_dataframe_sort_pearson, file)   
    
    # Create and sort DataFrames for Spearman results
    predictions_dataframe_spearman = pd.DataFrame(predictions_spearman.items(), columns=['Name', folder + ' Spearman'])
    predictions_dataframe_sort_spearman = predictions_dataframe_spearman.sort_values(by=[folder + ' Spearman'], ascending=False)
    
    # Save Spearman results
    with open(results_dir + "drug_predictions_dataframe_sort_test_spearman.pkl", 'wb') as file:
        pickle.dump(predictions_dataframe_sort_spearman, file)

    return predictions_dataframe_sort_pearson, predictions_dataframe_sort_spearman
        
def all_folds_metrics_and_plots(run,input_folder,output_folder,druginput_name,drug2id,genomics_name,cell2id,gene2id,labels_name,drugs_names,model_name,samples_folders,predictions_name,cuda,log_artifact=True):
    # Calculate both Pearson and Spearman correlation coefficients for each drug across the different folds
    for folder in samples_folders:
        get_correlations_per_fold(input_folder,output_folder,druginput_name,drug2id,genomics_name,cell2id,gene2id,labels_name,drugs_names,model_name,cuda,folder)
    
    # Compute the average correlation for each drug
    
    # PEARSON ----------------------------------------------------
    # Import all the results 
    list_correlation = {sample: None for sample in samples_folders}
    for folder in samples_folders:
        with open(output_folder+folder+"/"+"drug_predictions_dataframe_sort_test_pearson.pkl", 'rb') as dictionary_file:
            list_correlation[folder] = pickle.load(dictionary_file)        
      
    # Join results   
    # Start with an empty dataframe
    all_df = pd.DataFrame()
    
    # Loop through the list of dataframes and merge them
    for key in list_correlation:
        df = list_correlation[key]
        if all_df.empty:
            all_df = df
        else:
            all_df = pd.merge(all_df, df, on=['Name'])
    all_df = all_df.set_index("Name")
    
    # Get the mean and std
    all_df_mean = all_df.mean(axis=1).reset_index(name='Mean')
    all_df_std = all_df.std(axis=1).reset_index(name='Std')
    
    mean_pearson = all_df_mean["Mean"].mean()
    print("Average pearson correlations of all drugs: ",mean_pearson)
    run.log({"Average pearson correlations of all drugs:": mean_pearson})
    
    all_df_values = pd.merge(all_df_mean,all_df_std,on=['Name'])
    all_df_values.columns = ["Name","Mean","Std"]
    all_df_values =all_df_values.sort_values(by=["Mean"], ascending=False)
    
    all_df_values.to_csv(output_folder+'pearson_means.txt', sep='\t', index=False)
    
    artifact = wandb.Artifact("means_pearson_drug",type="drug means")
    artifact.add_file(output_folder+'pearson_means.txt')
    if log_artifact: run.log_artifact(artifact)
    
    # Waterfall Plot Pearson
    create_waterfall_plot_correlation(all_df_values, output_folder+'WaterfallDrugsSparseGO_pearson.png', correlation_type='pearson')
    artifact = wandb.Artifact("WaterfallDrugsSparseGO_pearson",type="plots")
    artifact.add_file(output_folder+'WaterfallDrugsSparseGO_pearson.png')
    if log_artifact: run.log_artifact(artifact)
      
    # Top 10 drugs bar chart, pearson
    create_top_drugs_bar_chart(all_df_values, output_folder+'top10sparse_pearson.png', correlation_type='pearson')
    artifact = wandb.Artifact("top10sparse_pearson",type="plots")
    artifact.add_file(output_folder+'top10sparse_pearson.png')
    if log_artifact: run.log_artifact(artifact)
    
    # SPEARMAN ----------------------------------------------------
    # Import all the results 
    list_correlation = {sample: None for sample in samples_folders}
    for folder in samples_folders:
        with open(output_folder+folder+"/"+"drug_predictions_dataframe_sort_test_spearman.pkl", 'rb') as dictionary_file:
            list_correlation[folder] = pickle.load(dictionary_file)
            
    # Join results   
    # Start with an empty dataframe
    all_df = pd.DataFrame()
    
    # Loop through the list of dataframes and merge them
    for key in list_correlation:
        df = list_correlation[key]
        if all_df.empty:
            all_df = df
        else:
            all_df = pd.merge(all_df, df, on=['Name'])
    all_df = all_df.set_index("Name")
    
    all_df_mean = all_df.mean(axis=1).reset_index(name='Mean')
    all_df_std = all_df.std(axis=1).reset_index(name='Std')
    
    mean_spearman = all_df_mean["Mean"].mean()
    print("Average spearman correlations of all drugs: ",mean_spearman)
    run.log({"Average spearman correlations of all drugs:": mean_spearman})
    
    all_df_values = pd.merge(all_df_mean,all_df_std,on=['Name'])
    all_df_values.columns = ["Name","Mean","Std"]
    all_df_values =all_df_values.sort_values(by=["Mean"], ascending=False)
    
    all_df_values.to_csv(output_folder+'spearman_means.txt', sep='\t', index=False)
    
    artifact = wandb.Artifact("means_spearman_drug",type="drug means")
    artifact.add_file(output_folder+'spearman_means.txt')
    if log_artifact: run.log_artifact(artifact)
    
    # Waterfall plot Spearman
    create_waterfall_plot_correlation(all_df_values, output_folder+'WaterfallDrugsSparseGO_spearman.png', correlation_type='spearman')
    artifact = wandb.Artifact("WaterfallDrugsSparseGO_spearman",type="plots")
    artifact.add_file(output_folder+'WaterfallDrugsSparseGO_spearman.png')
    if log_artifact: run.log_artifact(artifact) 
        
    # Top 10 drugs bar chart, spearman
    create_top_drugs_bar_chart(all_df_values, output_folder+'top10sparse_spearman.png', correlation_type='spearman')
    artifact = wandb.Artifact("top10sparse_spearman",type="plots")
    artifact.add_file(output_folder+'top10sparse_spearman.png')
    if log_artifact: run.log_artifact(artifact) 
        
    
    # DENSITY AND FINAL METRICS ----------------------------------------------------
    # Create a dictionary for SMILES to drug name mapping
    drugs_info = get_compound_names(input_folder +folder+"/"+ drugs_names)
    drugs_info.pop(0)
    smiles_to_drug = {smiles: name for smiles, name in drugs_info}
    
    # Calculate the density plot of all the models  
    list_predictions = {sample: None for sample in samples_folders}
    list_models_pearsons = {sample: None for sample in samples_folders}
    list_models_spearmans = {sample: None for sample in samples_folders}
    for folder in samples_folders:
        file_labels = input_folder + folder + "/" + labels_name
        file_predictions = output_folder + folder + "/" + predictions_name
        
        real_AUDRC = []
        real_drugs = []
        sparse_AUDRC = []
        
        with open(file_labels, 'r') as fi:
            for line in fi:
                tokens = line.strip().split('\t')
                real_AUDRC.append(float(tokens[2]))
                real_drugs.append(tokens[1])
                
        real_AUDRCA = np.array(real_AUDRC)
        real_drugs = np.array(real_drugs)
                
        with open(file_predictions, 'r') as fi:
            for line in fi:
                tokens = line.strip().split('\t')
                sparse_AUDRC.append(float(tokens[0]))
        
        sparse_AUDRCA = np.array(sparse_AUDRC)
        
        list_predictions[folder] = pd.DataFrame(list(zip(real_AUDRC, sparse_AUDRC,real_drugs,["SparseGO"]*len(real_AUDRC))),columns =['Real AUDRC', 'Predicted AUDRC',"Drug",'Class',])
        list_models_pearsons[folder] = float(pearson_corr(torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy())
        list_models_spearmans[folder] = float(spearman_corr(torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Predicted AUDRC"].to_numpy()),torch.from_numpy(list_predictions[folder].loc[list_predictions[folder].loc[:,"Class"]=="SparseGO","Real AUDRC"].to_numpy())).numpy())


    all_predictions = pd.DataFrame()
    # Loop through the list of dataframes and concatenate them
    for key in list_predictions:
        df = list_predictions[key]
        if all_predictions.empty:
            all_predictions = df
        else:
            all_predictions = pd.concat([all_predictions, df])

    # Replace SMILES with drug names
    all_predictions['Drug'] = all_predictions['Drug'].map(smiles_to_drug)

    # Calculate average correlations
    pe_average = np.around(statistics.mean(list(list_models_pearsons.values())),4)
    sp_average =  np.around(statistics.mean(list(list_models_spearmans.values())),4)
    run.log({"Average pearson cor:": pe_average})
    run.log({"Average spearman cor:": sp_average})
    # PLOT RESULTS   
    create_density_plot(all_predictions,list_models_pearsons,list_models_spearmans, output_folder+'density_plot.png')    
    artifact = wandb.Artifact("density_plot",type="plots")
    artifact.add_file(output_folder+'density_plot.png')
    if log_artifact: run.log_artifact(artifact) 
    
    create_linear_models_plot(all_predictions,list_models_pearsons,list_models_spearmans, all_df_values, output_folder+'linear_models_plot.png')    
    artifact = wandb.Artifact("linear_models_plot",type="plots")
    artifact.add_file(output_folder+'linear_models_plot.png')
    if log_artifact: run.log_artifact(artifact) 

    print(all_df_mean.sort_values(by='Mean', ascending=False)) # print drug correlations means
    print("\nDrug correlations of all folds")
    pd.set_option('display.max_columns', None)  # None means display all columns
    print(all_df)

if __name__ == "__main__":
    mac = "/Users/katyna/Library/CloudStorage/OneDrive----/"
    windows = "C:/Users/ksada/OneDrive - Tecnun/"
    manitou = "/manitou/pmg/users/ks4237/" 
    computer = mac # CHANGE

    parser = argparse.ArgumentParser(description='SparseGO metrics')

    parser.add_argument('-project', help='W&B project name', type=str, default="Test")
    parser.add_argument('-entity', help='W&B entity', type=str, default="miramon_team")
    parser.add_argument('-tags', help='Tags of type of data or/and model we are testing', type=list_of_strings, default=['ChEMBL500_1_5'])
    parser.add_argument('-job_type', help='Job type', type=str, default="job-test")
    parser.add_argument('-sweep_name', help='Job type', type=str, default="final metrics drugs")

    parser.add_argument('-input_folder', help='Directory containing the input data folders.', type=str, default=computer+"SparseGO_lightning/data/PDCs_multiomics_LELO/")
    parser.add_argument('-output_folder', help='Directory containing the folders that have the resulting models', type=str, default=computer+"SparseGO_lightning/results/PDCs_multiomics_LELO/")
    parser.add_argument('-samples_folders', help='Folders to analyze', type=list_of_strings, default=["samples1","samples2","samples3"]) # ,"samples3", "samples4","samples5"

    parser.add_argument('-model_name', help='Model to use to compute individual drug predictions', type=str, default="best_model_d2.ckpt")
    parser.add_argument('-predictions_name', help='Which results to use for the density plot', type=str, default="d2_test_predictions.txt")

    parser.add_argument('-input_type', help='Type of omics data used', type=str, default="multiomics")
    parser.add_argument('-labels_name', help='Which results to use for the density plot', type=str, default="sparseGO_test.txt")
    parser.add_argument('-genomics_name', help='Which results to use for the density plot', type=str, default="cell2multiomics.txt")
    parser.add_argument('-gene2id', help='Which results to use for the density plot', type=str, default="gene2ind_multiomics.txt")
    parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default="drug2ind.txt")
    parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default="cell2ind.txt")
    parser.add_argument('-druginput_name', help='Which results to use for the density plot', type=str, default="drug2fingerprint.txt")
    parser.add_argument('-drugs_names', help='Drugs names and SMILEs', type=str, default="compound_names.txt")
    parser.add_argument('-drugs_fake_lelo', help='Drugs used in fake LELO (only if needed)', required=False, type=list_of_strings)
    parser.add_argument('-cuda', help='Cuda ID', type=str, default=0)

    opt, unknown = parser.parse_known_args()

    # Resulting figures are uploaded to w&b
    
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

    input_folder = opt.input_folder
    output_folder = opt.output_folder

    druginput_name = opt.druginput_name
    drug2id = opt.drug2id
    genomics_name = opt.genomics_name

    cell2id = opt.cell2id
    gene2id = opt.gene2id

    labels_name =  opt.labels_name
    predictions_name = opt.predictions_name

    drugs_names = opt.drugs_names
    model_name = opt.model_name

    samples_folders = opt.samples_folders

    cuda = opt.cuda

    all_folds_metrics_and_plots(run,input_folder,output_folder,druginput_name,drug2id,genomics_name,cell2id,gene2id,labels_name,drugs_names,model_name,samples_folders,predictions_name,cuda,log_artifact=True)
    run.finish()