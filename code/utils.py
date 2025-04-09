# Pytorch modules change 
import torch

# Other
import pandas as pd
import numpy as np
import GPUtil
import scipy.stats as ss # for spearman_corr 

def print_gpu_memory_info():
    # Get the list of GPUs
    gpus = GPUtil.getGPUs()

    # Iterate over the GPUs and print their details
    for gpu in gpus:
        memory_free_gb = gpu.memoryFree / 1024  # Convert free memory to GB
        memory_used_gb = gpu.memoryUsed / 1024  # Convert used memory to GB
        gpu_load_percentage = gpu.load * 100  # Convert GPU load to percentage

        print(f"GPU: {gpu.id} {gpu.name} | Memory Free: {memory_free_gb:.2f} GB | Memory Used: {memory_used_gb:.2f} GB | GPU Load: {gpu_load_percentage:.2f}%")
    print()  # Empty line for better readability

def print_tensor_info(tensor, var_name="not specified"):
    """
    Prints the name, shape, size in bytes, size in gigabytes, and location (CPU or GPU) of the given tensor in one line.

    Args:
        tensor (torch.Tensor): The tensor to analyze.
        var_name (str): The name of the variable.
    """
    # Ensure the input is a tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    # Get tensor shape
    tensor_shape = tensor.shape
    
    # Calculate tensor size in bytes and GB
    tensor_size_bytes = tensor.element_size() * tensor.numel()
    tensor_size_gb = tensor_size_bytes / (1024 ** 3)  # Convert bytes to GB

    # Determine if the tensor is on CPU or GPU
    device = tensor.device
    
    # dtype 
    dtype_tensor = tensor.dtype

    # Print the information in one line
    print(f"Variable: {var_name} | Tensor Shape: {tensor_shape} | Tensor Size (GB): {tensor_size_gb:.2f} | dtype: {dtype_tensor} | Device: {device}")

def sort_pairs(genes_terms_pairs, terms_pairs, dG, gene2id_mapping):
    """
    Function concatenates the pairs and orders them, the parent term goes first.

        Output
        ------
        level_list: list
            Each array of the list stores the elements on a level of the hierarchy

        level_number: dict
            Has the gene and GO terms with their corresponding level number

        sorted_pairs: numpy.ndarray
            Contains the term-gene or term-term pairs with the parent element on the first column
    """

    all_pairs = np.concatenate((genes_terms_pairs,terms_pairs))
    graph = dG.copy() #  Copy the graph to avoid modifying the original

    level_list = []   # level_list stores the elements on each level of the hierarchy
    level_list.append(list(gene2id_mapping.keys())) # add the genes

    while True:
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if len(leaves) == 0:
            break

        level_list.append(leaves) # add the terms on each level
        graph.remove_nodes_from(leaves)

    level_number = {} # Has the gene and GO terms with their corresponding level number
    for i, layer in enumerate(level_list):
        for _,item in enumerate(layer):
            level_number[item] = i

    sorted_pairs = all_pairs.copy() # order pairs based on their level
    for i, pair in enumerate(sorted_pairs):
        level1 = level_number[pair[0]]
        level2 = level_number[pair[1]]
        if level2 > level1:  # the parent term goes first
            sorted_pairs[i][1] = all_pairs[i][0]
            sorted_pairs[i][0] = all_pairs[i][1]

    return sorted_pairs, level_list, level_number

def pairs_in_layers(sorted_pairs, level_list, level_number):
    """
    This function divides all the pairs of GO terms and genes by layers and adds the virtual nodes

        Output
        ------
        layer_connections: numpy.ndarray
            Contains the pairs that will be part of each layer of the model.
            Not all terms are connected to a term on the level above it. "Virtual nodes" are added to establish the connections between non-subsequent levels.

    """
    total_layers = len(level_list)-1 # Number of layers that the model will contain
    # Will contain the GO terms connections and gene-term connections by layers
    layer_connections = [[] for i in range(total_layers)]

    for i, pair in enumerate(sorted_pairs):
       parent = level_number[pair[0]]
       child = level_number[pair[1]]

       # Add the pair to its corresponding layer
       layer_connections[child].append(pair)

       # If the pair is not directly connected virtual nodes have to be added
       dif = parent-child # number of levels in between
       if dif!=1:
          virtual_node_layer = parent-1
          for j in range(dif-1): # Add the necessary virtual nodes
              layer_connections[virtual_node_layer].append([pair[0],pair[0]])
              virtual_node_layer = virtual_node_layer-1

    # Delete pairs that are duplicated (added twice on the above step)
    for i,_ in enumerate(layer_connections):
        layer_connections[i] = np.array(layer_connections[i]) # change list to array
        layer_connections[i] = np.unique(layer_connections[i], axis=0)

    return layer_connections

def create_index(array):
    unique_array = pd.unique(array)

    index = {}
    for i, element in enumerate(unique_array):
        index[element] = i

    return index

def pearson_corr(x, y): # comprobado que esta bien (con R)
    xx = torch.round(x - torch.mean(x), decimals=4)
    yy = torch.round(y - torch.mean(y), decimals=4)
    # xx = x - torch.mean(x)
    # yy = y - torch.mean(y)

    return ((torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))))*100

def spearman_corr(x, y):  # comprobado que esta bien (con R)
    # Note: Changing a number slightly in either x or y can lead to significant changes in the Spearman correlation output,
    # especially when the input data points are closely clustered or have low variance.
    # The corr_per_drug_val sometimes is different when models are exactly the same because of the amount of decimals, a slight change can cause the correlation go from 0 to 0.15
    # If the predictions (x) are all the same, the corr is nan (Insufficient Variability: If one or both of the variables have constant values (e.g., all values are the same), the calculation cannot proceed because the standard deviation is zero.)
    # If nan, in some functions we change to 0 (per_drug_corr)
    
    x = np.round(x, decimals=4)  # Round x to 7 decimal places
    y = np.round(y, decimals=4)  # Round y to 7 decimal places
    
    # Calculate ranks using torch
    # x_rank = x.argsort().argsort().float()
    # y_rank = y.argsort().argsort().float()

    # # Calculate Pearson correlation on the ranks
    # spearman = pearson_corr(x_rank, y_rank)
    
    # Calculate Spearman correlation by ranking the data and then computing Pearson correlation on the ranks
    spearman = pearson_corr(torch.from_numpy(ss.rankdata(x) * 1.), torch.from_numpy(ss.rankdata(y) * 1.))

    return spearman

def per_drug_corr_spearman(drugs_ids, predictions, labels,yes=1):
    """
    Calculate the correlation per drug.

    Parameters:
    - drugs_ids (torch.Tensor): Tensor containing drug IDs for each prediction-label pair.
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.

    Returns:
    - float: Mean squared loss.
    """

    corr = 0     
    # Iterate over the unique drug IDs
    for chosen_id in torch.unique(drugs_ids):
        # Get the indices of the labels that have that drug id
        indices = (drugs_ids == chosen_id).nonzero(as_tuple=True)[0]

        # Get the predictions and labels for the corresponding drug
        grouped_tensor_predictions = predictions[indices]
        grouped_tensor_labels = labels[indices]
        
        # if yes==1:
        #     print("predictions: ")
        #     print(grouped_tensor_predictions)
            
        #     print("labels: ")
        #     print(grouped_tensor_labels)
        #     yes=0

        # Calculate the Spearman correlation for the current drug ID
        # Avoiding NaNs by using nan_to_num
        current_accuracy = torch.nan_to_num(spearman_corr(grouped_tensor_predictions.cpu().detach().numpy(), grouped_tensor_labels.cpu())).item()
        
        # Store the drug ID and its correlation
        # print(f"Drug ID: {chosen_id.item()}, Spearman Correlation: {current_accuracy:.4f}")
        
        # Add the current correlation to the total correlation
        corr += float(current_accuracy)

    # Return the corr mean 
    corr_mean = (corr / torch.unique(drugs_ids).shape[0])
    print(f"Mean per drug: {corr_mean}")
    return corr_mean

def low_corr(predictions, labels, threshold=0.2):
    """
    Calculate the correlation for low AUDRC values.

    Parameters:
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.
    - threshold 

    Returns:
    - float: correlation
    """

    # Get the indices of the labels that have that drug id
    indices = (labels < threshold).nonzero(as_tuple=True)[0]

    # Get the predictions and labels for the corresponding drug
    low_tensor_predictions = predictions[indices]
    low_tensor_labels = labels[indices]

    corr = spearman_corr(low_tensor_predictions.cpu().detach().numpy(), low_tensor_labels.cpu())

    return corr

def load_hyperparameters(filename):
    hyperparams = {}
    with open(filename, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            hyperparams[key] = value
    return hyperparams

def list_of_strings(arg):
    """
    Define a custom argument type for a list of strings (for tags) in argparse.

    Parameters:
    arg (str): Input argument containing comma-separated strings or a single string.

    Returns:
    list: A list of strings extracted from the input argument.

    Example:
    If arg = "tag1,tag2,tag3", the function will return ['tag1', 'tag2', 'tag3'].
    If arg = "tag1", the function will return ['tag1'].

    Usage:
    parser.add_argument('--tags', type=list_of_strings, help='List of tags separated by commas')

    """
    if ',' in arg:
        return arg.split(',')
    else:
        return [arg]
    
def get_compound_names(file_name):
    """
    Retrieve compound names from a specified file.

    Parameters:
    file_name (str): The name of the file containing compound names.

    Returns:
    list: A list of compound names extracted from the file.
    """

    compounds = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            compounds.append([tokens[1], tokens[2]])

    return compounds