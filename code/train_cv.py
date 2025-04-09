# Standard library imports
import os
import sys
import time
import random
import argparse

# Pytorch modules
import torch
import torch.nn as nn
import torch.utils.data as du

# Pytorch-Lightning
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader

# Weights & Biases
import wandb
from lightning.pytorch.loggers import WandbLogger

# Ontology 
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

# Other
import numpy as np
from scipy import sparse # for genes_layer

# My functions
import utils
from utils import *
from sparse_linear_new import SparseLinearNew

import warnings
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)


class SparseGO(LightningModule):
    def __init__(self, output_folder, fold, input_type, num_neurons_per_GO, num_neurons_per_final_GO, num_neurons_drug, num_neurons_final, layer_connections, p_drop_final, p_drop_genes, p_drop_terms, p_drop_drugs, learning_rate, decay_rate, optimizer_type, momentum, loss_type, cell_features, drug_features, gene2id_mapping_ont=None,genes_genes_pairs=None, gene2id_mapping_multiomics=None, ckpt_path=None):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.fold = fold
        # Set the output folder path
        self.output_folder = output_folder + self.fold + "/"
    
        # Model parameters
        self.num_neurons_per_GO = num_neurons_per_GO  # Number of neurons per GO term
        self.num_neurons_per_final_GO = num_neurons_per_final_GO  # Number of neurons in final GO layer
        self.num_neurons_drug = num_neurons_drug  # Number of neurons for drug features
        self.num_neurons_final = num_neurons_final  # Number of neurons in the final layer
        self.p_drop_genes = p_drop_genes  # Dropout rate for genes
        self.p_drop_terms = p_drop_terms  # Dropout rate for terms
        self.p_drop_drugs = p_drop_drugs  # Dropout rate for drugs
        self.p_drop_final = p_drop_final  # Dropout rate for final layer
        
        # Ontology structure
        self.input_type = input_type  # Type of input data (e.g., multiomics)
        self.layer_connections = layer_connections  # Connections between layers in the model
        self.gene2id_mapping_ont = gene2id_mapping_ont  # Mapping of genes to IDs for ontology
        self.gene_dim_ont = len(gene2id_mapping_ont)  # Dimension of gene mapping for ontology
                
        # Training parameters
        self.initial_learning_rate = learning_rate  # Learning rate for the optimizer
        self.decay_rate = decay_rate  # Decay rate for learning
        self.optimizer_type = optimizer_type  # Type of optimizer to use
        self.momentum = momentum  # Momentum for the optimizer
        self.loss_type = loss_type  # Type of loss function to use
        
        # Input data characteristics
        self.cell_features = cell_features  # Features for cell data
        self.drug_features = drug_features  # Features for drug data
        # self.cell_number = len(cell_features)  # Dimension of cell features
        self.drug_dim = len(self.drug_features[0,:])  

        if os.environ.get("LOCAL_RANK")=="0":
            print(f"""
            Configuration:
            Neuron Counts:
                - GO term (num_neurons_per_GO): {num_neurons_per_GO}
                - Drug neurons (num_neurons_drug): {num_neurons_drug}
                - Final GO term (num_neurons_per_final_GO): {num_neurons_per_final_GO}
                - Final neurons (num_neurons_final): {num_neurons_final}
                - Number of term-term hierarchy levels (len(layer_connections)): {len(layer_connections)}
                
            Dropout rates:
                - Genes (p_drop_genes): {p_drop_genes}
                - Terms (p_drop_terms): {p_drop_terms}
                - Drugs (p_drop_drugs): {p_drop_drugs}
                - Final (p_drop_final): {p_drop_final}
            
            Training parameters:
                - Learning rate (learning_rate): {learning_rate}
                - Decay rate (decay_rate): {decay_rate}
                - Optimizer (optimizer_type): {optimizer_type}
                - Momentum (momentum): {momentum}
                - Loss type (loss_type): {loss_type}
            """)
            
        # Configure loss
        self.configure_loss()
        
        # Intialize metrics
        self.initialize_metrics()
        
        if self.input_type == "multiomics":
            self.gene_dim_multiomics = len(gene2id_mapping_multiomics)
            self.gene_dim_input = self.gene_dim_multiomics
            self.gene2id_mapping_multiomics = gene2id_mapping_multiomics
            self.genes_genes_pairs = genes_genes_pairs
        
            # (0) Layer of genes with genes
            self.multiomics_layer(genes_genes_pairs, gene2id_mapping_ont, gene2id_mapping_multiomics)
        else:
            self.gene_dim_input = self.gene_dim_ont
        
        # Define an example input array based on the input type and dimensions
        num_features = self.gene_dim_input + self.drug_dim
        self.example_input_array = torch.randn(5, num_features) 
        
        # (1) Layer of genes with terms
        input_id = self.genes_layer(layer_connections[0],p_drop_genes, gene2id_mapping_ont)
        
        # (2...) Layers of terms with terms
        for i in range(1,len(layer_connections)):
            if i == len(layer_connections)-1:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_final_GO,p_drop_terms)
            else:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_GO,p_drop_terms)
        
        # Layers to process drugs
        self.construct_NN_drug(p_drop_drugs)
        
        # Concatenate branches
        if len(num_neurons_drug)==0: # to concatenate directly the drugs layer
            final_input_size = num_neurons_per_final_GO + self.drug_dim
        else:
            final_input_size = num_neurons_per_final_GO + num_neurons_drug[-1] 
        self.final_batchnorm_layer = nn.BatchNorm1d(final_input_size) 
        self.drop_final = nn.Dropout(p_drop_final)
        self.final_linear_layer = nn.Linear(final_input_size, num_neurons_final)
        self.final_tanh = nn.Tanh()
        self.final_aux_batchnorm_layer = nn.BatchNorm1d(num_neurons_final)
        self.drop_aux_final = nn.Dropout(p_drop_final)
        self.final_aux_linear_layer = nn.Linear(num_neurons_final,1)
        self.final_aux_tanh = nn.Tanh()
        self.final_linear_layer_output = nn.Linear(1, 1)
        
        if os.environ.get("LOCAL_RANK")=="0":
            # Print details of linear layers in the model
            for name, layer in self.named_modules():
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"Layer Name: {name}\tInput Features: {layer.in_features}\tOutput Features: {layer.out_features}")
        
        self.save_hyperparameters(logger=False)
        
    def initialize_metrics(self):

        # Initialize train metrics
        self.train_loss = MeanMetric()
        self.train_corr_pearson = MeanMetric()
        self.train_corr_spearman = MeanMetric()
        self.train_corr_per_drug = MeanMetric()
        self.train_corr_low = MeanMetric()

        # Initialize validation metrics
        self.validation_loss = MeanMetric()
        self.validation_corr_pearson = MeanMetric()
        self.validation_corr_sperman = MeanMetric()
        self.validation_corr_per_drug = MeanMetric()
        self.validation_corr_low = MeanMetric()
        
        # Initialize test metrics
        self.test_loss = MeanMetric()
        self.test_corr_pearson = MeanMetric()
        self.test_corr_spearman = MeanMetric()
        self.test_corr_per_drug = MeanMetric()
        self.test_corr_low = MeanMetric()
        
        self.all_test_outputs = []  # List to accumulate test outputs
    
    def multiomics_layer(self, genes_genes_pairs, gene2id_mapping_ont, gene2id_mapping_multiomics):
        # Change genes of the ontology and genes on the input to its indexes
        
        ids_ontology = [gene2id_mapping_ont[gene] for gene in genes_genes_pairs[:,1]] # rows
        ids_multiomics = [gene2id_mapping_multiomics[gene] for gene in genes_genes_pairs[:,0]] # columns
        
        connections_layer = torch.stack((
            torch.tensor(ids_ontology, device=self.device), 
            torch.tensor(ids_multiomics, device=self.device)
        ))

        input_terms = len(gene2id_mapping_multiomics)
        output_terms = len(gene2id_mapping_ont)
        
        self.genes_genes_sparse_linear = SparseLinearNew(input_terms, output_terms, connectivity=connections_layer)
        self.genes_genes_tanh = nn.Tanh()
        self.genes_genes_batchnorm = nn.BatchNorm1d(input_terms)
           
    def genes_layer(self, genes_terms_pairs, p_drop_genes, gene2id):
        # Define the layer of terms with genes, each pair is repeated n times (for the n neurons)
        
        term2id = create_index(genes_terms_pairs[:,0])

        self.term_dim = len(term2id)

        # Change term and genes to its indexes
        rows = [term2id[term] for term in genes_terms_pairs[:,0]]
        columns = [gene2id[gene] for gene in genes_terms_pairs[:,1]]

        data = np.ones(len(rows))

        # Create sparse matrix of terms connected to genes 
        genes_terms = sparse.coo_matrix((data, (rows, columns)), shape=(self.term_dim, self.gene_dim_ont))

        # Add n neurons to each term
        genes_terms_more_neurons = sparse.lil_matrix((self.term_dim*self.num_neurons_per_GO, self.gene_dim_ont))
        genes_terms = genes_terms.tolil()
        # Repeat the rows of the sparse matrix to match the 6 neurons
        row=0
        for i in range(genes_terms_more_neurons.shape[0]):
            if (i != 0) and (i%self.num_neurons_per_GO) == 0 :
                row=row+1
            genes_terms_more_neurons[i,:]=genes_terms[row,:]

        # Get the indexes of the matrix to define the connections of the sparse layer
        rows_more_neurons = torch.from_numpy(sparse.find(genes_terms_more_neurons)[0]).view(1,-1).long().to(self.device)
        columns_more_neurons = torch.from_numpy(sparse.find(genes_terms_more_neurons)[1]).view(1,-1).long().to(self.device)
        connections_layer1 = torch.cat((rows_more_neurons, columns_more_neurons), dim=0) # connections of the first layer each gene-term pair is repeated n times

        input_terms = len(gene2id)
        output_terms = self.num_neurons_per_GO*len(term2id) # n * GOterms
        self.genes_terms_sparse_linear_1 = SparseLinearNew(input_terms, output_terms, connectivity=connections_layer1)
        self.genes_terms_batchnorm = nn.BatchNorm1d(input_terms)
        self.genes_terms_tanh = nn.Tanh()
        self.drop_0 = nn.Dropout(p_drop_genes)

        return term2id
    
    def terms_layer(self, input_id, layer_pairs, number,neurons_per_GO,p_drop_terms):

        output_id = create_index(layer_pairs[:,0])

        # change term and genes to its indexes
        rows = [output_id[term] for term in layer_pairs[:,0]]
        columns = [input_id[term] for term in layer_pairs[:,1]]

        data = np.ones(len(rows))

        # Create sparse matrix of terms connected to terms
        connections_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(len(output_id), len(input_id)))

        # Add the n neurons with kronecker
        ones = sparse.csr_matrix(np.ones([neurons_per_GO, self.num_neurons_per_GO], dtype = int))
        connections_matrix_more_neurons = sparse.csr_matrix(sparse.kron(connections_matrix, ones))

        # Find the rows and columns of the connections
        rows_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[0]).view(1,-1).long().to(self.device)
        columns_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[1]).view(1,-1).long().to(self.device)
        connections = torch.cat((rows_more_neurons, columns_more_neurons), dim=0)

        input_terms = self.num_neurons_per_GO*len(input_id)
        output_terms = neurons_per_GO*len(output_id)
        self.add_module('GO_terms_sparse_linear_'+number, SparseLinearNew(input_terms, output_terms, connectivity=connections))
        self.add_module('drop_'+number, nn.Dropout(p_drop_terms))
        self.add_module('GO_terms_tanh_'+number, nn.Tanh())
        self.add_module('GO_terms_batchnorm_'+number, nn.BatchNorm1d(input_terms))
        return output_id
    
    def construct_NN_drug(self,p_drop_drugs):
        input_size = self.drug_dim  # Initialize input size based on drug dimensions

        for i in range(len(self.num_neurons_drug)):
            self.add_module('drug_linear_layer_' + str(i+1), nn.Linear(input_size, self.num_neurons_drug[i]))
            self.add_module('drug_drop_' + str(i+1),nn.Dropout(p_drop_drugs)) # Add a dropout layer with the given probability to prevent overfitting
            self.add_module('drug_tanh_' + str(i+1), nn.Tanh())
            self.add_module('drug_batchnorm_layer_' + str(i+1), nn.BatchNorm1d(input_size)) # Add batch normalization to stabilize learning
            
            # Update the input size for the next layer based on the current layer's number of neurons
            input_size = self.num_neurons_drug[i]
            
    def forward(self, x):
        if self.input_type in("mutations","expression"): 
            gene_input = x.narrow(1, 0, self.gene_dim_ont) # features of genes (Returns a new tensor that is a narrowed version)
            drug_input = x.narrow(1, self.gene_dim_ont, self.drug_dim) # features of drugs
        elif self.input_type  == "multiomics":
            multiomic_input = x.narrow(1, 0, self.gene_dim_multiomics) # features of genes (Returns a new tensor that is a narrowed version)
            drug_input = x.narrow(1, self.gene_dim_multiomics, self.drug_dim) # features of drugs
            # (0) Layer of genes with genes + tanh
            # batch --> dropout --> dense --> activation
            gene_input = self.genes_genes_batchnorm(multiomic_input)
            # gene_input = multiomic_input # Remove batchnorm 
            gene_input  = self.genes_genes_tanh(self.genes_genes_sparse_linear(gene_input))

        # define forward function for GO terms and genes #############################################
        # (1) Layer 1 + tanh
        # batch --> dropout --> dense --> activation
        gene_output = self.genes_terms_batchnorm(gene_input)
        gene_output = self.drop_0(gene_output)
        terms_output  = self.genes_terms_tanh(self.genes_terms_sparse_linear_1(gene_output))

        # (2...) Layer 2,3,4... + tanh
        for i in range(1,len(self.layer_connections)):
            # batch --> dropout --> dense --> activation
            terms_output = self._modules['GO_terms_batchnorm_'+str(i)](terms_output)
            terms_output = self._modules['drop_'+str(i)](terms_output)
            terms_output =  self._modules['GO_terms_tanh_'+str(i)](self._modules['GO_terms_sparse_linear_'+str(i)](terms_output))

        # define forward function for drugs #################################################
        drug_out = drug_input

        for i in range(1, len(self.num_neurons_drug)+1, 1):
            # batch --> dropout --> dense --> activation
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)](drug_out)
            drug_out = self._modules['drug_drop_'+str(i)](drug_out)
            drug_out = self._modules['drug_tanh_'+str(i)](self._modules['drug_linear_layer_' + str(i)](drug_out))

        # connect two neural networks #################################################
        final_input = torch.cat((terms_output, drug_out), 1)

        # batch --> dropout --> dense --> activation
        output = self.final_batchnorm_layer(final_input)
        output = self.drop_final(output)
        output = self.final_tanh(self.final_linear_layer(output))
        output = self.final_aux_batchnorm_layer(output)
        output = self.drop_aux_final(output)
        output = self.final_aux_tanh(self.final_aux_linear_layer(output))
        final_output = self.final_linear_layer_output(output)
        return final_output
            
    def on_train_epoch_start(self):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()
                
        # Calculate the new learning rate using the provided formula
        learning_rate = self.initial_learning_rate * (1 / (1 + self.decay_rate * self.current_epoch))  # or epoch * epoch
        # Apply the new learning rate to the optimizer
        self.trainer.optimizers[0].param_groups[0]['lr'] = learning_rate  # Set the new learning rate

        # Optional: Print the new learning rate for debugging
        print(f'Epoch {self.current_epoch}: Learning rate adjusted to {learning_rate:.6f}')
    def training_step(self, train_batch, batch_idx):
            
        start_time = time.time()
        
        input_data, labels = train_batch 
        features = self.build_input_vector(input_data)
        
        outputs = self(features) # Forward pass
        loss = self.criterion(outputs,  labels)
        
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Batch {batch_idx} loaded and forward passed in {load_time:.4f} seconds with {self.trainer.datamodule.num_workers} workers.")
        
        corr_pearson = pearson_corr(outputs, labels) # compute pearson correlation
        corr_spearman = spearman_corr(outputs.cpu().detach().numpy(), labels.cpu()) # compute spearman correlation
        corr_per_drug = per_drug_corr_spearman(input_data[:,1], outputs, labels) # REVISAAR
        corr_low = low_corr(outputs, labels)

        # Update metrics
        self.train_loss(loss)
        self.train_corr_pearson(corr_pearson)
        self.train_corr_spearman(corr_spearman)
        self.train_corr_per_drug(corr_per_drug)
        self.train_corr_low(corr_low)
        
        # Print device, batch index, loss, Spearman correlation, and correlation per drug in a single line
        print(f"{self.device}, Train Batch: {batch_idx}, Loss: {loss.detach():.4f}, "
              f"Spearman: {corr_spearman.detach():.4f}, "
              f"Correlation per Drug Mean: {corr_per_drug:.4f}", end=" --- ")
        print_tensor_info(features, "features")
        
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        
        start_time = time.time()
        
        input_data, labels = val_batch 
        features = self.build_input_vector(input_data)
        
        outputs = self(features) # Forward pass
        loss = self.criterion(outputs,  labels)
        
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Batch {batch_idx} loaded and forward passed in {load_time:.4f} seconds with {self.trainer.datamodule.num_workers} workers.")
        
        corr_pearson = pearson_corr(outputs, labels) # compute pearson correlation
        corr_spearman = spearman_corr(outputs.cpu().detach().numpy(), labels.cpu()) # compute spearman correlation
        corr_per_drug = per_drug_corr_spearman(input_data[:,1], outputs, labels) # REVISAAR
        corr_low = low_corr(outputs, labels)

        # Update metrics
        self.validation_loss(loss)
        self.validation_corr_pearson(corr_pearson)
        self.validation_corr_sperman(corr_spearman)
        self.validation_corr_per_drug(corr_per_drug)
        self.validation_corr_low(corr_low)
        
        # Print device, batch index, loss, Spearman correlation, and correlation per drug in a single line
        print(f"{self.device}, Val. Batch: {batch_idx}, Loss: {loss.detach():.4f}, "
              f"Spearman: {corr_spearman.detach():.4f}, "
              f"Correlation per Drug Mean: {corr_per_drug:.4f}", end=" --- ")
        print_tensor_info(features, "features")
        
    def on_validation_epoch_end(self):
        # Skip logging and printing during validation sanity check
        if self.trainer.sanity_checking:
            return
        
        # Compute average metrics from all batches
        avg_validation_loss = self.validation_loss.compute()
        avg_validation_corr_pearson = self.validation_corr_pearson.compute()
        avg_validation_corr_sperman = self.validation_corr_sperman.compute()
        avg_validation_corr_per_drug = self.validation_corr_per_drug.compute()
        
        # Log training metrics
        self.log_dict({
            "training_loss": self.train_loss.compute(),  # Average training loss
            "training_corr_pearson": self.train_corr_pearson.compute(),  # Average Pearson correlation
            "training_corr_spearman": self.train_corr_spearman.compute(),  # Average Spearman correlation
            "training_corr_per_drug": self.train_corr_per_drug.compute(),  # Average correlation per drug
            "training_corr_low": self.train_corr_low.compute(),  # Average low correlation metric
            "validation_loss": avg_validation_loss,  # Average validation loss
            "validation_corr_pearson": avg_validation_corr_pearson,  # Average validation Pearson correlation
            "validation_corr_spearman": avg_validation_corr_sperman,  # Average validation Spearman correlation
            "validation_corr_per_drug": avg_validation_corr_per_drug,  # Average validation correlation per drug
            "validation_corr_low": self.validation_corr_low.compute(),  # Average validation low correlation metric
        }, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        # Get the current device
        device = self.device  # Get the current device
        # Get the current epoch number
        epoch = self.current_epoch  

        # Print metrics and device information in a single line
        print(f"Epoch end {epoch} | "
              f"Device: {device} | "
              f"Training Loss: {self.train_loss.compute():.4f} | "
              f"Training Pearson Corr.: {self.train_corr_pearson.compute():.4f} | "
              f"Training Spearman Corr.: {self.train_corr_spearman.compute():.4f} | "
              f"Training Training Mean Corr. Per Drug: {self.train_corr_per_drug.compute():.4f} | "
              f"Validation Loss: {avg_validation_loss:.4f} | "
              f"Validation Pearson Corr.: {avg_validation_corr_pearson:.4f} | "
              f"Validation Spearman Corr.: {avg_validation_corr_sperman:.4f} | "
              f"Validation Training Mean Corr. Per Drug: {avg_validation_corr_per_drug:.4f} | ")

        # Optional: Print GPU memory info
        if self.trainer.local_rank==0:
            print_gpu_memory_info()
        
        # Reset metrics for the next epoch
        self.train_loss.reset()
        self.train_corr_pearson.reset()
        self.train_corr_spearman.reset()
        self.train_corr_per_drug.reset()
        self.train_corr_low.reset()
        self.validation_loss.reset()
        self.validation_corr_pearson.reset()
        self.validation_corr_sperman.reset()
        self.validation_corr_per_drug.reset()
        self.validation_corr_low.reset()
        
        # Calculate and print the time taken for the epoch
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch} took {epoch_time:.2f} seconds")

    def on_test_epoch_start(self):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()
        
        # Extract the metric from the checkpoint path
        if self.ckpt_path is not None:
            # Assuming the checkpoint path is structured like "output_folder/fold/best_model_d.ckpt"
            # Extract the metric from the filename
            _, filename = os.path.split(self.ckpt_path)  # Get the filename from the path
            self.metric = filename.split('_')[-1].split('.')[0]  # Get the last part before the extension
        else:
            self.metric = "other"  # Fallback if ckpt_path is not set   
                         
    def test_step(self, test_batch, batch_idx):
        
        start_time = time.time()
        
        input_data, labels = test_batch 
        features = self.build_input_vector(input_data)
        
        outputs = self(features) # Forward pass
        loss = self.criterion(outputs,  labels)
        
        end_time = time.time()
        load_time = end_time - start_time
        print(f"Batch {batch_idx} loaded and forward passed in {load_time:.4f} seconds with {self.trainer.datamodule.num_workers} workers.")
        
        corr_pearson = pearson_corr(outputs, labels) # compute pearson correlation
        corr_spearman = spearman_corr(outputs.cpu().detach().numpy(), labels.cpu()) # compute spearman correlation
        corr_per_drug = per_drug_corr_spearman(input_data[:,1], outputs, labels) # REVISAAR
        corr_low = low_corr(outputs, labels)
        
        # Accumulate the outputs
        self.all_test_outputs.append(outputs.cpu().detach())

        # Update metrics
        self.test_loss(loss)
        self.test_corr_pearson(corr_pearson)
        self.test_corr_spearman(corr_spearman)
        self.test_corr_per_drug(corr_per_drug)
        self.test_corr_low(corr_low)
        
        # Print device, batch index, loss, Spearman correlation, and correlation per drug in a single line
        print(f"{self.device}, Test Batch: {batch_idx}, Loss: {loss.detach():.4f}, "
              f"Spearman: {corr_spearman.detach():.4f}, "
              f"Correlation per Drug Mean: {corr_per_drug:.4f}", end=" --- ")
        print_tensor_info(features, "features")
        
        return {"predictions": outputs, "labels": labels,"input_data":input_data}
        
    def on_test_epoch_end(self):
        device = self.device  # Get the current device
        # Print metrics and device information in a single line
        print(f"Device: {device} | "
              f"Test Loss: {self.test_loss.compute():.4f} | "
              f"Test Training Pearson Corr.: {self.test_corr_pearson.compute():.4f} | "
              f"Test Training Spearman Corr.: {self.test_corr_spearman.compute():.4f} | "
              f"Test Training Mean Corr. Per Drug: {self.test_corr_per_drug.compute():.4f} | ")
        
        self.log_dict({
            "test_loss_" + self.metric: self.test_loss.compute(),  # Average training loss
            "test_corr_pearson_" + self.metric: self.test_corr_pearson.compute(),  # Average Pearson correlation
            "test_corr_spearman_" + self.metric: self.test_corr_spearman.compute(),  # Average Spearman correlation
            "test_corr_per_drug_" + self.metric: self.test_corr_per_drug.compute(),  # Average correlation per drug
            "test_corr_low_" + self.metric :self.test_corr_low.compute(),  # Average low correlation metric
        }, on_epoch=True, prog_bar=False, sync_dist=True, on_step=False)
        
        # Convert list of tensors to a single tensor
        all_test_outputs = torch.cat(self.all_test_outputs, dim=0)

        # Save the accumulated outputs to a file
        np.savetxt(self.output_folder + self.metric + '_test_predictions.txt', 
                   all_test_outputs.numpy(), 
                   fmt='%.5e')
        
        # Reset metrics for the next epoch
        self.test_loss.reset()
        self.test_corr_pearson.reset()
        self.test_corr_spearman.reset()
        self.test_corr_per_drug.reset()
        self.test_corr_low.reset()
        self.all_test_outputs = []
    
    def configure_optimizers(self):
        # falta añadir lo del learning rate, el decay_rate
        # Define the optimizer
        if self.optimizer_type=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_learning_rate, momentum=self.momentum)
        elif self.optimizer_type=='adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate, betas=(0.9, 0.99), eps=1e-05)
        return optimizer
    
    def configure_loss(self):
        # Define the loss criterion based on the configuration
        if self.loss_type=='MSELoss':
            self.criterion = nn.MSELoss()
            self.test_model = '/best_model_d.pt' # Test model is spearman
        elif self.loss_type=='L1Loss':
            self.criterion = nn.L1Loss()
            self.test_model = '/best_model_p.pt' # Test model is pearson
       
    def build_input_vector(self, input_data):
        """
        Build an input vector for training by combining cell and drug features based on the input data.

        Parameters:
        - input_data (torch.Tensor): A tensor containing indices that reference specific cell and drug features. 
                                    Each row corresponds to a specific sample, where the first column 
                                    contains the index for the cell features and the second column 
                                    contains the index for the drug features.

        Returns:
        - torch.Tensor: A tensor of shape (num_samples, num_features), where num_features is the 
                        sum of the number of cell features and drug features. Each row corresponds 
                        to the combined features of a specific cell and drug pair.
        """
    
        # Initialize a numpy array to hold the combined features for each input sample
        features = np.zeros((input_data.size()[0], (self.gene_dim_input + self.drug_dim)))

        # Iterate over each sample in the inputdata
        for i in range(input_data.size()[0]):
            # Concatenate the cell features and drug features based on the indices specified in inputdata
            features[i] = np.concatenate(
                (self.cell_features[int(input_data[i, 0])], self.drug_features[int(input_data[i, 1])]),
                axis=None  # Concatenate along the first axis (flatten)
            )

        # Convert the numpy array to a PyTorch tensor of type float
        features = torch.from_numpy(features).float().to(self.device)
        
        # Return the constructed feature tensor
        return features

class SparseGOnometrics(LightningModule):
    def __init__(self, output_folder, fold, input_type, num_neurons_per_GO, num_neurons_per_final_GO, num_neurons_drug, num_neurons_final, layer_connections, p_drop_final, p_drop_genes, p_drop_terms, p_drop_drugs, learning_rate, decay_rate, optimizer_type, momentum, loss_type, cell_features, drug_features, gene2id_mapping_ont=None,genes_genes_pairs=None, gene2id_mapping_multiomics=None, ckpt_path=None):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.fold = fold
        # Set the output folder path
        self.output_folder = output_folder + self.fold + "/"
    
        # Model parameters
        self.num_neurons_per_GO = num_neurons_per_GO  # Number of neurons per GO term
        self.num_neurons_per_final_GO = num_neurons_per_final_GO  # Number of neurons in final GO layer
        self.num_neurons_drug = num_neurons_drug  # Number of neurons for drug features
        self.num_neurons_final = num_neurons_final  # Number of neurons in the final layer
        self.p_drop_genes = p_drop_genes  # Dropout rate for genes
        self.p_drop_terms = p_drop_terms  # Dropout rate for terms
        self.p_drop_drugs = p_drop_drugs  # Dropout rate for drugs
        self.p_drop_final = p_drop_final  # Dropout rate for final layer
        
        # Ontology structure
        self.input_type = input_type  # Type of input data (e.g., multiomics)
        self.layer_connections = layer_connections  # Connections between layers in the model
        self.gene2id_mapping_ont = gene2id_mapping_ont  # Mapping of genes to IDs for ontology
        self.gene_dim_ont = len(gene2id_mapping_ont)  # Dimension of gene mapping for ontology
                
        # Training parameters
        self.initial_learning_rate = learning_rate  # Learning rate for the optimizer
        self.decay_rate = decay_rate  # Decay rate for learning
        self.optimizer_type = optimizer_type  # Type of optimizer to use
        self.momentum = momentum  # Momentum for the optimizer
        self.loss_type = loss_type  # Type of loss function to use
        
        # Input data characteristics
        self.cell_features = cell_features  # Features for cell data
        self.drug_features = drug_features  # Features for drug data
        # self.cell_number = len(cell_features)  # Dimension of cell features
        self.drug_dim = len(self.drug_features[0,:])  

        if os.environ.get("LOCAL_RANK")=="0":
            print(f"""
            Configuration:
            Neuron Counts:
                - GO term (num_neurons_per_GO): {num_neurons_per_GO}
                - Drug neurons (num_neurons_drug): {num_neurons_drug}
                - Final GO term (num_neurons_per_final_GO): {num_neurons_per_final_GO}
                - Final neurons (num_neurons_final): {num_neurons_final}
                - Number of term-term hierarchy levels (len(layer_connections)): {len(layer_connections)}
                
            Dropout rates:
                - Genes (p_drop_genes): {p_drop_genes}
                - Terms (p_drop_terms): {p_drop_terms}
                - Drugs (p_drop_drugs): {p_drop_drugs}
                - Final (p_drop_final): {p_drop_final}
            
            Training parameters:
                - Learning rate (learning_rate): {learning_rate}
                - Decay rate (decay_rate): {decay_rate}
                - Optimizer (optimizer_type): {optimizer_type}
                - Momentum (momentum): {momentum}
                - Loss type (loss_type): {loss_type}
            """)
        
        if self.input_type == "multiomics":
            self.gene_dim_multiomics = len(gene2id_mapping_multiomics)
            self.gene_dim_input = self.gene_dim_multiomics
            self.gene2id_mapping_multiomics = gene2id_mapping_multiomics
            self.genes_genes_pairs = genes_genes_pairs
        
            # (0) Layer of genes with genes
            self.multiomics_layer(genes_genes_pairs, gene2id_mapping_ont, gene2id_mapping_multiomics)
        else:
            self.gene_dim_input = self.gene_dim_ont
        
        # Define an example input array based on the input type and dimensions
        num_features = self.gene_dim_input + self.drug_dim
        self.example_input_array = torch.randn(5, num_features) 
        
        # (1) Layer of genes with terms
        input_id = self.genes_layer(layer_connections[0],p_drop_genes, gene2id_mapping_ont)
        
        # (2...) Layers of terms with terms
        for i in range(1,len(layer_connections)):
            if i == len(layer_connections)-1:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_final_GO,p_drop_terms)
            else:
                input_id = self.terms_layer(input_id, layer_connections[i], str(i),num_neurons_per_GO,p_drop_terms)
        
        # Layers to process drugs
        self.construct_NN_drug(p_drop_drugs)
        
        # Concatenate branches
        if len(num_neurons_drug)==0: # to concatenate directly the drugs layer
            final_input_size = num_neurons_per_final_GO + self.drug_dim
        else:
            final_input_size = num_neurons_per_final_GO + num_neurons_drug[-1] 
        self.final_batchnorm_layer = nn.BatchNorm1d(final_input_size) 
        self.drop_final = nn.Dropout(p_drop_final)
        self.final_linear_layer = nn.Linear(final_input_size, num_neurons_final)
        self.final_tanh = nn.Tanh()
        self.final_aux_batchnorm_layer = nn.BatchNorm1d(num_neurons_final)
        self.drop_aux_final = nn.Dropout(p_drop_final)
        self.final_aux_linear_layer = nn.Linear(num_neurons_final,1)
        self.final_aux_tanh = nn.Tanh()
        self.final_linear_layer_output = nn.Linear(1, 1)
        
        if os.environ.get("LOCAL_RANK")=="0":
            # Print details of linear layers in the model
            for name, layer in self.named_modules():
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"Layer Name: {name}\tInput Features: {layer.in_features}\tOutput Features: {layer.out_features}")
        
        self.save_hyperparameters(logger=False)
    
    def multiomics_layer(self, genes_genes_pairs, gene2id_mapping_ont, gene2id_mapping_multiomics):
        # Change genes of the ontology and genes on the input to its indexes
        
        ids_ontology = [gene2id_mapping_ont[gene] for gene in genes_genes_pairs[:,1]] # rows
        ids_multiomics = [gene2id_mapping_multiomics[gene] for gene in genes_genes_pairs[:,0]] # columns
        
        connections_layer = torch.stack((
            torch.tensor(ids_ontology, device=self.device), 
            torch.tensor(ids_multiomics, device=self.device)
        ))

        input_terms = len(gene2id_mapping_multiomics)
        output_terms = len(gene2id_mapping_ont)
        
        self.genes_genes_sparse_linear = SparseLinearNew(input_terms, output_terms, connectivity=connections_layer)
        self.genes_genes_tanh = nn.Tanh()
        self.genes_genes_batchnorm = nn.BatchNorm1d(input_terms)
           
    def genes_layer(self, genes_terms_pairs, p_drop_genes, gene2id):
        # Define the layer of terms with genes, each pair is repeated n times (for the n neurons)
        
        term2id = create_index(genes_terms_pairs[:,0])

        self.term_dim = len(term2id)

        # Change term and genes to its indexes
        rows = [term2id[term] for term in genes_terms_pairs[:,0]]
        columns = [gene2id[gene] for gene in genes_terms_pairs[:,1]]

        data = np.ones(len(rows))

        # Create sparse matrix of terms connected to genes 
        genes_terms = sparse.coo_matrix((data, (rows, columns)), shape=(self.term_dim, self.gene_dim_ont))

        # Add n neurons to each term
        genes_terms_more_neurons = sparse.lil_matrix((self.term_dim*self.num_neurons_per_GO, self.gene_dim_ont))
        genes_terms = genes_terms.tolil()
        # Repeat the rows of the sparse matrix to match the 6 neurons
        row=0
        for i in range(genes_terms_more_neurons.shape[0]):
            if (i != 0) and (i%self.num_neurons_per_GO) == 0 :
                row=row+1
            genes_terms_more_neurons[i,:]=genes_terms[row,:]

        # Get the indexes of the matrix to define the connections of the sparse layer
        rows_more_neurons = torch.from_numpy(sparse.find(genes_terms_more_neurons)[0]).view(1,-1).long().to(self.device)
        columns_more_neurons = torch.from_numpy(sparse.find(genes_terms_more_neurons)[1]).view(1,-1).long().to(self.device)
        connections_layer1 = torch.cat((rows_more_neurons, columns_more_neurons), dim=0) # connections of the first layer each gene-term pair is repeated n times

        input_terms = len(gene2id)
        output_terms = self.num_neurons_per_GO*len(term2id) # n * GOterms
        self.genes_terms_sparse_linear_1 = SparseLinearNew(input_terms, output_terms, connectivity=connections_layer1)
        self.genes_terms_batchnorm = nn.BatchNorm1d(input_terms)
        self.genes_terms_tanh = nn.Tanh()
        self.drop_0 = nn.Dropout(p_drop_genes)

        return term2id
    
    def terms_layer(self, input_id, layer_pairs, number,neurons_per_GO,p_drop_terms):

        output_id = create_index(layer_pairs[:,0])

        # change term and genes to its indexes
        rows = [output_id[term] for term in layer_pairs[:,0]]
        columns = [input_id[term] for term in layer_pairs[:,1]]

        data = np.ones(len(rows))

        # Create sparse matrix of terms connected to terms
        connections_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(len(output_id), len(input_id)))

        # Add the n neurons with kronecker
        ones = sparse.csr_matrix(np.ones([neurons_per_GO, self.num_neurons_per_GO], dtype = int))
        connections_matrix_more_neurons = sparse.csr_matrix(sparse.kron(connections_matrix, ones))

        # Find the rows and columns of the connections
        rows_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[0]).view(1,-1).long().to(self.device)
        columns_more_neurons = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[1]).view(1,-1).long().to(self.device)
        connections = torch.cat((rows_more_neurons, columns_more_neurons), dim=0)

        input_terms = self.num_neurons_per_GO*len(input_id)
        output_terms = neurons_per_GO*len(output_id)
        self.add_module('GO_terms_sparse_linear_'+number, SparseLinearNew(input_terms, output_terms, connectivity=connections))
        self.add_module('drop_'+number, nn.Dropout(p_drop_terms))
        self.add_module('GO_terms_tanh_'+number, nn.Tanh())
        self.add_module('GO_terms_batchnorm_'+number, nn.BatchNorm1d(input_terms))
        return output_id
    
    def construct_NN_drug(self,p_drop_drugs):
        input_size = self.drug_dim  # Initialize input size based on drug dimensions

        for i in range(len(self.num_neurons_drug)):
            self.add_module('drug_linear_layer_' + str(i+1), nn.Linear(input_size, self.num_neurons_drug[i]))
            self.add_module('drug_drop_' + str(i+1),nn.Dropout(p_drop_drugs)) # Add a dropout layer with the given probability to prevent overfitting
            self.add_module('drug_tanh_' + str(i+1), nn.Tanh())
            self.add_module('drug_batchnorm_layer_' + str(i+1), nn.BatchNorm1d(input_size)) # Add batch normalization to stabilize learning
            
            # Update the input size for the next layer based on the current layer's number of neurons
            input_size = self.num_neurons_drug[i]
            
    def forward(self, x):
        if self.input_type in("mutations","expression"): 
            gene_input = x.narrow(1, 0, self.gene_dim_ont) # features of genes (Returns a new tensor that is a narrowed version)
            drug_input = x.narrow(1, self.gene_dim_ont, self.drug_dim) # features of drugs
        elif self.input_type  == "multiomics":
            multiomic_input = x.narrow(1, 0, self.gene_dim_multiomics) # features of genes (Returns a new tensor that is a narrowed version)
            drug_input = x.narrow(1, self.gene_dim_multiomics, self.drug_dim) # features of drugs
            # (0) Layer of genes with genes + tanh
            # batch --> dropout --> dense --> activation
            gene_input = self.genes_genes_batchnorm(multiomic_input)
            # gene_input = multiomic_input # Remove batchnorm 
            gene_input  = self.genes_genes_tanh(self.genes_genes_sparse_linear(gene_input))

        # define forward function for GO terms and genes #############################################
        # (1) Layer 1 + tanh
        # batch --> dropout --> dense --> activation
        gene_output = self.genes_terms_batchnorm(gene_input)
        gene_output = self.drop_0(gene_output)
        terms_output  = self.genes_terms_tanh(self.genes_terms_sparse_linear_1(gene_output))

        # (2...) Layer 2,3,4... + tanh
        for i in range(1,len(self.layer_connections)):
            # batch --> dropout --> dense --> activation
            terms_output = self._modules['GO_terms_batchnorm_'+str(i)](terms_output)
            terms_output = self._modules['drop_'+str(i)](terms_output)
            terms_output =  self._modules['GO_terms_tanh_'+str(i)](self._modules['GO_terms_sparse_linear_'+str(i)](terms_output))

        # define forward function for drugs #################################################
        drug_out = drug_input

        for i in range(1, len(self.num_neurons_drug)+1, 1):
            # batch --> dropout --> dense --> activation
            drug_out = self._modules['drug_batchnorm_layer_'+str(i)](drug_out)
            drug_out = self._modules['drug_drop_'+str(i)](drug_out)
            drug_out = self._modules['drug_tanh_'+str(i)](self._modules['drug_linear_layer_' + str(i)](drug_out))

        # connect two neural networks #################################################
        final_input = torch.cat((terms_output, drug_out), 1)

        # batch --> dropout --> dense --> activation
        output = self.final_batchnorm_layer(final_input)
        output = self.drop_final(output)
        output = self.final_tanh(self.final_linear_layer(output))
        output = self.final_aux_batchnorm_layer(output)
        output = self.drop_aux_final(output)
        output = self.final_aux_tanh(self.final_aux_linear_layer(output))
        final_output = self.final_linear_layer_output(output)
        return final_output
            
    def on_train_epoch_start(self):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()
                
        # Calculate the new learning rate using the provided formula
        learning_rate = self.initial_learning_rate * (1 / (1 + self.decay_rate * self.current_epoch))  # or epoch * epoch
        # Apply the new learning rate to the optimizer
        self.trainer.optimizers[0].param_groups[0]['lr'] = learning_rate  # Set the new learning rate

        # Optional: Print the new learning rate for debugging
        print(f'Epoch {self.current_epoch}: Learning rate adjusted to {learning_rate:.6f}')
    
    def on_test_epoch_start(self):
        # Record the start time of the epoch
        self.epoch_start_time = time.time()
        
        # Extract the metric from the checkpoint path
        if self.ckpt_path is not None:
            # Assuming the checkpoint path is structured like "output_folder/fold/best_model_d.ckpt"
            # Extract the metric from the filename
            _, filename = os.path.split(self.ckpt_path)  # Get the filename from the path
            self.metric = filename.split('_')[-1].split('.')[0]  # Get the last part before the extension
        else:
            self.metric = "other"  # Fallback if ckpt_path is not set   

    def configure_optimizers(self):
        # falta añadir lo del learning rate, el decay_rate
        # Define the optimizer
        if self.optimizer_type=='sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.initial_learning_rate, momentum=self.momentum)
        elif self.optimizer_type=='adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate, betas=(0.9, 0.99), eps=1e-05)
        return optimizer
    
    def build_input_vector(self, input_data):
        """
        Build an input vector for training by combining cell and drug features based on the input data.

        Parameters:
        - input_data (torch.Tensor): A tensor containing indices that reference specific cell and drug features. 
                                    Each row corresponds to a specific sample, where the first column 
                                    contains the index for the cell features and the second column 
                                    contains the index for the drug features.

        Returns:
        - torch.Tensor: A tensor of shape (num_samples, num_features), where num_features is the 
                        sum of the number of cell features and drug features. Each row corresponds 
                        to the combined features of a specific cell and drug pair.
        """
    
        # Initialize a numpy array to hold the combined features for each input sample
        features = np.zeros((input_data.size()[0], (self.gene_dim_input + self.drug_dim)))

        # Iterate over each sample in the inputdata
        for i in range(input_data.size()[0]):
            # Concatenate the cell features and drug features based on the indices specified in inputdata
            features[i] = np.concatenate(
                (self.cell_features[int(input_data[i, 0])], self.drug_features[int(input_data[i, 1])]),
                axis=None  # Concatenate along the first axis (flatten)
            )

        # Convert the numpy array to a PyTorch tensor of type float
        features = torch.from_numpy(features).float().to(self.device)
        
        # Return the constructed feature tensor
        return features
    
    def configure_loss(self):
        # Define the loss criterion based on the configuration
        if self.loss_type=='MSELoss':
            self.criterion = nn.MSELoss()
            self.test_model = '/best_model_d.pt' # Test model is spearman
        elif self.loss_type=='L1Loss':
            self.criterion = nn.L1Loss()
            self.test_model = '/best_model_p.pt' # Test model is pearson

class SparseGODataModule(LightningDataModule):

    def __init__(self, input_folder, fold, input_type, cell2id_mapping_file, drug2id_mapping_file, gene2id_mapping_ont_file, ontology_file, genotype_file, fingerprint_file, train_file, val_file, test_file, multiomics_layer, gene2id_mapping_multiomics_file="None", batch_size=1000, num_workers=4):
        super().__init__()
        
        # Set the input folder path
        self.fold = fold
        self.input_folder = input_folder + self.fold + "/"
        self.input_type = input_type
        
        # Set batch size and number of workers
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set file paths for mappings and data
        self.cell2id_mapping = self.load_mapping(self.input_folder + cell2id_mapping_file)  # Load cell to ID mapping
        self.drug2id_mapping = self.load_mapping(self.input_folder + drug2id_mapping_file)  # Load drug to ID mapping
        self.gene2id_mapping_ont_file = self.input_folder + gene2id_mapping_ont_file  # Path for gene to ID mapping for ontology
        self.ontology_file = self.input_folder + ontology_file  # Path for ontology file
        self.genotype_file = self.input_folder + genotype_file  # Path for genotype file
        self.fingerprint_file = self.input_folder + fingerprint_file  # Path for fingerprint file
        self.train_file = self.input_folder + train_file  # Path for training data
        self.val_file = self.input_folder + val_file  # Path for validation data
        self.test_file = self.input_folder + test_file  # Path for test data
        
        # Multiomics settings
        if self.input_type == "multiomics":
            self.gene2id_mapping_multiomics_file = self.input_folder + gene2id_mapping_multiomics_file  # Path for multiomics gene to ID mapping
            self.multiomics_layer_file = self.input_folder + multiomics_layer  # Path for multiomics layer file
        
    def prepare_data(self):
        print("Preparing data...")
        
        # Load ontology: create the graph of connected GO terms
        since1 = time.time()
        self.gene2id_mapping_ont = self.load_mapping(self.gene2id_mapping_ont_file)
        dG, terms_pairs, genes_terms_pairs = self.load_ontology(self.ontology_file, self.gene2id_mapping_ont)
        time_elapsed1 = time.time() - since1
        print('Load ontology complete in {:.0f}m {:.0f}s'.format(
        time_elapsed1 // 60, time_elapsed1 % 60))
        
        # Layer connections contains the pairs on each layer (including virtual nodes)
        since2 = time.time()
        sorted_pairs, level_list, level_number = sort_pairs(genes_terms_pairs, terms_pairs, dG, self.gene2id_mapping_ont)
        self.layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)
        time_elapsed2 = time.time() - since2
        print('\nLayer connections complete in {:.0f}m {:.0f}s'.format(
        time_elapsed2 // 60, time_elapsed2 % 60))
        
        # Load cell/drug features
        self.cell_features = np.genfromtxt(self.genotype_file, delimiter=',')
        self.drug_features = np.genfromtxt(self.fingerprint_file, delimiter=',')
        self.drug_dim = len(self.drug_features[0,:])
        
        # For the multiomics network
        if self.input_type == "multiomics":
            self.gene2id_mapping_multiomics = self.load_mapping(self.gene2id_mapping_multiomics_file)  # Load multiomics gene to ID mapping
            self.genes_genes_pairs = np.loadtxt(self.multiomics_layer_file, dtype=str)  # Load multiomics layer pairs
        else:
            self.gene2id_mapping_multiomics = None  # Set to None if not multiomics
            self.genes_genes_pairs = None  # Set to None if not multiomics

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
        
            self.train_data = self.prepare_features_and_labels(self.train_file)
            self.val_data = self.prepare_features_and_labels(self.val_file)

            # Print the shapes of the resulting tensors and layers details only on rank 0
            if self.trainer.is_global_zero:
                print("Train set shape: Features = {}, Samples = {}".format(self.train_data[0].shape, len(self.train_data)))
                print("Validation set shape: Features = {}, Samples = {}".format(self.val_data[0].shape, len(self.val_data)))
                

        if stage == 'test' or stage is None:

            self.test_data = self.prepare_features_and_labels(self.test_file)
            
            if self.trainer.is_global_zero:
                print("Test set shape: Features = {}, Samples = {}".format(self.test_data[0].shape, len(self.test_data)))
                       
    def prepare_features_and_labels(self, file_name):
        """
        Loads training data from a specified file and maps cell and drug identifiers 
        to their corresponding IDs using the provided mappings in this data module.

        Parameters
        ----------
        file_name: str
            Path to the training data file that contains cell and drug identifiers and labels.

        Returns
        -------
        feature: torch.Tensor
            A tensor containing features for each training example, where each feature is a 
            list consisting of [cell_id, drug_id].

        label: torch.Tensor
            A tensor containing labels for each training example, where each label is a 
            list consisting of the target values.
        """
        
        feature = []  # Initialize an empty list to hold features
        label = []    # Initialize an empty list to hold labels

        # Open the specified file for reading
        with open(file_name, 'r') as fi:
            # Iterate over each line in the file
            for line in fi:
                # Strip whitespace and split the line into tokens based on tab characters
                tokens = line.strip().split('\t')
                
                # Process the line based on the number of tokens
                if len(tokens) == 3:
                    # If there are three tokens, append the corresponding IDs and label
                    feature.append([self.cell2id_mapping[tokens[0]], self.drug2id_mapping[tokens[1]]])  # Map cell and drug names to IDs
                    label.append([float(tokens[2])])  # Convert the label to float and append it
                elif len(tokens) == 4:  # If there are four tokens (indicating two output neurons)
                    # Append the corresponding IDs and both labels
                    feature.append([self.cell2id_mapping[tokens[0]], self.drug2id_mapping[tokens[1]]])  # Map cell and drug names to IDs
                    label.append([float(tokens[2]), float(tokens[3])])  # Convert both labels to float and append them

        # Convert the accumulated features and labels to tensors before returning
        return torch.Tensor(feature), torch.Tensor(label)

    def load_mapping(self, mapping_file):
        """
        Opens a txt file with two columns and saves the second column as the key of the dictionary 
        and the first column as a value.
        """
        
        # Initialize an empty dictionary to hold the mapping
        mapping = {}  # Dictionary that will store the mapping from the file

        # Open the specified mapping file for reading
        with open(mapping_file) as file_handle:  # Use a context manager to open the file
            # Iterate over each line in the file
            for line in file_handle:
                # Remove trailing whitespace and split the line into columns based on whitespace
                line = line.rstrip().split()  # Split the line into a list; e.g., ['3007', 'ZMYND8']
                
                # Save the second column (gene/drug name) as the key and the first column (index) as the value
                mapping[line[1]] = int(line[0])  # Key is the gene/drug name; value is the index as an integer

        return mapping  # Return the completed mapping

    def train_dataloader(self):
        '''returns training dataloader'''
        start_time = time.time()
        train_feature, train_label  = self.train_data
        train_data_loader = DataLoader(du.TensorDataset(train_feature,train_label), 
                                       batch_size=self.batch_size, 
                                       num_workers=self.num_workers, # Can't pin memory because it is not available for sparseCUDA 
                                       )
        end_time = time.time()
        # print(f"Train DataLoader created with {self.num_workers} workers in {end_time - start_time:.2f} seconds.")
        return train_data_loader

    def val_dataloader(self):
        '''returns validation dataloader'''
        start_time = time.time()
        val_feature, val_label   = self.val_data
        val_data_loader = DataLoader(du.TensorDataset(val_feature,val_label), 
                                     batch_size=self.batch_size, 
                                     num_workers=self.num_workers,
                                     )
        end_time = time.time()
        # print(f"Validation DataLoader created with {self.num_workers} workers in {end_time - start_time:.2f} seconds.")
        return val_data_loader

    def test_dataloader(self):
        '''returns test dataloader'''
        start_time = time.time()
        test_feature, test_label   = self.test_data
        test_data_loader = DataLoader(
        du.TensorDataset(test_feature, test_label), 
        batch_size=10000,  # Use a large batch size to ensure all samples are included for predictions.
                        # This is essential for making the test results comparable, as smaller batch sizes can lead to insufficient data for some drugs.
        num_workers=self.num_workers,
        )

        end_time = time.time()
        # print(f"Test DataLoader created with {self.num_workers} workers in {end_time - start_time:.2f} seconds.")
        return test_data_loader 

    def load_ontology(self, ontology_file, gene2id_mapping):
        """
        Creates the directed graph of the GO terms and stores the connected elements in arrays.

        Parameters
        ----------
        ontology_file: str
            Path to the ontology file containing GO terms and their connections.

        gene2id_mapping: dict
            A dictionary mapping gene names to their corresponding IDs.

        Returns
        -------
        dG: networkx.classes.digraph.DiGraph
            Directed graph of all terms.

        terms_pairs: numpy.ndarray
            Store the connection between a term and a term.

        genes_terms_pairs: numpy.ndarray
            Store the connection between a gene and a term.
        """

        dG = nx.DiGraph()  # Initialize a directed graph for GO terms

        file_handle = open(ontology_file)  # Open the file that contains genes and GO terms

        terms_pairs = []  # Initialize a list to store pairs of connected terms
        genes_terms_pairs = []  # Initialize a list to store pairs of genes and terms

        gene_set = set()  # Create a set to store unique genes (elements can't repeat)
        term_direct_gene_map = {}  # Mapping of terms to their direct associated genes
        term_size_map = {}  # Mapping of terms to the size of their associated gene sets

        # Iterate through each line in the ontology file
        for line in file_handle:
            line = line.rstrip().split()  # Remove trailing spaces and split the line into a list

            # Check if the third element in the line is 'default'
            if line[2] == 'default':  # If the third element is 'default', connect the terms in the graph
                dG.add_edge(line[0], line[1])  # Add a directed edge between term line[0] and term line[1]
                terms_pairs.append([line[0], line[1]])  # Store the term connection pair in terms_pairs
            else:
                # Skip the gene if it is not part of the gene2id_mapping
                if line[1] not in gene2id_mapping:  # Check if the gene is in the mapping
                    print(line[1])  # Print the gene name that is being skipped
                    continue  # Skip to the next line if the gene is not found in the mapping

                genes_terms_pairs.append([line[0], line[1]])  # Store the gene-term connection pair

                # Check if the term is already in the direct gene mapping
                if line[0] not in term_direct_gene_map:  # If the term is not in the mapping, add it
                    term_direct_gene_map[line[0]] = set()  # Create a new set for the term

                term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])  # Add the gene ID to the set for this term
                gene_set.add(line[1])  # Add the gene to the total set of genes

        # Convert the pairs lists to NumPy arrays for further processing
        terms_pairs = np.array(terms_pairs)  # Convert the term pairs to a 2D NumPy array
        genes_terms_pairs = np.array(genes_terms_pairs)  # Convert the gene-term pairs to a 2D NumPy array

        file_handle.close()  # Close the file handle to free up system resources

        print('There are', len(gene_set), 'genes')  # Print the total number of unique genes

        # Iterate through each term in the directed graph
        for term in dG.nodes():
            term_gene_set = set()  # Create a set to hold genes associated with the term

            if term in term_direct_gene_map:
                term_gene_set = term_direct_gene_map[term]  # Get the set of genes connected to this term

            deslist = nx.descendants(dG, term)  # Get all descendant GO terms for the current term

            # For each descendant term, add associated genes to the term_gene_set
            for child in deslist:
                if child in term_direct_gene_map:  # Check if the child term has associated genes
                    term_gene_set = term_gene_set | term_direct_gene_map[child]  # Union of both sets

            # Check if the term_gene_set is empty, indicating no genes connected to the term
            if len(term_gene_set) == 0:
                print('There is an empty term, please delete term:', term)  # Alert that the term has no genes
                sys.exit(1)  # Exit the program with an error code
            else:
                # Store the number of genes associated with the term (including descendants)
                term_size_map[term] = len(term_gene_set)  # Count of genes in this term

        # Identify root nodes (terms with no incoming edges)
        leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]  # Find all root terms

        uG = dG.to_undirected()  # Convert the directed graph to an undirected graph
        connected_subG_list = list(nx.connected_components(uG))  # List all connected components in the undirected graph

        # Verify the integrity of the constructed graph
        print('There are', len(leaves), 'roots:', leaves[0])  # Print the number of roots and the first root
        print('There are', len(dG.nodes()), 'terms')  # Print the total number of terms in the graph
        print('There are', len(connected_subG_list), 'connected components')  # Print the number of connected components

        # Check for multiple roots and connected components
        if len(leaves) > 1:
            print('There are more than one root of ontology. Please use only one root.')  # Alert for multiple roots
            sys.exit(1)  # Exit the program with an error code

        if len(connected_subG_list) > 1:
            print('There are more than one connected components. Please connect them.')  # Alert for multiple components
            sys.exit(1)  # Exit the program with an error code

        return dG, terms_pairs, genes_terms_pairs  # Return the directed graph and the pairs of terms and genes

if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK")=="0":
        # Make sure all GPUs are empty
        print_gpu_memory_info()

    # 1. Argument parser setup
    mac = "/Users/katyna/Library/CloudStorage/OneDrive----/"
    windows = "C:/Users/ksada/OneDrive - Tecnun/"
    manitou = "/manitou/pmg/users/ks4237/" 
    atlas = "/scratch/ksada/"
    computer = mac # CHANGE
    
    input_folder = computer + "SparseGO_lightning/data/PDCs_multiomics_LELO/"
    output_folder = computer+ "SparseGO_lightning/results/PDCs_multiomics_LELO/"

    parser = argparse.ArgumentParser(description='Train Autoencoder')

    parser.add_argument('-project', help='W&B project name', type=str, default="Test_error")
    parser.add_argument('-entity', help='W&B entity', type=str, default="miramon_team")
    parser.add_argument('-tags', help='Tags of type of data or/and model we are testing', type=list_of_strings, default=['normal'])
    parser.add_argument('-job_type', help='Job type', type=str, default="job-test")

    parser.add_argument('-input_folder', help='Directory containing the input data folders.', type=str, default=input_folder)
    parser.add_argument('-output_folder', help='Directory containing the folders that have the resulting models', type=str, default=output_folder)
    parser.add_argument('-fold', help='Folder to analyze', type=str, default="samples2")

    parser.add_argument('-train', help='Training dataset', type=str, default="sparseGO_train.txt") 
    parser.add_argument('-validation', help='Validation dataset', type=str, default="sparseGO_val.txt")
    parser.add_argument('-test', help='Dataset to be predicted', type=str, default="sparseGO_test.txt")

    parser.add_argument('-input_type', help='Type of omics data used', type=str, default="multiomics")
    parser.add_argument('-ontology', help='Ontology file used to guide the neural network', type=str, default="sparseGO_ont.txt")
    parser.add_argument('-multiomics_layer', help='Genes to genes layer', type=str, default="multiomics_layer.txt")
    parser.add_argument('-gene2id_multiomics', help='Gene to ID mapping file', type=str, default="gene2ind_multiomics.txt")
    parser.add_argument('-gene2id_ont', help='Gene to ID mapping file', type=str, default="gene2ind_ont.txt")
    parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default="drug2ind.txt")
    parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default="cell2ind.txt")
    parser.add_argument('-genotype', help='Mutation information for cell lines', type=str, default="cell2multiomics.txt")
    parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str, default="drug2fingerprint.txt")

    parser.add_argument('-hyperparameters_file', help='Result file name', type=str, default=output_folder+"hyperparams.txt")
    parser.add_argument('-epochs', help='Training epochs for training', type=int, default=10)
    parser.add_argument('-num_workers', help='DataLoader number of workers', type=int, default=0)
    parser.add_argument('-min_delta', help='Minimum change to qualify as an improvement (for early stopping)', type=float, default=0.001)
    parser.add_argument('-patience', help='How many epochs to wait after the last improvement (for early stopping)', type=int, default=30)
    parser.add_argument('-precision', help='Precision of the Trainer', type=str, default="32-true")
    parser.add_argument('-strategy', help='Strategy of the Trainer', type=str, default="auto")
    parser.add_argument('-early_stopping_metric', help='Early stopping callback metric', type=str, default="validation_corr_per_drug")


    # Parse the known arguments
    #opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    # 2. Check the environment variables
    env_vars = {
        'RANK': 'None',
        'WORLD_SIZE': 'None',
        'LOCAL_RANK': 'None',
        'MASTER_ADDR': 'None', # hostname,
        'MASTER_PORT': 'None', # str(port)
        'SLURM_NTASKS': 'None'
    }

    # Print the values 
    print("\nEnvironment variables:")
    for var in env_vars.keys():
        # Use get() to avoid KeyError
        print(f"{var} = {os.environ.get(var, 'Not Set')}")

    # 3. Ensure that training is as reproducible as possible
    seed=123
    torch.manual_seed(seed) # Random seed for CPU tensors
    torch.cuda.manual_seed_all(seed) # Seed for generating random numbers for the current GPU
    torch.backends.cudnn.deterministic = True #  Is done to ensure that the computations are deterministic, meaning they will produce the same results given the same input.
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8' # Set the environment variable for deterministic behavior
    # torch.use_deterministic_algorithms(True)
    pl.seed_everything(1, workers=seed)

    # Set PyTorch print options
    torch.set_printoptions(precision=5, threshold=10_000)

    # Faster, but less precise
    torch.set_float32_matmul_precision("high")

    sweep_config_1 = {
    'method': 'bayes', #bayes, random, grid

    'metric': {
      'name': 'av_spearman_drugs_spmodel',
      'goal': 'maximize'
    },

    'parameters': {
        # 'epochs': {
        #     'value': 10
        # },
        'batch_size': {
            'values': [2000]
        },
        'learning_rate': {
            'value': 0.01
        },
        'optimizer': {
            'value': 'sgd'
        },
        'decay_rate': {
            'value': 0.02
        },
        'loss_type': {
            'value': 'MSELoss'
        },
        'momentum': {
            'value':0.88
        },
        'num_neurons_per_GO': {
            'value': 5
        },
        'num_neurons_final_GO': {
            'value': 6
        },
        'drug_neurons': {
            'value':list(map(int, '100,50,25'.split(',')))
        },
        'num_neurons_final': {
            'value': 12
        },
        'p_drop_genes': {
            'value': 0
        },
        'p_drop_terms': {
            'value': 0
        },
        'p_drop_drugs': {
            'value': 0
        },
        'p_drop_final': {
            'value': 0
        },

    }
}
    # Load the hyperparameters from the text file
    hyperparams_file = opt.hyperparameters_file
    hyperparameters = load_hyperparameters(hyperparams_file)

    # Now you can use the hyperparameters in your config
    config = sweep_config_1["parameters"]
    
    #  Example of setting values from the hyperparameters
    config["batch_size"]['value'] = int(hyperparameters["batch_size"])
    config["learning_rate"]['value'] = float(hyperparameters["learning_rate"])
    config["optimizer"]['value'] = hyperparameters["optimizer_type"]
    config["momentum"]['value'] = float(hyperparameters["momentum"])
    config["num_neurons_per_GO"]['value'] = int(hyperparameters["num_neurons_per_GO"])
    config["num_neurons_final_GO"]['value'] = int(hyperparameters["num_neurons_final_GO"])
    config["drug_neurons"]['value'] = list(map(int, [x for x in  hyperparameters["drug_neurons"].split(',') if x]))  # Ensure to convert values to int
    config["num_neurons_final"]['value'] = int(hyperparameters["num_neurons_final"])
    config["p_drop_genes"]['value'] = float(hyperparameters["p_drop_genes"])
    config["p_drop_terms"]['value'] = float(hyperparameters["p_drop_terms"])
    config["p_drop_drugs"]['value'] = float(hyperparameters["p_drop_drugs"])
    config["p_drop_final"]['value'] = float(hyperparameters["p_drop_final"])
    
    if len(config["drug_neurons"]['value']) == 0:
        # Add "no_drugs_branch" tag if the condition is met
        opt.tags.append("no_drugs_branch")
        
    fold = opt.fold
    # Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
    wandb_logger = WandbLogger(log_model=True, project=opt.project, name=fold, tags=opt.tags, job_type=opt.job_type, checkpoint_name="checkpoint_callback_d")

    # Initialize your criterion and data module
    sparseGO_data = SparseGODataModule(
        input_folder=opt.input_folder, 
        fold=fold, 
        input_type=opt.input_type, 
        cell2id_mapping_file=opt.cell2id, 
        drug2id_mapping_file=opt.drug2id, 
        gene2id_mapping_ont_file=opt.gene2id_ont, 
        ontology_file=opt.ontology, 
        genotype_file=opt.genotype, 
        fingerprint_file=opt.fingerprint, 
        train_file=opt.train, 
        val_file=opt.validation, 
        test_file=opt.test, 
        multiomics_layer=opt.multiomics_layer, 
        gene2id_mapping_multiomics_file=opt.gene2id_multiomics, 
        batch_size=config["batch_size"]['value'],  # Use the configured batch size
        num_workers=opt.num_workers
    )
    sparseGO_data.prepare_data()

    # Initialize your model with values from the config and hyperparameters
    model = SparseGO(
        output_folder=opt.output_folder, 
        fold=sparseGO_data.fold, 
        input_type=sparseGO_data.input_type, 
        num_neurons_per_GO=config["num_neurons_per_GO"]['value'], 
        num_neurons_per_final_GO=config["num_neurons_final_GO"]['value'],  
        num_neurons_drug=config["drug_neurons"]['value'],   
        num_neurons_final=config["num_neurons_final"]['value'],  
        layer_connections=sparseGO_data.layer_connections, 
        gene2id_mapping_ont=sparseGO_data.gene2id_mapping_ont,  
        p_drop_final=config["p_drop_final"]['value'],  
        p_drop_genes=config["p_drop_genes"]['value'],  
        p_drop_terms=config["p_drop_terms"]['value'],  
        p_drop_drugs=config["p_drop_drugs"]['value'],  
        learning_rate=config["learning_rate"]['value'],  
        decay_rate=config["decay_rate"]['value'],  # Assuming decay_rate is defined elsewhere
        optimizer_type=config["optimizer"]['value'],  
        momentum=config["momentum"]['value'],  
        loss_type=config["loss_type"]['value'],  # Assuming loss_type is defined elsewhere
        cell_features=sparseGO_data.cell_features,  # Assuming cell_features is defined elsewhere
        drug_features=sparseGO_data.drug_features,  # Assuming drug_features is defined elsewhere
        genes_genes_pairs=sparseGO_data.genes_genes_pairs,  # Optional, set to None
        gene2id_mapping_multiomics=sparseGO_data.gene2id_mapping_multiomics  # Optional, set to None
    )
    
    if os.environ.get("LOCAL_RANK")=="0":
        wandb_logger.experiment.config["batch_size"] = config["batch_size"]['value']
        wandb_logger.experiment.config["learning_rate"] = config["learning_rate"]['value']
        wandb_logger.experiment.config["decay_rate"] = config["decay_rate"]['value']
        wandb_logger.experiment.config["optimizer_type"] = config["optimizer"]['value']
        wandb_logger.experiment.config["loss_type"] = config["loss_type"]['value']
        wandb_logger.experiment.config["momentum"] = config["momentum"]['value']
        wandb_logger.experiment.config["num_neurons_per_GO"] = config["num_neurons_per_GO"]['value']
        wandb_logger.experiment.config["num_neurons_final_GO"] = config["num_neurons_final_GO"]['value']
        wandb_logger.experiment.config["drug_neurons"] = config["drug_neurons"]['value']
        wandb_logger.experiment.config["num_neurons_final"] = config["num_neurons_final"]['value']
        wandb_logger.experiment.config["p_drop_genes"] = config["p_drop_genes"]['value']
        wandb_logger.experiment.config["p_drop_terms"] = config["p_drop_terms"]['value']
        wandb_logger.experiment.config["p_drop_drugs"] = config["p_drop_drugs"]['value']
        wandb_logger.experiment.config["p_drop_final"] = config["p_drop_final"]['value']
        wandb_logger.experiment.config["batch_size"] = config["batch_size"]['value']
        wandb_logger.experiment.config["max_epochs"] = opt.epochs
        wandb_logger.experiment.config["precision"] = opt.precision
        wandb_logger.experiment.config["num_workers"] = opt.num_workers
        wandb_logger.experiment.config["strategy"] = opt.strategy
        wandb_logger.experiment.config["input_type"] = opt.input_type
        wandb_logger.experiment.config["fold"] = sparseGO_data.fold
    
    # Define the EarlyStopping callback
    opt.patience=25
    early_stopping = EarlyStopping(
        monitor=opt.early_stopping_metric,    # The metric to monitor validation_corr_per_drug
        min_delta=opt.min_delta,        # Minimum change to qualify as an improvement
        patience=opt.patience,            # How many epochs to wait after the last improvement
        verbose=True,          # Print messages when stopping
        mode='max'             # 'min' for loss, 'max' for accuracy
    )
    # device_stats = DeviceStatsMonitor(cpu_stats=True)
    # Create a checkpoint callback
    checkpoint_callback_d = ModelCheckpoint(
        monitor='validation_corr_per_drug',  # Replace with your metric
        dirpath=opt.output_folder+fold,  # Custom directory for checkpoints
        filename='best_model_d',  # Customize filename
        verbose=True,
        save_top_k=1,
        mode='max',  # 'min' or 'max' based on your metric
        enable_version_counter=False,
        save_weights_only=False
    )
    
    checkpoint_callback_s = ModelCheckpoint(
        monitor='validation_corr_spearman',  # Replace with your metric
        dirpath=opt.output_folder+fold,  # Custom directory for checkpoints
        filename='best_model_s',  # Customize filename
        verbose=True,
        save_top_k=1,
        mode='max',  # 'min' or 'max' based on your metric
        enable_version_counter=False,
        save_weights_only=False
    )
    
    trainer = Trainer(
        accelerator="cpu",                # Use CPU
        #accelerator="gpu",                # Use GPU
        devices= 1, # Extract GPUs per node int(os.environ.get('SLURM_NTASKS'))
        num_nodes= 1,  # Extract number of nodes int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        max_epochs=opt.epochs,  # Number of epochs opt.epochs
        strategy=opt.strategy,         # Distributed strategy ddp_notebook
        logger=wandb_logger,             # Logger for tracking experiments
        precision=opt.precision,             # Mixed precision training "16-mixed"
        min_epochs=20,                  # Minimum number of epochs (default None)
        log_every_n_steps=1,            # Log every 50 steps
        check_val_every_n_epoch=1,       # Validation check every epoch
        deterministic=False,               # Set to True for reproducibility
        callbacks=[early_stopping,checkpoint_callback_d,checkpoint_callback_s],       # Add the early stopping callback
        max_steps=-1,                     # Maximum number of steps (default -1, means no limit)
        min_steps=None,                   # Minimum number of steps (default None)
        max_time=None,                    # Maximum training time (default None)
        limit_train_batches=None,         # Limit the number of training batches (default None)
        limit_val_batches=None,           # Limit the number of validation batches (default None)ss
        limit_test_batches=None,          # Limit the number of test batches (default None)
        limit_predict_batches=None,       # Limit the number of prediction batches (default None)
        overfit_batches=0.0,             # Overfit on a fraction of batches (default 0.0)
        val_check_interval=None,          # Check validation every X epochs (default None)
        num_sanity_val_steps=None,        # Number of sanity validation steps (default None)
        gradient_clip_val=None,           # Gradient clipping value (default None)
        gradient_clip_algorithm=None,     # Gradient clipping algorithm (default None)
        accumulate_grad_batches=1,        # Accumulate gradients over multiple batches (default 1) -- In this case, if your global batch size is 20,000 and you set accumulate_grad_batches=4, each GPU will still receive 5,000 samples per mini-batch, but the optimizer will only perform an update after processing 4 mini-batches, effectively simulating a global batch size of 20,000.
        benchmark=None,                   # Enables benchmarking if True (default None)
        inference_mode=True,              # Whether to run in inference mode (default True)
        use_distributed_sampler=True,     # Use distributed sampler (default True)
        # profiler=PyTorchProfiler(dirpath=opt.result,filename="profiler.txt"),                    # Profiler for performance tracking (default None)
        detect_anomaly=False,             # Detect anomalies in training (default False)
        barebones=False,                  # Use barebones (default False)
        # plugins=SLURMEnvironment(auto_requeue=False), # Custom plugins (default None)
        sync_batchnorm=True,             # Synchronize batch normalization (default False) PROBAR CON ESTO EN FALSE
        reload_dataloaders_every_n_epochs=0,  # Reload the dataloaders every n epochs (default 0)
        default_root_dir=None             # Default directory for checkpoints (default None)
    )

    trainer.fit(model, sparseGO_data)
    
    # artifact = wandb.Artifact("means_spearman_drug",type="drug means")
    # artifact.add_file(output_folder+'spearman_means.txt')
    # run.log_artifact(artifact)
    
    # artifact = wandb.Artifact("means_spearman_drug",type="drug means")
    # artifact.add_file(output_folder+'spearman_means.txt')
    # run.log_artifact(artifact)
    
    # create the model
    # filepath = opt.output_folder+fold+"model.onnx"    
    # torch.onnx.dynamo_export(model,model.example_input_array)
    
    # trainer = Trainer(
    #     accelerator="gpu",                # Use GPU
    #     devices= 1, # Extract GPUs per node int(os.environ.get('SLURM_NTASKS'))
    #     num_nodes= 1,  # Extract number of nodes int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    #     logger=wandb_logger,             # Logger for tracking experiments
    # ) 
    model.ckpt_path = opt.output_folder+fold+"/best_model_d.ckpt"
    trainer.test(model, sparseGO_data)
    model.ckpt_path = opt.output_folder+fold+"/best_model_s.ckpt"
    trainer.test(model, sparseGO_data)