# #!/bin/bash
# nvidia-smi
# Activate your Python environment
# source activate /scratch/ksada/envs/SparseGOnew # CHANGE THIS - your environment
source activate  /Users/katyna/envs/SparseGOnew # CHANGE THIS - your environment

# Login to W&B
wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account

# Set OMP_NUM_THREADS to the value of SLURM_CPUS_PER_TASK
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the Python script with the required arguments
folder_name="PDCs_expression_LELO" 
entity="miramon_team"
project="Hyper_PDCs_CV_LELO_lightning_2" # CHANGE THIS - name you want to give to your W&B project
tags=selfieslatent184chembl
foldername="PDCs_expression_LELO" 

input_folder="../data/"$folder_name"/"
output_folder="../results/"$folder_name"/" # CHANGE THIS - folder to store results
mkdir $output_folder

epochs=200

num_workers=0
precision="32-true"
strategy="auto"

min_delta=0.001
patience=35 
early_stopping_metric="validation_corr_spearman"

# Define file paths for datasets, ontology, and other resources
input_type="expression" # CHANGE 
gene2id_ont_file="gene2ind.txt" # CHANGE gene2ind gene2ind_ont
genotype_file="cell2expression.txt" # CHANGE  cell2multiomics
fingerprint_file="drug2selfieslatent184_new.txt" # CHANGE drug2selfieslatent184 drug2fingerprint

train_file="sparseGO_train.txt"
validation_file="sparseGO_val.txt"
test_file="sparseGO_test.txt"
onto_file="sparseGO_ont.txt"
drug2id_file="drug2ind.txt"
cell2id_file="cell2ind.txt"
drugs_names_file="compound_names.txt"
multiomics_layer_file="multiomics_layer.txt"
gene2id_multiomics_file="gene2ind_multiomics.txt"

# Make output folder for each fold
samples_folders=samples1,samples2,samples3,samples4,samples5
for sample_folder in "samples1" "samples2" "samples3" "samples4" "samples5" 
do
  mkdir $output_folder$sample_folder
done

# Loop to execute the process 200 times
for i in {1..2}; do
    echo "Running iteration $i..."
    job_type=$(date +%s | md5sum | cut -c 1-8)  # Generates an 8-character random ID based on the current timestamp
    echo "Generated Job ID: $job_type"

    # Randomly select hyperparameters
    batch_size=$(shuf -e 25 52 100 200 300 400 500 1000 -n 1)
    learning_rate=$(shuf -e 0.0003 0.1 0.01 0.3 0.001 0.2 0.0001 0.05 -n 1)
    optimizer_type="sgd"  # Static value in this config
    decay_rate=$(shuf -e 0 0.01 0.0001 0.002 0.003 0.001 -n 1)
    loss_type="MSELoss"  # Static value
    momentum=$(shuf -e 0.75 0.8 0.85 0.88 0.9 0.95 -n 1)
    num_neurons_per_GO=$(shuf -e 5 6 -n 1)
    num_neurons_final_GO=$(shuf -e 6 12 24 -n 1)
    drug_neurons=$(shuf -e "," -n 1) # drug_neurons=$(shuf -e "80,40" "100," "60,40" "20," "30," "50," -n 1)
    num_neurons_final=$(shuf -e 12 18 16 20 -n 1)
    p_drop_genes=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_terms=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_drugs=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_final=$(shuf -e 0 0.05 0.1 0.15 -n 1)

    # Create a hyperparameter file
    hyperparams_file=$output_folder"/hyperparams.txt"

    # Write the selected hyperparameters to the file
    {
        echo "batch_size=$batch_size"
        echo "learning_rate=$learning_rate"
        echo "optimizer_type=$optimizer_type"
        echo "decay_rate=$decay_rate"
        echo "loss_type=$loss_type"
        echo "momentum=$momentum"
        echo "num_neurons_per_GO=$num_neurons_per_GO"
        echo "num_neurons_final_GO=$num_neurons_final_GO"
        echo "drug_neurons=$drug_neurons"
        echo "num_neurons_final=$num_neurons_final"
        echo "p_drop_genes=$p_drop_genes"
        echo "p_drop_terms=$p_drop_terms"
        echo "p_drop_drugs=$p_drop_drugs"
        echo "p_drop_final=$p_drop_final"
    } > "$hyperparams_file"

    # Print the selected hyperparameters (for logging)
    echo "Selected Hyperparameters written to $hyperparams_file"
    
    for sample_folder in "samples1" "samples2" "samples3" "samples4" "samples5" 
    do
        # Run the training script with torchrun and the specified hyperparameters $SLURM_NTASKS
            # --nproc_per_node=$SLURM_NTASKS \
            # --nnodes=$SLURM_JOB_NUM_NODES \
            # --node_rank=$SLURM_NODEID \
            # --master_addr=$(hostname) \
            # --master_port=$(shuf -i 20000-30000 -n 1) \
        torchrun \
            --nproc_per_node=1 \
            --nnodes=1 \
            ../code/train_cv.py \
            -project $project \
            -entity $entity \
            -tags $tags \
            -job_type $job_type \
            -input_folder $input_folder \
            -output_folder $output_folder \
            -fold $sample_folder \
            -train $train_file \
            -validation $validation_file \
            -test $test_file \
            -input_type $input_type \
            -ontotology $onto_file \
            -multiomics_layer $multiomics_layer_file \
            -gene2id_multiomics $gene2id_multiomics_file \
            -gene2id_ont $gene2id_ont_file \
            -drug2id $drug2id_file \
            -cell2id $cell2id_file \
            -genotype $genotype_file \
            -fingerprint $fingerprint_file \
            -hyperparameters_file $hyperparams_file \
            -precision $precision \
            -strategy $strategy \
            -patience $patience \
            -min_delta $min_delta \
            -num_workers $num_workers \
            -epochs $epochs \
            -early_stopping_metric $early_stopping_metric \
            > "$output_folder/train_cv_$i.log"
    done

    # CAREFUL -gene2id $gene2id_multiomics_file gene2id_ont_file
    python -u ../code/per_drug_correlation.py \
    -project $project  \
    -entity $entity  \
    -tags $tags \
    -job_type $job_type \
    -input_folder $input_folder \
    -output_folder $output_folder \
    -samples_folders $samples_folders \
    -sweep_name "final metrics drugs" \
    -model_name "best_model_d.ckpt" \
    -predictions_name "d_test_predictions.txt" \
    -labels_name $test_file \
    -genomics_name $genotype_file \
    -gene2id $gene2id_ont_file \
    -drug2id $drug2id_file \
    -cell2id $cell2id_file \
    -druginput_name $fingerprint_file \
    -drugs_names $drugs_names_file \
    -cuda 0
    > "$output_folder/metrics_$i.log"

    # CAREFUL -gene2id $gene2id_multiomics_file gene2id_ont_file
    python -u ../code/per_drug_correlation.py \
    -project $project  \
    -entity $entity  \
    -tags $tags \
    -job_type $job_type \
    -input_folder $input_folder \
    -output_folder $output_folder \
    -samples_folders $samples_folders \
    -sweep_name "final metrics spearman" \
    -model_name "best_model_s.ckpt" \
    -predictions_name "s_test_predictions.txt" \
    -labels_name $test_file \
    -genomics_name $genotype_file \
    -gene2id $gene2id_ont_file \
    -drug2id $drug2id_file \
    -cell2id $cell2id_file \
    -druginput_name $fingerprint_file \
    -drugs_names $drugs_names_file \
    -cuda 0
    > "$output_folder/metrics2_$i.log"

    echo "Iteration $i completed."
done