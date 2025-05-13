# #!/bin/bash
# nvidia-smi
# Activate your Python environment
source activate /scratch/ksada/envs/SparseGOnew # CHANGE THIS - your environment
# source activate  /Users/katyna/envs/SparseGOnew # CHANGE THIS - your environment

# Login to W&B
wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account

# Set OMP_NUM_THREADS to the value of SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the Python script with the required arguments
folder_name="PDCs_logexpression_audrc_star_cell-blind"
split_type="cell-blind" # CHANGE this - cell-blind drug-blind cancer-blind
entity="miramon_team" 
project="Hyper_PDCs_expression_variants" # CHANGE THIS - name you want to give to your W&B project
tags=log_expression,audrc_star,cell-blind,morgan_fingerprint #  selfieslatent184 morgan_fingerprint drug-blind cancer-blind

input_folder="../data/"$folder_name"/"
output_folder="../results/"$folder_name"/" # CHANGE THIS - folder to store results
mkdir $output_folder
mkdir $output_folder"figures/"

epochs=250

num_workers=4
precision="32-true" # 32-true 16-mixed
strategy="auto"

min_delta=0.001
patience=30 
early_stopping_metric="validation_corr_spearman"

# Define file paths for datasets, ontology, and other resources
input_type="expression" # CHANGE multiomics mutations expression 
gene2id_ont_file="gene2ind.txt" # CHANGE gene2ind gene2ind_ont
genotype_file="cell2expression.txt" # CHANGE cell2expression cell2multiomics cell2mutations
fingerprint_file="drug2fingerprint.txt" # CHANGE drug2selfieslatent184 drug2fingerprint drug2selfieslatent184_new drug2selfieslatent184_largermod

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
for sample_folder in "samples1" "samples2" "samples3" "samples4" "samples5" "allsamples"
do
  mkdir $output_folder$sample_folder
done

#     drug_neurons=$(shuf -e "100,50,6" "80,40" "100," "60,40" "20," "30," "200,100,50" "," -n 1)
# en pdcs con 4 GPUs no se puede un batch size de mas de 550 aprox
# Loop to execute the process 200 times
for i in {1..1}; do
    echo "Running iteration $i..."
    job_type=$(date +%s | md5sum | cut -c 1-8)  # Generates an 8-character random ID based on the current timestamp
    echo "Generated Job ID: $job_type"

    # Randomly select hyperparameters
    batch_size=$(shuf -e 50 100 200 500 1000 1500 -n 1)
    learning_rate=$(shuf -e 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.2 0.3 -n 1)
    optimizer_type="sgd"  # Static value in this config
    decay_rate=$(shuf -e 0 0.003 0.01 0.03 0.1 -n 1)
    loss_type="MSELoss"  # Static value
    momentum=$(shuf -e 0.85 0.88 0.9 0.93 0.95 0.96 -n 1)
    num_neurons_per_GO=$(shuf -e 4 5 6 7 -n 1)
    num_neurons_final_GO=$(shuf -e 6 12 24 -n 1)
    drug_neurons=$(shuf -e "100,50,6" "80,40" "100,30,15" "60,40" "100,50,100" "30," "200,100,50" -n 1)
    num_neurons_final=$(shuf -e 12 18 16 20 -n 1)
    p_drop_genes=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_terms=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_drugs=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    p_drop_final=$(shuf -e 0 0.05 0.1 0.15 -n 1)
    
    # batch_size=550
    # learning_rate=0.1
    # optimizer_type="sgd"  # Static value in this config
    # decay_rate=0.02
    # loss_type="MSELoss"  # Static value
    # momentum=0.8
    # num_neurons_per_GO=4
    # num_neurons_final_GO=6
    # drug_neurons="60,40"
    # num_neurons_final=18
    # p_drop_genes=0.05
    # p_drop_terms=0
    # p_drop_drugs=0.1
    # p_drop_final=0.15

    # Create a hyperparameter file
    hyperparams_file=$output_folder"hyperparams.txt"

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
    
    for sample_folder in "samples1" "samples2" "samples3" "samples4" "samples5" #"samples1" "samples2" "samples3" "samples4" "samples5" 
    do
        # Run the training script with torchrun and the specified hyperparameters $SLURM_NTASKS
        torchrun \
            --nproc_per_node=$SLURM_NTASKS \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --node_rank=$SLURM_NODEID \
            --master_addr=$(hostname) \
            --master_port=$(shuf -i 20000-30000 -n 1) \
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

    python -u ../code/final_metrics.py \
    -project $project  \
    -entity $entity  \
    -tags $tags \
    -job_type $job_type \
    -sweep_name "final metrics drugs" \
    -input_type $input_type \
    -split_type $split_type \
    -input_folder $input_folder \
    -output_folder $output_folder \
    -samples_folders $samples_folders \
    -predictions_name "d_test_predictions.txt" \
    -labels_name $test_file \
    -drugs_names $drugs_names_file \
    -log_artifact True \

    python -u ../code/final_metrics.py \
    -project $project  \
    -entity $entity  \
    -tags $tags \
    -job_type $job_type \
    -sweep_name "final metrics spearman" \
    -input_type $input_type \
    -split_type $split_type \
    -input_folder $input_folder \
    -output_folder $output_folder \
    -samples_folders $samples_folders \
    -predictions_name "s_test_predictions.txt" \
    -labels_name $test_file \
    -drugs_names $drugs_names_file \
    -log_artifact True \

    # echo "Iteration $i completed."
done