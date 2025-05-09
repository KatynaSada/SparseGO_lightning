#' @description
#' The aim of this code is to separate the samples in 5 groups to perform 5-fold cross-validation.
#' All samples of a same DRUG must be on the same group.
#' 
#' For each fold 3 files are created and stored in a folder: 
#'  - drugcell_train.txt
#'  - drugcell_validate.txt
#'  - drugcell_test.txt
#'  
set.seed(12345)

mac <- "/Users/katyna/Library/CloudStorage/OneDrive----/SparseGO_lightning/data/"
windows <- "D:/Katyna/OneDrive - Tecnun/SparseGO_lightning/data/"
computer <- mac

outputdir <- paste(computer,"PDCs_multiomics_LECO/",sep="")
inputdir <- paste(computer,"PDCs_multiomics_LECO/allsamples/auc.csv",sep="")
data <- read.csv(inputdir, header = FALSE,sep = "\t",fileEncoding="UTF-8-BOM")

drug_count <- as.data.frame(table(data$V2))
# Shuffle the list of drugs randomly
drug_count_shuffle <- drug_count[sample(1:nrow(drug_count), nrow(drug_count), replace = F),]

# Split the list of drugs into k groups
n <- sum(drug_count$Freq)
k <- 5 # number of folds 5 in CLs

# For each unique group:
# 1. Take the group as a hold out or test data set
# 2. Take the remaining groups as a training data set
# 3. Take some samples from the training data set as the validation set (drugs must be different as that of the training set)


grupos<- cumsum(drug_count_shuffle$Freq-1) %/% round(n/k) # there must be approximately the same amount of samples in each group

grupo1 <- data[data$V2 %in% as.character(drug_count_shuffle[grupos==0,]$Var1),]
grupo2 <- data[data$V2 %in% as.character(drug_count_shuffle[grupos==1,]$Var1),]
grupo3 <- data[data$V2 %in% as.character(drug_count_shuffle[grupos==2,]$Var1),]
grupo4 <- data[data$V2 %in% as.character(drug_count_shuffle[grupos==3,]$Var1),]
grupo5 <- data[data$V2 %in% as.character(drug_count_shuffle[grupos==4,]$Var1),]

# create the 5 folds... 

num_grupos_train <- c(2:k)


for (i in 1:k){
  print(paste("Creating fold number:",i,"- test group:",i,"- train groups:", paste(num_grupos_train, collapse = ",")))
  
  test <- eval(parse(text = paste("grupo",as.character(i),sep="")))
  
  # join the other 4 groups for train
  train_data <- eval(parse(text = paste("grupo",as.character(num_grupos_train[1]),sep="")))
  print(num_grupos_train[1])
  for (j in 2:(k-1)){
    train_data <- rbind(train_data,eval(parse(text = paste("grupo",as.character(num_grupos_train[j]),sep=""))))
    print(num_grupos_train[j])
  }
  # create validation 
  train_count <- as.data.frame(table(train_data$V2))
  train_count <- train_count[sample(1:nrow(train_count), nrow(train_count), replace = F),]
  
  n_train <- sum(train_count$Freq)
  k_train <- 4 # 6 en PDCs,25 CLs
  grupos_train <- cumsum(train_count$Freq-1) %/% round(n_train/k_train)
  
  validate <- train_data[train_data$V2 %in% as.character(train_count[grupos_train==0,]$Var1),]
  train <- train_data[train_data$V2 %in% as.character(train_count[grupos_train!=0,]$Var1),]
  
  # save txts
  sample_folder <- paste(outputdir,"samples",as.character(i),"/",sep="")
  write.table(train[sample(nrow(train)), ], file = paste(sample_folder,"sparseGO_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
  write.table(test[sample(nrow(test)), ], file =paste(sample_folder,"sparseGO_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
  write.table(validate[sample(nrow(validate)), ], file = paste(sample_folder,"sparseGO_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
  
  num_grupos_train[i] <- i # for next fold we need to change the train groups (change current test for next test group)
}

# save all samples for final model
sample_folder <- paste(outputdir,"allsamples/",sep="")
write.table(data[sample(nrow(data)), ], file = paste(sample_folder,"sparseGO_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save train data
write.table(data[sample(nrow(data)), ], file =paste(sample_folder,"sparseGO_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save test data
write.table(data[sample(nrow(data)), ], file = paste(sample_folder,"sparseGO_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save validation data


# train_data2 <- rbind(grupo1,grupo3,grupo4,grupo5)
# train_count <- as.data.frame(table(train_data$V2))
# train_count <- train_count[sample(1:nrow(train_count), nrow(train_count), replace = F),]
# 
# n_train <- sum(train_count$Freq)
# k_train <- 80
# grupos_train <- cumsum(train_count$Freq-1) %/% round(n_train/k_train)
# 
# validate <- train_data[train_data$V2 %in% as.character(train_count[grupos_train==0,]$Var1),]
# train <- train_data[train_data$V2 %in% as.character(train_count[grupos_train!=0,]$Var1),]
# test <-grupo5


