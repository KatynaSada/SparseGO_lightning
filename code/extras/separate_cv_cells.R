#' @description
#' The aim of this code is to separate the samples in 5 groups to perform 5-fold cross-validation.
#' All samples of a same CELL must be on the same group.
#' 
#' For each fold 3 files are created and stored in a folder: 
#'  - drugcell_train.txt
#'  - drugcell_validate.txt
#'  - drugcell_test.txt
#'  

set.seed(1234) 
mac <- "/Users/katyna/Library/CloudStorage/OneDrive----/SparseGO_code/data/"
windows <- "D:/Katyna/OneDrive - Tecnun/SparseGO_code/data/"
computer <- mac
# REMEMBER THERE'S A SEPARATE CELLS FIXED!!! (for PDCs)
outputdir <- paste(computer,"PDCs_multiomics_LELO/",sep="")
inputdir <- paste(computer,"PDCs_multiomics_LELO/allsamples/auc.csv",sep="")
data <- read.csv(inputdir, header = FALSE,sep = "\t",fileEncoding="UTF-8-BOM")

k_train <- 5 # 4-5 en PDCs,25 en CLs (tener cuidado, que haya suficientes en validation)  5 EN EL ULTIMO CON 5CV Y PDCS

cell_count <- as.data.frame(table(data$V1))
# Shuffle the list of cells randomly
cell_count <- cell_count[sample(1:nrow(cell_count), nrow(cell_count), replace = F),]

# Split the list of cells into k groups
n <- sum(cell_count$Freq)
k <- 5 # number of folds-

# For each unique group:
# 1. Take the group as a hold out or test data set
# 2. Take the remaining groups as a training data set
# 3. Take some samples from the training data set as the validation set (drugs must be different as that of the training set)

grupos <- cumsum(cell_count$Freq-1) %/% round(n/k) # there must be approximately the same amount of samples in each group

grupo1 <- data[data$V1 %in% as.character(cell_count[grupos==0,]$Var1),]
grupo2 <- data[data$V1 %in% as.character(cell_count[grupos==1,]$Var1),]
grupo3 <- data[data$V1 %in% as.character(cell_count[grupos==2,]$Var1),]
grupo4 <- data[data$V1 %in% as.character(cell_count[grupos==3,]$Var1),]
grupo5 <- data[data$V1 %in% as.character(cell_count[grupos==4,]$Var1),]

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
  train_count <- as.data.frame(table(train_data$V1))
  train_count <- train_count[sample(1:nrow(train_count), nrow(train_count), replace = F),]
  
  n_train <- sum(train_count$Freq)
  grupos_train <- cumsum(train_count$Freq-1) %/% round(n_train/k_train)
  
  validate <- train_data[train_data$V1 %in% as.character(train_count[grupos_train==0,]$Var1),]
  train <- train_data[train_data$V1 %in% as.character(train_count[grupos_train!=0,]$Var1),]
  
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

# train_data <- rbind(grupo5,grupo2,grupo3,grupo4)
# train_count <- as.data.frame(table(train_data$V1))
# train_count <- train_count[sample(1:nrow(train_count), nrow(train_count), replace = F),]
# 
# n_train <- sum(train_count$Freq)
# k_train <- 80
# grupos_train <- cumsum(train_count$Freq-1) %/% round(n_train/k_train)
# 
# validate <- train_data[train_data$V1 %in% as.character(train_count[grupos_train==0,]$Var1),]
# train <- train_data[train_data$V1 %in% as.character(train_count[grupos_train!=0,]$Var1),]
# test <-grupo1
# 
# outputdir <- paste(computer,"my_code/data/cross_validation_expression_cells/",sep="")
# 
# write.table(train, file = paste(outputdir,"samples5/drugcell_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
# write.table(test, file =paste(outputdir,"samples5/drugcell_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
# write.table(validate, file = paste(outputdir,"samples5/drugcell_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)

