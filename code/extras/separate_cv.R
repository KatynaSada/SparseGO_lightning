#' @description
#' The aim of this code is to separate the samples in 5 groups to perform 5-fold cross-validation.
#' Pairs are randomly distributed in 5 groups
#' 
#' For each fold 3 files are created and stored in a folder: 
#'  - drugcell_train.txt
#'  - drugcell_validate.txt
#'  - drugcell_test.txt
#'  
set.seed(123)
mac <- "/Users/katyna/Library/CloudStorage/OneDrive----/"
windows <- "D:/Katyna/OneDrive - Tecnun/"
computer <- mac
file <- paste(computer,"SparseGO_code/data/CLs_mutations4transfer/allsamples/auc.csv",sep = "")
outputdir <- paste(computer,"SparseGO_code/data/CLs_mutations4transfer/",sep = "") # set output directory 
a <- read.csv(file, header = FALSE,sep = "\t",fileEncoding="UTF-8-BOM")

n_samples_validation <- 30000 # 500 for pdcs, 7000 for CLs, 30000 with PRISM

shuffled_data <- a[sample(nrow(a)), ]
# For each unique group:
# 1. Take the group as a hold out or test data set
# 2. Take the remaining groups as a training data set
# 3. Take some samples from the training data set as the validation set

spec = c(group1 = 0.2, group2 = 0.2, group3 = 0.2, group4=0.2, group5=0.2)

g = sample(cut(seq(nrow(shuffled_data)), 
               nrow(shuffled_data)*cumsum(c(0,spec)),
               labels = names(spec)
))

res = split(shuffled_data, g)

# create the 5 folds...

grupo1 <- res[["group1"]] # get data for group 1
grupo2 <- res[["group2"]] # get data for group 2
grupo3 <- res[["group3"]] # get data for group 3
grupo4 <- res[["group4"]] # get data for group 4
grupo5 <- res[["group5"]] # get data for group 5

k <- 5
num_grupos_train <- c(2:k) # set the initial train groups (excluding the current test group)

for (i in 1:k){ # loop through each fold (k is the number of folds)
  
  print(paste("Creating fold number:",i,"- test group:",i,"- train groups:", paste(num_grupos_train, collapse = ","))) # print the fold number and train/test groups
  
  test <- eval(parse(text = paste("grupo",as.character(i),sep=""))) # get the data for the current test group
  
  # join the other 4 groups for train
  train_data <- eval(parse(text = paste("grupo",as.character(num_grupos_train[1]),sep=""))) # get the data for the first train group
  print(num_grupos_train[1])
  for (j in 2:(k-1)){ # loop through the remaining train groups
    train_data <- rbind(train_data,eval(parse(text = paste("grupo",as.character(num_grupos_train[j]),sep="")))) # combine the train data
    print(num_grupos_train[j])
  }
  
  # create validation data
  validate <- train_data[1:n_samples_validation,] # select the first n rows for validation
  train <- train_data[(n_samples_validation+1):nrow(train_data),] # use the remaining rows for training
  
  # save the data files for the current fold
  sample_folder <- paste(outputdir,"samples",as.character(i),"/",sep="") # output directory for the current fold
  write.table(train[sample(nrow(train)), ], file = paste(sample_folder,"sparseGO_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save train data
  write.table(test[sample(nrow(test)), ], file =paste(sample_folder,"sparseGO_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save test data
  write.table(validate[sample(nrow(validate)), ], file = paste(sample_folder,"sparseGO_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save validation data
  
  num_grupos_train[i] <- i # update the train groups for the next fold (change current test for next test group)
}

# save all samples for final model
sample_folder <- paste(outputdir,"allsamples/",sep="")
write.table(a[sample(nrow(a)), ], file = paste(sample_folder,"sparseGO_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save train data
write.table(a[sample(nrow(a)), ], file =paste(sample_folder,"sparseGO_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save test data
write.table(a[sample(nrow(a)), ], file = paste(sample_folder,"sparseGO_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE) # save validation data

# train <- rbind(res[["group1"]],res[["group2"]],res[["group3"]],res[["group4"]])
# nrow(train)
# validate <- train[1:5000,]
# train <- train[5001:nrow(train),]
# nrow(train)
# test <- res[["group5"]]
# 
# # outputdir <- paste(computer,"my_code/data/cross_validation_expression/",sep="")
# outputdir <- computer
# 
# write.table(train, file = paste(outputdir,"samples5/sparseGO_train.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
# write.table(test, file =paste(outputdir,"samples5/sparseGO_test.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
# write.table(validate, file = paste(outputdir,"samples5/sparseGO_val.txt",sep=""), sep = "\t", row.names = F, col.names=F, quote = FALSE)
# 
# # 203.718 (152788 + 50929)
# dim(train)[1]+dim(validate)[1]
# 
# dim(train)[1]
# dim(validate)[1]
# dim(test)[1]

