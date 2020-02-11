
# Load required libraries
# Required to read .mat file
library(R.matlab)
# Required for dirichlet implementation
library(sirt)

# Read train data and train labels
train_data <- data.frame(readMat("Extracted_Features/file_feat_x_train.mat"))
train_labels <- data.frame(readMat("Extracted_Features/file_feat_lab_y_train.mat"))

# Adding noise to avoid zeros while normalization
eta = 1
##============================================================================================
# Seperating data for each class
# Temporary dataframe
req_data <- data.frame(train_data, t(train_labels))

# Class 0 data
req_data_C0 <- subset(req_data, req_data$t.train_labels. == 0)
req_data_C0 <- req_data_C0[,-ncol(req_data_C0)]

# Class 1 data
req_data_C1 <- subset(req_data, req_data$t.train_labels. == 1)
req_data_C1 <- req_data_C1[,-ncol(req_data_C1)]

# Class 2 data
req_data_C2 <- subset(req_data, req_data$t.train_labels. == 2)
req_data_C2 <- req_data_C2[,-ncol(req_data_C2)]

##============================================================================================
# Find sum of features/dimensions
sum_C0 <- apply(req_data_C0, 1, sum)
sum_C1 <- apply(req_data_C1, 1, sum)
sum_C2 <- apply(req_data_C2, 1, sum)

# Laplace smoothing
norm_data_C0 <- (req_data_C0 + eta)/(sum_C0 + (ncol(req_data_C0)*eta))
norm_data_C1 <- (req_data_C1 + eta)/(sum_C1 + (ncol(req_data_C1)*eta))
norm_data_C2 <- (req_data_C2 + eta)/(sum_C2 + (ncol(req_data_C2)*eta))

##============================================================================================
# Finding alphas for dirichlet distribution
# Class 0
C0_req_alpha <- dirichlet.mle(as.matrix(norm_data_C0))
C0_alpha <- C0_req_alpha$alpha
# saveRDS(C0_req_alpha, "C0_alpha.rds")

# Class 1
C1_req_alpha <- dirichlet.mle(as.matrix(norm_data_C1))
C1_alpha <- C1_req_alpha$alpha
# saveRDS(C1_req_alpha, "C1_alpha.rds")

# Class 2
req_alpha_C2 <- dirichlet.mle(as.matrix(norm_data_C2))
C2_alpha <- req_alpha_C2$alpha
# saveRDS(req_alpha_C2, "C2_alpha.rds")
##============================================================================================
##============================================================================================

##============================================================================================
# Testing
##============================================================================================
# Reading test dat and test labels
test_data <- data.frame(readMat("Extracted_Features/file_feat_x_test.mat"))
test_labels <- data.frame(readMat("Extracted_Features/file_feat_lab_y_test.mat"))

# Laplace smoothing
sum_test <- apply(test_data, 1, sum)
req_test_data <- (test_data + eta)/(sum_test + (ncol(req_data_C0)*eta))

##============================================================================================
# Dataframe to store results
req_test_res <- data.frame()

# Finding maximum likelihood
# For all samples
for (i in 1:nrow(req_test_data)) {
  
  # Class 0
  req_likelihood_C0 <- 0
  sum_gamma_alpha_C0 <- 0
  log_gamma_alpha_C0 <- 0
  # For all features
  for (j in 1:length(C0_alpha)) {
    req_likelihood_C0 <- req_likelihood_C0 + (C0_alpha[j]-1) * log(req_test_data[i,j])
    sum_gamma_alpha_C0 <- sum_gamma_alpha_C0 + lgamma(C0_alpha[j])
    log_gamma_alpha_C0 <- log_gamma_alpha_C0 + C0_alpha[j]
  }
  req_test_res[i,1] <- lgamma(log_gamma_alpha_C0) - sum_gamma_alpha_C0 + req_likelihood_C0
  
  ##============================================================================================
  # Class 1
  req_likelihood_C1 <- 0
  sum_gamma_alpha_C1 <- 0
  log_gamma_alpha_C1 <- 0
  for (j in 1:length(C1_alpha)) {
    req_likelihood_C1 <- req_likelihood_C1 + (C1_alpha[j]-1) * log(req_test_data[i,j])
    sum_gamma_alpha_C1 <- sum_gamma_alpha_C1 + lgamma(C1_alpha[j])
    log_gamma_alpha_C1 <- log_gamma_alpha_C1 + C1_alpha[j]
  }
  req_test_res[i,2] <- lgamma(log_gamma_alpha_C1) - sum_gamma_alpha_C1 + req_likelihood_C1
  
  ##============================================================================================
  # Class 2
  req_likelihood_C2 <- 0
  sum_gamma_alpha_C2 <- 0
  log_gamma_alpha_C2 <- 0
  for (j in 1:length(C2_alpha)) {
    req_likelihood_C2 <- req_likelihood_C2 + (C2_alpha[j]-1) * log(req_test_data[i,j])
    sum_gamma_alpha_C2 <- sum_gamma_alpha_C2 + lgamma(C2_alpha[j])
    log_gamma_alpha_C2 <- log_gamma_alpha_C2 + C2_alpha[j]
  }
  req_test_res[i,3] <- lgamma(log_gamma_alpha_C2) - sum_gamma_alpha_C2 + req_likelihood_C2
  
}
colnames(req_test_res)[1:3] <- c("C0", "C1", "C2")
req_test_res$Pred_Class <- 1

##============================================================================================
# Finding class labels

for (i in 1:nrow(req_test_res)) {
  req_test_res[i, 4] <- (which(req_test_res[i,] == max(req_test_res[i,]))) - 1
}

req_test_res$Actual_Class <- t(test_labels)

##============================================================================================
# Computing accuracy
right <-  0
wrong <- 0

for (i in 1:nrow(req_test_res)) {
  if(req_test_res[i,4] == req_test_res[i,5]){
    right <- right +1
  }else{
    wrong <- wrong + 1
  }
}

accuracy <- (right/(right+wrong))*100

##============================================================================================


