# Simple ML for each item (Ensemble XGBoost and LightGBM)
# Correction of 1.03

# Libraries
library(lightgbm)
library(caret)

# Load data
trainRaw <- read.csv("../Data/train.csv")
testRaw <- read.csv("../Data/test.csv")

# Adjust number of threads (XGBoost issue)
library(OpenMPController)
omp_set_num_threads(4)

# Preprocess data
auxDF <- rbind.data.frame(trainRaw[, -ncol(trainRaw)],
                          testRaw[, -1])

auxDF$date <- as.character(auxDF$date)
auxDF$day <- as.numeric(substr(auxDF$date, 9, 10))
auxDF$month <- as.numeric(substr(auxDF$date, 6, 7))
auxDF$year <- as.numeric(substr(auxDF$date, 1, 4))
auxDF$dayWeek <- as.numeric(as.factor(weekdays(as.Date(auxDF$date))))
auxDF$date <- NULL

auxDF$quarter <- auxDF$month / 3
auxDF$quarter <- ifelse(auxDF$quarter <= 1, 1, auxDF$quarter)
auxDF$quarter <- ifelse(auxDF$quarter > 1 & auxDF$quarter <= 2, 2, auxDF$quarter)
auxDF$quarter <- ifelse(auxDF$quarter > 2 & auxDF$quarter <= 3, 3, auxDF$quarter)
auxDF$quarter <- ifelse(auxDF$quarter > 3 & auxDF$quarter <= 4, 4, auxDF$quarter)

train <- auxDF[1:nrow(trainRaw), ]
train$sales <- trainRaw$sales
train <- train[-which(train$day == 29 & train$month == 2), ]
train$day <- NULL

test <- auxDF[(nrow(trainRaw) + 1):nrow(auxDF), ]
test$id <- testRaw$id
test <- test[, c(8, 1:7)]
test$day <- NULL

# SMAPE functions
SMAPEcaret <- function(data, lev = NULL, model = NULL) {
  
  smapeCaret <- Metrics::smape(data$obs, data$pred)
  c(SMAPEcaret = -smapeCaret)
  
}

SMAPElightgbm <- function(preds, dtrain){
  
  actual <- getinfo(dtrain, "label")
  score <- Metrics::smape(preds, actual)
  
  return(list(name = "SMAPE", value = score, higher_better = FALSE))
  
}

# Training settings caret
modelsXGB <- list()

fitControl <- trainControl(method = 'none',
                           summaryFunction = SMAPEcaret)

parametersXGB <- expand.grid(nrounds = 100, 
                             max_depth = 6,
                             eta = 0.2, 
                             gamma = 0.6, 
                             colsample_bytree = 0.5,
                             min_child_weight = 2, 
                             subsample = 1)

# Training settings lightgbm
modelsLGB <- list()

lgbGrid  <-  list(objective = "regression", 
                  min_sum_hessian_in_leaf = 1, 
                  feature_fraction = 0.7, 
                  bagging_fraction = 0.7, 
                  bagging_freq = 5, 
                  max_bin = 50, 
                  lambda_l1 = 1, 
                  lambda_l2 = 1.3, 
                  min_data_in_bin = 100, 
                  min_gain_to_split = 10, 
                  min_data_in_leaf = 30)

# Model for each item
for(item in 1:50) {
  
  cat(item, ' | 50', sep = '')
  
  auxTrain <- train[which(train$item == item), ]
  auxTrain$item <- NULL
  
  # XGBoost
  modelsXGB[[item]] <- train(x = auxTrain[, -ncol(auxTrain)],
                             y = auxTrain[, ncol(auxTrain)],
                             method = 'xgbTree',
                             trControl = fitControl,
                             tuneGrid = parametersXGB,
                             metric = 'SMAPEcaret')
  
  # LightGBM
  label <- auxTrain$sales
  auxTrainMatrix <- Matrix::Matrix(as.matrix(auxTrain[, -ncol(auxTrain)]), sparse = TRUE)
  
  auxTrainLGB <- lgb.Dataset(data = auxTrainMatrix, label = label)
  
  modelsLGB[[item]] <- lgb.train(params = lgbGrid, data = auxTrainLGB, learning_rate = 0.1,
                                 num_leaves = 10, num_threads = 3, nrounds = 500,
                                 eval_freq = 20, eval = SMAPElightgbm, verbose = -1)
  
}

# Generate predictions
predsXGB <- c()
predsLGB <- c()

for(item in 1:50) {
  
  cat(item, ' | 50', sep = '')
  
  indices <- which(test$item == item)
  auxTest <- test[indices, ]
  auxTest$item <- NULL
  
  # XGBoost
  auxXGB <- predict(modelsXGB[[item]], auxTest[, -1])
  
  # LightGBM
  auxTest <- Matrix::Matrix(as.matrix(auxTest[, -1]), sparse = TRUE)
  
  auxLGB <- predict(modelsLGB[[item]], auxTest)
  
  predsXGB <- c(predsXGB, auxXGB)
  predsLGB <- c(predsLGB, auxLGB)
  
}

predsXGB <- predsXGB * 1.03
predsLGB <- predsLGB * 1.03
test$sales <- rowMeans(data.frame(predsXGB, predsLGB))

# Submission file
submission <- read.csv('../Data/sample_submission.csv')
submission$sales <- test$sales
write.csv(submission, 'lgb_xgb_ensemble.csv', row.names = FALSE)