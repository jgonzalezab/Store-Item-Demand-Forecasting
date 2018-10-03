# Simple ML for each item individually (XGBoost)

# Libraries
library(ggplot2)
library(caret)

# Adjust number of threads (XGBoost issue)
library(OpenMPController)
omp_set_num_threads(4)

# Load data
trainRaw <- read.csv("../Data/train.csv")
testRaw <- read.csv("../Data/test.csv")

# Preprocess data
auxDF <- rbind.data.frame(trainRaw[, -ncol(trainRaw)],
                          testRaw[, -1])

auxDF$date <- as.character(auxDF$date)
auxDF$day <- as.numeric(substr(auxDF$date, 9, 10))
auxDF$month <- as.numeric(substr(auxDF$date, 6, 7))
auxDF$year <- as.numeric(substr(auxDF$date, 1, 4))
auxDF$dayWeek <- as.numeric(as.factor(weekdays(as.Date(auxDF$date))))
auxDF$date <- NULL

train <- auxDF[1:nrow(trainRaw), ]
train$sales <- trainRaw$sales
train <- train[-which(train$day == 29 & train$month == 2), ]
train$day <- NULL

test <- auxDF[(nrow(trainRaw) + 1):nrow(auxDF), ]
test$id <- testRaw$id
test <- test[, c(7, 1:6)]
test$day <- NULL

# Train and valid
trainSet <- train[-which(train$year == 2017), ]
validSet <- train[which(train$year == 2017 &
                          (train$month == 1 | train$month == 2 | train$month == 3)), ]

dataToUse <- rbind.data.frame(trainSet, validSet)

# Iterate through each item
SMAPE <- function(data, lev = NULL, model = NULL) {
  
  smapeCaret <- Metrics::smape(data$obs, data$pred)
  c(SMAPEcaret = -smapeCaret)
  
}

modelsXGB <- list()

for(item in 1:50) {
  
  cat('\014')
  cat(item, ' | 50', sep = '')
  
  # Selecting the correct subsets
  indx <- which(dataToUse$item == item & dataToUse$year == 2017)
  validItem <- dataToUse[indx, ]
  validItem$item <- NULL
  
  auxTrainSet <- trainSet[which(trainSet$item == item), ]
  auxTrainSet$item <- NULL
  
  auxFull <- rbind.data.frame(auxTrainSet, validItem)
  indx <- (nrow(auxTrainSet) + 1):nrow(auxFull)
  
  # Searching the optimal hyperparameters
  fitControl <- trainControl(method = 'cv',
                             number = 1,
                             search = 'random',
                             index = list(Fold2017 = indx),
                             summaryFunction = SMAPE)
  
  
  modelsXGB[[item]] <- train(x = auxFull[, -ncol(auxFull)],
                             y = auxFull[, ncol(auxFull)],
                             method = 'xgbTree',
                             trControl = fitControl,
                             tuneLength = 100,
                             metric = 'SMAPEcaret')
  
  # Fitting the final model
  auxTrain <- train[which(train$item == item), ]
  auxTrain$item <- NULL
  
  defFitControl <- trainControl(method = 'none',
                                summaryFunction = SMAPE)
  
  modelsXGB[[item]] <- train(x = auxTrain[, -ncol(auxTrain)],
                             y = auxTrain[, ncol(auxTrain)],
                             method = 'xgbTree',
                             trControl = defFitControl,
                             tuneGrid = modelsXGB[[item]]$bestTune,
                             metric = 'SMAPEcaret')
  
}

# Generate predictions
test$sales <- rep(NA, nrow(test))

for(item in 1:50) {
  
  cat('\014')
  cat(item, ' | 50', sep = '')
  
  indices <- which(test$item == item)
  
  auxTest <- test[indices, ]
  auxTest$item <- NULL
  
  aux <- predict(modelsXGB[[item]], auxTest[, -1])
  aux <- aux * 1.05
  
  test[indices, 'sales'] <- aux
  
}

# Submission file
submission <- read.csv('../Data/sample_submission.csv')
submission$sales <- test$sales
write.csv(submission, 'xgb_per_item_CV.csv', row.names = FALSE)