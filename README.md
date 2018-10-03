# Store Item Demand Forecasting Results
This repository contains my own scripts, predictions and results on the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) hosted in Kaggle.

Quoting the Overview of the competition on Kaggle:

_This competition is provided as a way to explore different time series techniques on a relatively simple and clean dataset._

_You are given 5 years of store-item sales data, and asked to predict 3 months of sales for 50 different items at 10 different stores._

_What's the best way to deal with seasonality? Should stores be modeled separately, or can you pool them together? Does deep learning work better than ARIMA? Can either beat xgboost?_

_This is a great competition to explore different models and improve your skills in forecasting._

The objective of the competition was to predict the sales of 10 different stores over 50 different items. The span of prediction were three months of the last year (2018). In order to do this Kaggle provided us with 4 years of data (2013-2017).

The metric used to evaluate the results was the SMAPE:

<a href="https://www.codecogs.com/eqnedit.php?latex=SMAPE&space;=&space;\frac{100}{n}&space;\sum_{t=1}^{n}&space;\frac{\mid&space;Y_t&space;-&space;Y_t&space;\mid}{(\mid&space;Y_t&space;\mid&space;-&space;\mid&space;Y_t&space;\mid)/2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SMAPE&space;=&space;\frac{100}{n}&space;\sum_{t=1}^{n}&space;\frac{\mid&space;Y_t&space;-&space;Y_t&space;\mid}{(\mid&space;Y_t&space;\mid&space;-&space;\mid&space;Y_t&space;\mid)/2}" title="SMAPE = \frac{100}{n} \sum_{t=1}^{n} \frac{\mid Y_t - Y_t \mid}{(\mid Y_t \mid - \mid Y_t \mid)/2}" /></a>

The sales of the different items had a strong seasonality and a similar behaviour between stores and items, some of this sales are shown in the next graph:

![alt text](https://github.com/jgonzalezab/Store-Item-Demand-Forecasting/blob/master/Images/sales_example.png "Sales Examples")

The approach I followed was the following:
* Extract date features (day, month, day of week, quarter...) and use them as independent variables in the model
* Iterate through different combinations of features in order to obtain the optimal ones
* Fit an XGB and lightGBM model per item (collapsing the stores)
* Search optimal hyperparameters through iterations
* Multiply the final predictions by 1.03 (magic number) and ensemble them

This machine-learning based approach obtained a 13.99955 SMAPE (137/462) on the public leaderboard and a **12.67808 SMAPE on the private leaderboard (93/462)**. The winner achieved a 12.58015 SMAPE on the private leaderboard. Finally, some graphs with the predictions are shown below:

![alt text](https://github.com/jgonzalezab/Store-Item-Demand-Forecasting/blob/master/Images/preds_example.png "Preds Examples")
