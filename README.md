# Rating-Prediction-of-Yelp-Reviews
STAT 628: Module 2
## Abstract
In this project, we aim to predict rating of businesses listed in the Yelp dataset based on review text and other variables, and check how the sentiment varied over geographical location, time and other factors. To find the combination of best machine learning technique and feature extraction method to solve the problem, linear regression and different classification techniques such as Naïve Bayes, Logistic regression, Long-Short Term Memory and Support Vector Machines were used. 
## Data Set
In the training set, there are 1,546,379 observations, which containing specific information about a review, like its date, text, star, location and categories. In the testing and validation data, there are 1016664 elements without ratings. 
In the data folder,  “uni_10000.csv”, “bi_20000.csv” and “tri_50000.csv” are the words with highest encounter. “new_stop.txt” is the stop words we selected. 
## Code
In the code folder, we have python files. “model selection” includes the code of creating matrix for models and all kind of models we tried(except LSTM). “LSTM” contains our best model LSTM. “variable selection” is the process of variable selection. 
## Plot
In the plot folder,  we have pre-analysis which includes “word cloud”, ratings over review length,  top 5 categories and so on. Besides we have the analysis of our result, like residual plot, QQ plot and so on.
