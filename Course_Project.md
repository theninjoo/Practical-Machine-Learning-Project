Practical Machine Learning Project
================
Jordan Miller-Ziegler
July 9, 2018

Summary
-------

This project involves using the caret package in R to train a model/models to predict the type of activity ("classe") being performed, given a host of body tracker data.

First, I set up my machine to run in parallel, since this is a large dataset with many variables, and model-fitting becomes time-consuming.

``` r
# Start parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```

Next, the data are loaded from the course website.

``` r
# Load data
if(!file.exists("train.csv")){
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "train.csv")}
if(!file.exists("test.csv")){
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "test.csv")}
training_raw <- read.csv("train.csv")
testing_raw <- read.csv("test.csv")
```

Preprocessing
-------------

The data contain several troublesome variables, the ID values (which contain no useful predictive data) and the classe variable (which ought to be removed for PCA, and isn't in the testing data set). These are removed.

``` r
# Remove ID variables and outcome variable
training_almost_raw <- training_raw[,-c(1:2,length(training_raw))]
testing_almost_raw <- testing_raw[,-c(1:2,length(testing_raw))]
```

Next, variables are removed which show little-or-no variance, as these just cause problems and add basically no predictive value.

``` r
# Remove near-zero variance variables
nzv <- nearZeroVar(training_almost_raw)
training_nonzero_var <- training_almost_raw[,-nzv]
testing_nonzero_var <- testing_almost_raw[,-nzv]
```

Finally, the data are preprocessed to both impute missing values with knnImpute and perform principal components analysis. These steps ensure the data has no NAs, and reduce the data to 35 components - much easier on the ol' CPU.

``` r
# Impute missing values and perform PCA
preprocess_model <- preProcess(training_nonzero_var, method = c("knnImpute", "pca"))
training_processed <- predict(preprocess_model, training_nonzero_var)
testing_processed <- predict(preprocess_model, testing_nonzero_var)
```

Model Training
--------------

In this section, I use the mighty power of my laptop's FOUR (4!) cores to (eventually...) build a couple of models for predicting the test data. Though the plan was originally to ensemble several models, the second model performed so well that I just called it a day after that.

``` r
# Add the classe variable to the training data again.
training_processed <- cbind(training_processed, training_raw$classe)
names(training_processed)[length(training_processed)] <- "classe"
```

Given the larger size of the data set, I set the resampling method to be cross-validation, rather than bootstrap.

``` r
controls <- trainControl(method = "cv")
```

I thought to train several models, and started with an rpart model. It looked... Not good. Next, I ran a random forest model, and it looked GREAT. Given how well this model performed, I stopped here and submitted the prediction test. I got 19/20, which is plenty good enough for me!

``` r
rpart_model <- train(classe ~ ., training_processed, method = "rpart", trControl = controls)
rpart_model
```

    ## CART 
    ## 
    ## 19622 samples
    ##    37 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 17660, 17660, 17659, 17659, 17660, 17661, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa    
    ##   0.01630822  0.6293419  0.5351893
    ##   0.02029625  0.6176193  0.5206920
    ##   0.03078859  0.5408633  0.4038647
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was cp = 0.01630822.

``` r
rf_model <- train(classe ~ ., training_processed, method = "rf", trControl = controls)
rf_model
```

    ## Random Forest 
    ## 
    ## 19622 samples
    ##    37 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 17659, 17659, 17661, 17659, 17659, 17659, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9594342  0.9486819
    ##   28    0.9814497  0.9765364
    ##   55    0.9729892  0.9658400
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 28.

Here are the predictions. (Numer 11 ends up being wrong.)

``` r
predict(rf_model, testing_processed)
```

    ##  [1] B A B A A E D B A A A C B A E E A B B B
    ## Levels: A B C D E

``` r
# Stop parallel processing
stopCluster(cluster)
registerDoSEQ()
```

Conclusion
----------

Using the caret package, I was able to pre-process, model, and predict the classe of activity being performed from body tracking data for 20 new test cases, successfully identifying 19/20 (95%).
