# Practical Machine Learning Course Project


-----

#### Synopsis: 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to 
collect a large amount of data about personal activity relatively inexpensively. 
These type of devices are part of the quantified self movement - a group of 
enthusiasts who take measurements about themselves regularly to improve their
health, to find patterns in their behavior, or because they are tech geeks. One 
thing that people regularly do is quantify how much of a particular activity they
do, but they rarely quantify how well they do it. In this project, your goal will
be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 
participants. They were asked to perform barbell lifts correctly and incorrectly 
n 5 different ways. More information is available from the website here: 
http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting 
Exercise Dataset).    


The goal of this project is to predict the manner of performing unilateral dumbbell 
biceps curls based on data from accelerometers on the belt, forearm, arm, and 
dumbell of 6 participants. The 5 possible methods include -
* A: exactly according to the specification 
* B: throwing the elbows to the front
* C: lifting the dumbbell only halfway 
* D: lowering the dumbbell only halfway
* E: throwing the hips to the front

#### Load libraries and setup working directory

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
trainingRaw <- read.csv(file="pml-training.csv", header=TRUE, 
                        as.is = TRUE, stringsAsFactors = FALSE, sep=',', 
                        na.strings=c('NA','','#DIV/0!'))
testingRaw <- read.csv(file="pml-testing.csv", header=TRUE, 
                       as.is = TRUE, stringsAsFactors = FALSE, sep=',', 
                       na.strings=c('NA','','#DIV/0!'))

trainingRaw$classe <- as.factor(trainingRaw$classe)  
```

#### Data Cleaning
When looking at the data columns it was obvious that the first 7 columns were of 
no use (date & time variables) when building a prediction model and were excluded.


```r
trainingRaw = trainingRaw[,8:length(trainingRaw)]
testingRaw = testingRaw[,8:length(testingRaw)]
```

A large number of columns had NA values in both training and testing datasets 
and needed to be removed to prevent them from skewing the model.  This left 
53 variables for furhter analysis.  


```r
trainingRaw = trainingRaw[,! apply(trainingRaw,2,function(x) any(is.na(x)))]
testingRaw = testingRaw[,! apply(testingRaw,2,function(x) any(is.na(x)))]
```


#### Removing the non zero variables
Variables with values near zero can be removing as they do not have a significant 
impact on the prediction. 


```r
nzv <- nearZeroVar(trainingRaw,saveMetrics=TRUE)
trainingRaw <- trainingRaw[,nzv$nzv==FALSE]

nzv <- nearZeroVar(testingRaw,saveMetrics=TRUE)
testingRaw <- testingRaw[,nzv$nzv==FALSE]
```

#### Create cross validation set
The training set is then divided in two parts - the training and the cross 
validation set. 


```r
set.seed(12080569)
inTrain = createDataPartition(trainingRaw$classe, p = 3/4, list=FALSE)
training = trainingRaw[inTrain,]
crossValidation = trainingRaw[-inTrain,]
```

#### Train model
Random forest models have a high accuracy rate so model is trained with Random
Forest.  The model is built on a training set of 53 of the original 160 
variables.  Cross validation is used as the training control method. 


```r
model <- randomForest(classe ~ .,data=training,trControl=trainControl(method='cv'))
```


#### Accuracy on training set and cross validation set
Accuracy of the training and CV datasets are as follows:

Training set:

```r
trainingPred <- predict(model, training)
confusionMatrix(trainingPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```
Accuracy for the training set is 100%.

Cross validation set

```r
cvPred <- predict(model, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    1    0    0    0
##          B    0  947    4    0    0
##          C    0    1  851    6    0
##          D    0    0    0  798    1
##          E    0    0    0    0  900
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.998    0.995    0.993    0.999
## Specificity             1.000    0.999    0.998    1.000    1.000
## Pos Pred Value          0.999    0.996    0.992    0.999    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.184
## Detection Prevalence    0.285    0.194    0.175    0.163    0.184
## Balanced Accuracy       1.000    0.998    0.997    0.996    0.999
```
Accuracy for the cross validation accuracy is 99.6%, which should be sufficient 
for predicting the 20 test observations.

#### RESULTS
Predictions on the real testing set

```r
testingPred <- predict(model, testingRaw)
testingPred
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
