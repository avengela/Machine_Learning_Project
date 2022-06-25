# Machine_Learning_Project
Final project Machine Learning - CDV 2022
## Created by: Angelika Bulkowska and Julia Grzegorowska

# Task - description
The data consists of 2 splits train_data.csv and test_data.csv. There are numeric observations labelled either -1 or +1. Labels for the training data are located in in train_labels.csv. 
The aim of the project is to predict labels for the testing data.

Desired output:
Save predictions in test_labels.csv,
Prepare a report (saved as report.md) with the explanations on how you came up with the solution. What obstacles you faced and how did you solve them. Add also information about data quality and so on.

# Steps

1) [EDA](#EDA)
2) [Metric](#Metric)
3) [Baseline](#Baseline)
4) [Dataset split](#Dataset_split)
5) [Classification](#Classification)
6) [Final code](#Final_code)
7) [Results](#Results)


<a name="EDA"></a>
## EDA

A basic check of data quality was performed. 
“Train_data” is a dataset containing 10 000 columns and 3750 rows. 
No duplicated and null data was identified within the dataset. 

However train_labels, that are to be predicted for test dataset, are unbalanced and the
values are binary.


Later, after using some **preprocessing methods such as standarization and normalization.**
<br />
<br />

Because we have a large dataset for training to procees at the beggining, we used **Univariate Feature Selection (SelectKBest method)** which selects the features according to the k highest score. We could apply the method for both classification and regression data. 

### Univariate Feature Selection ####
    
| Shape before | (3750, 10000) |
|--------------|---------------|
| Shape after  | (3750, 3177)  |

| min score | max score | Mean score |
|-----------|-----------|------------|
| 1.76      | 17.32     | 1.00038    |


<br />
<br />

To get rid of noisy data
and PCA reshaping data to examine std and get rid of extreme data. Because ourdataset have more columns than rows PCA helped us manage it in easy way.

### PCA - Principal component analysis ###


| Shape before | (3750, 3177) |
|--------------|--------------|
| Shape after  | (3750, 100)  |

<br />
<br />

To fits a model and removes the weakest feature (or features) until the specified number of features is reached we used RFE.

### RFE - Recursive Feature Elimination ###

| Shape before | (3750, 100) |
|--------------|--------------|
| Shape after  | (3750, 5)  |


Later trying to understand and set data types, data mixtures, shape, outliers, missing values, noisy data, skewness and kurtosis by creating heatmaps, matrix plots and others.

 While calling .value_count() on data, was that our dataset was unbalanced so the solution was to call:
- Random Undersampling
- [Oversampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

Here we used:

    imblearn.over_sampling.RandomOverSampler


and generate some plot

![scatter_plot](https://user-images.githubusercontent.com/100999656/175771732-bdb514a2-278c-4631-a792-8c31e0c4f546.jpg)

![correlation_heatmap](https://user-images.githubusercontent.com/100999656/175771715-ffbf2586-39d7-4b86-bde2-16d3e1661479.jpg)

![correlation_matrix](https://user-images.githubusercontent.com/100999656/175771723-98aaf3c5-4042-45b1-9ab8-b787650b583e.jpg)

![box_plot](https://user-images.githubusercontent.com/100999656/175770546-c95301bc-f7d6-4411-94c2-496f750ac9a2.jpg)




...


Some plots were also made for dataset.


<might be useful to add here saved plots (img)>
...

To sum up:
- binary classification needs to be considered when making model
- unbalanced labels were treated by oversampling method to make them more balanced
- multidimensional dataset was reduced by pca and tsna

<a name="Metric"></a>
## Metric

As our y is imbalanced we firmly reject accuracy. After all for our final results the most important was the biggest amount
of predicted True Positive and False Positive. And so we decided to stick with F-Measure, combining both Precision and 
Recall.

<a name="Baseline"></a>
## Baseline

For baseline we use DummyClassifier and checked f1_score, Confusion Matrix, as well as classification_report.

For preprocessed data it gave:

![Baseline_Preprocessed_data_matrix](https://user-images.githubusercontent.com/90214121/175790778-5498f974-3ec2-4d65-b003-2d2cdedd6832.png)

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| -1           | 0.36      | 0.36   | 0.36     | 169     |
| 1            | 0.68      | 0.69   | 0.68     | 338     |
| accuracy     |           |        | 0.58     | 507     |
| macro avg    | 0.52      | 0.52   | 0.52     | 507     |
| weighted avg | 0.57      | 0.58   | 0.57     | 507     |


The result of f-1=0.683 for 1 class and 0.36 for -1 class might be not satisfing at this point, but that is why we get through next steps.

For our curiosity we checked DummyClassifier on raw data as well:

![Baseline_Raw_data_matrix](https://user-images.githubusercontent.com/90214121/175790820-5a16ef93-b731-4037-9a45-e94d1a5ea634.png)

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| -1           | 0.06      | 0.05   | 0.06     | 94      |
| 1            | 0.90      | 0.91   | 0.90     | 844     |
| accuracy     |           |        | 0.82     | 938     |
| macro avg    | 0.48      | 0.48   | 0.48     | 938     |
| weighted avg | 0.81      | 0.82   | 0.82     | 938     |

So at first f1 score 0.901 might seems to be high result, but it is barely 6% for minority class.



<a name="Dataset_split"></a>
## Dataset split

Dataset was splited to 0.25 test size and stratified by train_labels. 

<a name="Classification"></a>
## Classification

We have chosen our classifiers based on our earlier results with GridSearch() and Hyperopt() (still to be found [here] https://github.com/avengela/Machine_Learning_Project/blob/3_Data_Preprocessing_AB/Data_preprocessing.ipynb ).

Hyperopt picked SVC, so we wanted to check that with more parameters, but as we discovered that SVC() itself tends to be slow so we decided to use LinearSVC instead. 

Gridsearch comparing way more classifications as results give us ExtraTreesClassifier() and so we wanted to check that
in final result.

As third classifier we chose KNeighborsClassifier().


As the results we get:

    Best model params: 
    {'classifier': KNeighborsClassifier(leaf_size=2, n_neighbors=58, weights='distance'), 
    'classifier__algorithm': 'auto',      
    'classifier__leaf_size': 2, 
    'classifier__n_neighbors': 58, 
    'classifier__weights': 'distance'}
    
  ![Baseline_Raw_data_matrix](https://user-images.githubusercontent.com/90214121/175791044-c78342b8-4e44-42fb-8a58-9a962f836166.png)
  
  With the f1_score 0.92 for 1 and 0.85 for -1 class.
  

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| -1           | 0.85      | 0.84   | 0.85     | 169     |
| 1            | 0.92      | 0.93   | 0.92     | 338     |
| accuracy     |           |        | 0.90     | 507     |
| macro avg    | 0.89      | 0.88   | 0.88     | 507     |
| weighted avg | 0.90      | 0.90   | 0.90     | 507     |    
    

<a name="Final_code"></a>
## Final code

In final code we have load best model and predict test dataset.

<a name="Results"></a>
## Results

