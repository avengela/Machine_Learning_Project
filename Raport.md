# Machine_Learning_Project
Final project Machine Learning - CDV 2022
## Created by: Angelika Bulkowska and Julia Grzegorowska

# Task - description
The data consists of 2 splits train_data.csv and test_data.csv. There are numeric observations labelled either -1 or +1. Labels for the training data are located in in train_labels.csv. 
The aim of the project is to predict labels for the testing data.

Desired output:
Save predictions in test_labels.csv,
Prepare a report (saved as report.md)with the explanations on how you came up with the solution. What obstacles you faced and how did you solve them. Add also information about data quality and so on.

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

    pandas.DataFrame.info() 
    pandas.DataFrame.describe() 
    pandas.DataFrame.describe() 
    pandas.DataFrame.shape
    pandas.DataFrame.isnull()
    pandas.DataFrame.value_counts()

Later, after using some preprocessing methods such as:

standarization using:
    
    sklearn.pipeline.Pipeline
    sklearn.preprocessing.StandardScaler

 and normalization using:
 
    sklearn.preprocessing.MinMaxScaler

We used Univariate Feature Selection 

    sklearn.feature_selection.SelectKBest
    
    
    Univariate Selection(Kbest selection):
    
    Shape before transformation: (3750, 10000)
    Min score: 1.7567042700994732e-09, max score: 17.321255892491074, mean score: 1.000380011727585
    Shape after transformation: (3750, 3333)
    Univariate Selection(Kbest selection)
    Shape before transformation: (3750, 10000)
    Min score: 1.7567042700994732e-09, max score: 17.321255892491074, mean score: 1.000380011727585
    Shape after transformation: (3750, 3177)


to get rid of noisy data
and PCA reshaping data to examine std and get rid of extreme data. 

    sklearn.decomposition.PCA
    
    
    PCA - Principal Component Analysis

    Shape before transformation: (3750, 3177)

    Explained Variance: 
    [0.00119644 0.0011572  0.00114932 0.00114815 0.00114109 0.00113676
     0.0011336  0.00113102 0.00112559 0.00112421 0.00111666 0.00111646
     0.00111411 0.00110989 0.00110611 0.00110364 0.00109874 0.00109676
     0.00109307 0.00109102 0.0010901  0.00108277 0.00108163 0.0010793
     0.00107742 0.00107588 0.00107237 0.00106929 0.00106777 0.00106614
     0.00106504 0.00106352 0.0010601  0.00105923 0.00105696 0.00105041
     0.00104802 0.00104656 0.00104519 0.00104268 0.00104112 0.00104055
     0.00103838 0.00103691 0.00103446 0.00102977 0.0010284  0.00102667
     0.00102507 0.00102168 0.00101906 0.00101652 0.00101368 0.00101249
     0.00101097 0.00101016 0.00100867 0.00100588 0.00100427 0.00100261
     0.00100051 0.00099868 0.00099487 0.00099332 0.00099223 0.00098997
     0.00098815 0.00098489 0.00098367 0.00098203 0.00097907 0.00097823
     0.00097593 0.00097449 0.00096828 0.00096637 0.00096389 0.00096264
     0.00096064 0.00095803 0.00095728 0.00095593 0.00095393 0.00095246
     0.00095021 0.000948   0.00094427 0.00094343 0.00094096 0.00094048
     0.00093632 0.00093286 0.00093093 0.00092972 0.00092523 0.00092259
     0.00092128 0.00091892 0.00091447 0.00091426]
    Warning: string series 'monitoring/stdout' value was longer than 1000 characters and was truncated. This warning is printed only once per series.

    Shape after transformation: (3750, 100)
    
    
   
to fits a model and removes the weakest feature (or features) until the specified number of features is reached we used RFE
    
    sklearn.feature_selection.RFE
    
    
    RFE - Recursive Feature Elimination

    Shape before transformation: (3750, 100)

    Feature Ranking: 
    [ 1 12  1 79 78  1  2  4 83 72  9 47  7 76 63 82 46  1  5 66 15 23 96 30
     48  3 61 77 55 13 85 42 81 18 32 53 27 68 25 57 69 20 28 24 11 80 10 40
     59 75 43 70 67 87 93 26 52 31 21 90 39 62 14  8 35 34 88 95 29 58 56 84
     54  6 71 22 86 33 49 74 65 17 37 45 41 89 50 60 44  1 36 91 19 51 92 38
     64 16 94 73]

    Shape after transformation: (3750, 5)


Later trying to understand and set data types, data mixtures, shape, outliers, missing values, noisy data, skewness and kurtosis by creating heatmaps, matrix plots and others.

 While calling .value_count() on data, was that our dataset was unbalanced so the solution was to call:
- Random Undersampling
- [Oversampling / SMOTE](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

Here we used:

    imblearn.over_sampling.RandomOverSampler
    

...


Some plots were also made for dataset.


<might be useful to add here saved plots (img)>
...

To sum up:
- binary classification needs to be considered why making model
- unbalanced labels were treated by oversampling method to make them more balanced
- multidimensional dataset was reduced by pca and ....

<a name="Metric"></a>
## Metric

As our y is imbalanced we firmly reject accuracy. After all for our final results the most important was the biggest amount
of predicted True Positive and False Positive. And so we decided to stick with F-Measure, combining both Precision and 
Recall.

<a name="Baseline"></a>
## Baseline

For baseline we use:

    sklearn.dummy.DummyClassifier()

Also for the baseline (and not processed data) we checked:

    sklearn.metrics.f1_score
    sklearn.metrics.ConfusionMatrixDisplay
    sklearn.metrics.classification_report


<a name="Dataset_split"></a>
## Dataset split

    seed = np.random.seed(147)
    X_train, X_test, y_train, y_test = train_test_split(train_data_preprocessed, train_labels_preprocessed,
                                                        test_size=0.25, random_state=seed,
                                                        stratify=train_labels_preprocessed)



<a name="Classification"></a>
## Classification

We have chosen our classifiers based on our earlier results with GridSearch() and Hyperopt() (still to be found in branch
...). 

Hyperopt picked SVC, so we wanted to check that with more parameters, but as we discovered that SVC() itself tends to be slow,
we decided to use LinearSVC instead.

Gridsearch comparing way more classifications as results give us ExtraTreesClassifier() and so we wanted to check that
in final result.

KNeighborsClassifier() is here just to check the results with slightly more basic classifier.


        search_space = [
            {"classifier": [LinearSVC(max_iter=10000, dual=False, random_state=seed)],
             "classifier__penalty": ["l1", "l2"],
             "classifier__C": np.logspace(1, 10, 25),
             "classifier__class_weight": [None, "balanced"]
             },

            {"classifier": [KNeighborsClassifier()],
             "classifier__n_neighbors": np.arange(2, 60, 2),
             "classifier__weights": ["uniform", "distance"],
             "classifier__algorithm": ["auto", "ball_tree", "kd_tree"],
             "classifier__leaf_size": np.arange(2, 60, 2)
             },

            {"classifier": [ExtraTreesClassifier(random_state=seed)],
             "classifier__n_estimators": np.arange(90, 135, 1),
             "classifier__criterion": ["gini", "entropy"],
             "classifier__class_weight": [None, "balanced", "balanced_subsample"],
             "classifier__min_samples_split": np.arange(2, 5, 1)
             }
        ]

As the results we get:

    Best model params: 
    {'classifier': ExtraTreesClassifier(min_samples_split=4, n_estimators=108), 
    'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__min_samples_split': 4,
    'classifier__n_estimators': 108}

<a name="Final_code"></a>
## Final code

<a name="Results"></a>
## Results


Best model params: 
{'classifier': KNeighborsClassifier(leaf_size=2, n_neighbors=58, weights='distance'), 'classifier__algorithm': 'auto', 'classifier__leaf_size': 2, 'classifier__n_neighbors': 58, 'classifier__weights': 'distance'}

Model scorer: 
make_scorer(f1_score, average=binary)

Model score: 
0.9424050742790956
              precision    recall  f1-score   support

          -1       0.85      0.84      0.85       169
           1       0.92      0.93      0.92       338

    accuracy                           0.90       507
   macro avg       0.89      0.88      0.88       507
weighted avg       0.90      0.90      0.90       507

    f1 score: 0.923
          0
    0     1
    1     1
    2     1
    3     1
    4     1
    ...  ..
    1245  1
    1246  1
    1247  1
    1248  1
    1249  1
