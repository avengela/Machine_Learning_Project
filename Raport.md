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