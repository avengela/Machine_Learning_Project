# Libraries

import os
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedStratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe

import neptune.new as neptune


# NEPTUN
""" Netune api token"""
run = neptune.init(
    project="julia.grzegorowska/ml-project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzMwMDM2OC04YzdlLTQxOGEtYmEzYi0xZTA3ZmQzMjlkNzIifQ==",
)

params = {
    "optimizer": "Julia"
}
run["parameters"] = params


def send_data_neptune(data, plot_name):
    for epoch in range(0, len(data)):
        run[plot_name].log(data[epoch])


def single_record(record, record_name):
    run[record_name] = record


def stop_run():
    run.stop()


os.environ["OMP_NUM_THREADS"] = "1"
seed = np.random.seed(147)


def data_load() -> list:
    train_data = pd.read_csv("data/train_data.csv", header=None)
    test_data = pd.read_csv("data/test_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    a = train_labels.values
    tmp = []
    for i in range(0, len(a)):
        tmp.append(int(a[i]))

    send_data_neptune(tmp, "train_labels")

    return [train_data, test_data, train_labels]


train_data, test_data, train_labels = data_load()
train_labels_ravel = train_labels.values.ravel()


def pipeline_standard_minmax(x_1, x_2: pd.DataFrame) -> np.array:
    pipeline = Pipeline([
        ("std", StandardScaler()),
        ("minmax", MinMaxScaler())])

    train_std_minmax = pipeline.fit_transform(x_1)
    test_std_minmax = pipeline.fit_transform(x_2)

    return [train_std_minmax, train_std_minmax]


train_std_minmax, test_std_minmax = pipeline_standard_minmax(train_data, test_data)

k = int(len(train_data.columns) / 3)


def kbest_select(x_1: np.array, x_2: np.array, y_1: np.array, n_of_kbest: int) -> list:
    print(f"Shape before: {x_1.shape}\n")

    test = SelectKBest(score_func=f_classif, k=n_of_kbest)
    fit = test.fit(x_1, y_1)
    features_first = fit.transform(x_1)
    features_second = fit.transform(x_2)

    scores = fit.scores_
    score_df = pd.DataFrame(scores, columns=["Scores"])
    print(
        f"Min score: {min(score_df.Scores)}, max score: {max(score_df.Scores)}, mean score: {np.mean(score_df.Scores)}\n")
    print(f"Shape after: {features_first.shape}\n")

    score_df.drop(score_df[score_df.Scores < 1].index, inplace=True)
    l = len(score_df)

    ## Save to Neptune
    single_record(min(score_df.Scores), 'kbest_select_min_score')
    single_record(max(score_df.Scores), 'kbest_select_max_score')
    single_record(np.mean(score_df.Scores), 'kbest_select_mean_score')

    if l != n_of_kbest:
        return kbest_select(train_std_minmax, test_std_minmax, train_labels_ravel, l)
    else:
        return [features_first, features_second]


kbest_train, kbest_test = kbest_select(train_std_minmax, test_std_minmax, train_labels_ravel, k)


def pca_select(x_1, x_2: np.array) -> np.array:
    print(f"Shape before transformation: {x_1.shape}\n")

    scaler = MinMaxScaler(feature_range=(0, 1))
    pca = PCA(n_components=100, random_state=seed)
    fit = pca.fit(x_1)
    features_first = fit.transform(x_1)
    features_second = fit.transform(x_2)

    print(f"Explained Variance: \n{fit.explained_variance_ratio_}\n")
    print(f"Shape after transormation: {features_first.shape}")

    send_data_neptune(fit.explained_variance_ratio_, "explained_variance_ration")

    return [features_first, features_second]


pca_train, pca_test = pca_select(kbest_train, kbest_test)


def rfe_select(x_1, x_2, y_1: np.array) -> np.array:
    print(f"Shape before transformation: {x_1.shape}\n")

    model = LogisticRegression()
    svc = SVC(kernel="linear", C=1, random_state=seed)
    rfe = RFE(estimator=svc, n_features_to_select=5)
    fit = rfe.fit(x_1, y_1)
    first_features = fit.transform(x_1)
    second_features = fit.transform(x_2)

    print(f"Feature Ranking: \n{fit.ranking_}\n")
    print(f"Shape after: {first_features.shape}\n")

    send_data_neptune(fit.ranking_, "fit-ranking")

    return [first_features, second_features]


rfe_train, rfe_test = rfe_select(pca_train, pca_test, train_labels_ravel)


def random_sampling(x_1: np.array, y_1: np.array) -> list:
    over = RandomOverSampler(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    x_resampled, y_resampled = pipeline.fit_resample(x_1, y_1)

    tmp_X = []
    for i in range(0, len(x_resampled)):
        for j in range(0, len(x_resampled[i])):
            tmp_X.append(x_resampled[i][j])

    send_data_neptune(tmp_X, "X-resampled")
    send_data_neptune(y_resampled, "y-resampled")

    return [x_resampled, y_resampled]


x_resampled, y_resampled = random_sampling(rfe_train, train_labels_ravel)


def save_data(train_x: np.array, test_x: np.array, train_y: np.array) -> None:
    np.save('project_data/processed_train_X.npy', train_x)
    np.save('project_data/processed_test_X.npy', test_x)
    np.save('project_data/processed_train_y.npy', train_y)

    print("Saving has been completed.")


save_data(x_resampled, rfe_test, y_resampled)


def scatter_plot(x_1: np.array, y_1: np.array) -> plt:

    counter = Counter(y_1)

    plt.figure(figsize=(15.1, 13))
    for label, _ in counter.items():
        row_ix = np.where(y_1 == label)[0]
        plt.scatter(x_1[row_ix, 0], x_1[row_ix, 1], label=str(label),
                    s=100, marker="o", alpha=0.5, edgecolor="black")
    plt.title(f"Scatter plot for preprocessed data with {counter}")
    plt.legend()

    return plt.show()

scatter_plot(x_resampled, y_resampled)
