import os
from collections import Counter

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

# NEPTUNE

""" Netune api token:
Neptune is a metadata store for MLOps, built for teams that run a lot of experiments.
It gives you a single place to log, store, display, organize, compare, and query all your model-building metadata.

Neptune is used for:
- Experiment tracking: Log, display, organize, and compare ML experiments in a single place.
- Model registry: Version, store, manage, and query trained models and model building metadata.
- Monitoring ML runs live: Record and monitor model training, evaluation, or production runs live."""

run = neptune.init(
    project="julia.grzegorowska/ml-project",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzMwMDM2OC04YzdlLTQxOGEtYmEzYi0xZTA3ZmQzMjlkNzIifQ==",
)

params = {"optimizer": "Julia"}
run["parameters"] = params


def send_data_neptune(data, plot_name):
    """Sending data array to Neptune"""
    for epoch in range(0, len(data)):
        run[plot_name].log(data[epoch])


def single_record(record, record_name):
    """Sending any single record to Neptune"""
    run[record_name] = record


def stop_run():
    """Stop running Neptune and sending data/end the program"""
    run.stop()


# Limit number of threads in numpy
os.environ["OMP_NUM_THREADS"] = "1"
seed = np.random.seed(147)


# Loading csv dataset from folder
def data_load() -> list:
    train_data = pd.read_csv("data/train_data.csv", header=None)
    test_data = pd.read_csv("data/test_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    """Saving to Neptune"""
    a = train_labels.values
    tmp = []
    for i in range(0, len(a)):
        tmp.append(int(a[i]))

    send_data_neptune(tmp, "train_labels")
    return [train_data, test_data, train_labels]


train_data, test_data, train_labels = data_load()
train_labels_ravel = train_labels.values.ravel()

# PREPROCESSING

# Data preprocessing - first step before any machine learning machinery can be applied.
# Because the algorithms learn from the data and the learning outcome for problem solving heavily depends on the
# proper data needed to solve a particular problem(features).

"""Problem is multidimensional dataset and solutions is going to be dimensional reduction using: 
 Univariate Selection(Kbest selection), PCA - Principal Component Analysis and RFE - Recursive Feature Elimination"""


def pipeline_standard_minmax(x_1, x_2: pd.DataFrame) -> np.array:
    """
    pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters:
        Data standardization and normalization:
        :param x_1: pd.DataFrame: train data
        :param x_2: pd.DataFrame: test data
        :return: list: of np.array-s of standardized train and test data
    """

    pipeline = Pipeline([("std", StandardScaler()), ("minmax", MinMaxScaler())])

    train_std_minmax = pipeline.fit_transform(x_1)
    test_std_minmax = pipeline.fit_transform(x_2)

    return [train_std_minmax, test_std_minmax]


train_std_minmax, test_std_minmax = pipeline_standard_minmax(train_data, test_data)

k = int(len(train_data.columns) / 3)


def kbest_select(x_1: np.array, x_2: np.array, y_1: np.array, n_of_kbest: int) -> list:
    """
    The SelectKBest method selects the features according to the k highest score.
    By changing the 'score_func' parameter we can apply the method for both classification and regression data:
        Univariate Selection
        :param x_1: pd.DataFrame: standardized train data
        :param x_2: pd.DataFrame: standardized test data
        :param y_1: np.array: ravel of train labels
        :param n_of_kbest: int: specify number of k best in SelectKBest
        :return: list: of np.array with univariate test and train data
    """

    print(f"Univariate Selection(Kbest selection)\n")
    print(f"Shape before transformation: {x_1.shape}\n")

    test = SelectKBest(score_func=f_classif, k=n_of_kbest)
    fit = test.fit(x_1, y_1)
    features_first = fit.transform(x_1)
    features_second = fit.transform(x_2)

    scores = fit.scores_
    score_df = pd.DataFrame(scores, columns=["Scores"])
    print(
        f"Min score: {min(score_df.Scores)}, max score: {max(score_df.Scores)}, mean score: {np.mean(score_df.Scores)}\n"
    )
    print(f"Shape after transformation: {features_first.shape}\n")

    score_df.drop(score_df[score_df.Scores < 1].index, inplace=True)
    df_l = len(score_df)

    # Sending to Neptune
    single_record(min(score_df.Scores), "kbest_select_min_score")
    single_record(max(score_df.Scores), "kbest_select_max_score")
    single_record(np.mean(score_df.Scores), "kbest_select_mean_score")

    if df_l != n_of_kbest:
        return kbest_select(train_std_minmax, test_std_minmax, train_labels_ravel, df_l)
    else:
        return [features_first, features_second]


kbest_train, kbest_test = kbest_select(
    train_std_minmax, test_std_minmax, train_labels_ravel, k
)


def pca_select(x_1, x_2: np.array) -> np.array:
    """
    PCA is to represent a multivariate data table as smaller set of variables (summary indices)
    in order to observe trends, jumps, clusters and outliers.
        Principal Component Analysis:
        :param x_1: np.array: univariate train data
        :param x_2: np.array: univariate test data
        :return: list: of np.array-s reshaped by PCA test and train data
    """

    print(f"\n\nPCA - Principal Component Analysis\n")
    print(f"Shape before transformation: {x_1.shape}\n")

    pca = PCA(n_components=100, random_state=seed)
    fit = pca.fit(x_1)
    features_first = fit.transform(x_1)
    features_second = fit.transform(x_2)

    print(f"Explained Variance: \n{fit.explained_variance_ratio_}\n")
    print(f"Shape after transformation: {features_first.shape}\n\n")

    # Sending to Neptune
    send_data_neptune(fit.explained_variance_ratio_, "explained_variance_ration\n\n")

    return [features_first, features_second]


pca_train, pca_test = pca_select(kbest_train, kbest_test)


def rfe_select(x_1, x_2, y_1: np.array) -> np.array:
    """
    Recursive Feature Elimination
    :param x_1: np.array: reshaped train data
    :param x_2: np.array: reshaped train data
    :param y_1: np.array: ravel of train labels
    :return: list: of np.array-s reshaped by RFE test and train data
    """

    print(f"RFE - Recursive Feature Elimination\n")
    print(f"Shape before transformation: {x_1.shape}\n")

    svc = SVC(kernel="linear", C=1, random_state=seed)
    rfe = RFE(estimator=svc, n_features_to_select=5)
    fit = rfe.fit(x_1, y_1)
    first_features = fit.transform(x_1)
    second_features = fit.transform(x_2)

    print(f"Feature Ranking: \n{fit.ranking_}\n")
    print(f"Shape after transformation: {first_features.shape}\n")

    # Sending to Neptune
    send_data_neptune(fit.ranking_, "fit-ranking")

    return [first_features, second_features]


rfe_train, rfe_test = rfe_select(pca_train, pca_test, train_labels_ravel)

""" Dataset is very unbalanced that is why we are using Random Oversampling"""


def random_sampling(x_1: np.array, y_1: np.array) -> list:
    """
    Random Oversampling/Undersampling
    :param x_1: np.array: reshaped train data
    :param y_1: np.array: ravel of train labels
    :return: list: of np.array-s after over and undersampling
    """

    over = RandomOverSampler(sampling_strategy=0.2, random_state=147)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [("o", over), ("u", under)]
    pipeline = imPipeline(steps=steps)

    x_resampled, y_resampled = pipeline.fit_resample(x_1, y_1)

    # Saving to neptune
    tmp_x = []
    for i in range(0, len(x_resampled)):
        for j in range(0, len(x_resampled[i])):
            tmp_x.append(x_resampled[i][j])

    send_data_neptune(tmp_x, "X-resampled")
    send_data_neptune(y_resampled, "y-resampled")

    return [x_resampled, y_resampled]


x_resampled, y_resampled = random_sampling(rfe_train, train_labels_ravel)


# Saving data to npy files
def save_data(train_x: np.array, test_x: np.array, train_y: np.array) -> None:
    np.save("project_data/processed_train_X.npy", train_x)
    np.save("project_data/processed_test_X.npy", test_x)
    np.save("project_data/processed_train_y.npy", train_y)

    print("Saving has been completed.")


save_data(x_resampled, rfe_test, y_resampled)


# plots
def scatter_plot(x_1: np.array, y_1: np.array) -> plt:
    counter = Counter(y_1)

    plt.figure(figsize=(15.1, 13))
    for label, _ in counter.items():
        row_ix = np.where(y_1 == label)[0]
        plt.scatter(
            x_1[row_ix, 0],
            x_1[row_ix, 1],
            label=str(label),
            s=100,
            marker="o",
            alpha=0.5,
            edgecolor="black",
        )
    plt.title(f"Scatter plot for preprocessed data with {counter}")
    plt.legend()

    return plt.show()


scatter_plot(x_resampled, y_resampled)

df = pd.DataFrame(x_resampled, index=None, columns=None)


def correlation_heatmap(data: pd.DataFrame) -> plt:
    """
    Correlation heatmap for preprocessed train data
    :param data: pd.DataFrame: of train data after over and undersampling
    :return: plt: correlation heatmap
    """

    plt.figure(figsize=(16.9, 8))
    heat_mask = np.triu(np.ones_like(data.corr(), dtype=bool))
    sns.heatmap(data.corr(), mask=heat_mask, vmin=-1, vmax=1, annot=True)
    plt.title("Correlation heatmap for preprocessed train data")

    return plt.show()


correlation_heatmap(df)


def correlation_matrix(data: pd.DataFrame) -> plt:
    corr = data.corr()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, ax=ax, annot=True, linewidths=3, cmap="YlGn")
    plt.title("Pearson correlation of Features", y=1.05, size=15)

    return plt.show()


correlation_matrix(df)


def main():

    if __name__ == "__main__":
        main()


