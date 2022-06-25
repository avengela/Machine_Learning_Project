import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

seed = np.random.seed(147)


def baseline_cals(training_x: np.array or pd.DataFrame, training_y: np.array or pd.DataFrame) -> None:
    """
    Uses DummyClassifier on training set
    Shows Confusion Matrix, F1 score and classification_report
    Data standardization and normalization:
            :param training_x: np.array/pd.DataFrame : training x data
            :param training_y: np.array/pd.DataFrame: training y data
            :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(training_x, training_y, test_size=0.25, random_state=seed,
                                                        stratify=training_y)

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_train, y_train)
    y_predicted = dummy_clf.predict(X_test)
    sc = dummy_clf.score(X_test, y_predicted)
    print(f"Dummy classifier score: {sc}")

    # metric checked on baseline
    ConfusionMatrixDisplay.from_estimator(dummy_clf, y_test, y_predicted)
    plt.show()

    f1 = f1_score(y_test, y_predicted)
    print(f"f1 score: {f1:.3f}")

    print(classification_report(y_test, y_predicted))


def main() -> None:
    train_data = pd.read_csv("data/train_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    train_data_preprocessed = np.load("project_data/processed_train_X.npy")
    train_labels_preprocessed = np.load("project_data/processed_train_y.npy")

    print("Data baseline")
    baseline_cals(train_data, train_labels)
    print("\n")

    print("Preprocessed data baseline")
    baseline_cals(train_data_preprocessed, train_labels_preprocessed)
    print("\n")


if __name__ == '__main__':
    main()
