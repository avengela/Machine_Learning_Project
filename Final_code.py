import numpy as np
import joblib
import pandas as pd

test_data = np.load("project_data/processed_test_X.npy")


def load_predict_model(to_predict: np.array) -> np.array:
    """
    Loading best model indicated earlier by grid search
    and predicting y for test data
    Data standardization and normalization:
            :param to_predict: np.array: test dataset
            :return: list: of np.array-s of predicted y
    """

    filename = "clf_model.sav"
    loaded_model = joblib.load(filename)
    predicted = loaded_model.predict(to_predict)

    return predicted


def main() -> None:
    y_test = load_predict_model(test_data)

    y_test = pd.DataFrame(y_test)
    y_test.to_csv("project_data/test_labels.csv")

    print(y_test)


if __name__ == '__main__':
    main()
