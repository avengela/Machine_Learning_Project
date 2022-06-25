def main():
    import numpy as np
    import joblib
    import pandas as pd

    test_data_preprocessed = np.load("project_data/processed_test_X.npy")

    def load_predict_model(to_pred: np.array) -> np.array:
        """
        Loading best model indicated earlier by grid search
        and predicting y for test data
        """

        filename = "clf_model.sav"
        loaded_model = joblib.load(filename)
        predicted = loaded_model.predict(to_pred)

        return predicted

    y_test = load_predict_model(test_data_preprocessed)

    y_test = pd.DataFrame(y_test)
    y_test.to_csv("project_data/test_labels.csv")

    print(y_test)


if __name__ == '__main__':
    main()
