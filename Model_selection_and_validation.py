def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    import joblib
    import neptune.new as neptune
    import os

    from imblearn.pipeline import Pipeline

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold

    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    from sklearn.metrics import f1_score
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import classification_report

    os.environ["OMP_NUM_THREADS"] = "1"
    seed = np.random.seed(147)

    # Neptune
    run = neptune.init(
        project="julia.grzegorowska/ml-project",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzMwMDM2OC04YzdlLTQxOGEtYmEzYi0xZTA3ZmQzMjlkNzIifQ==",
    )

    params = {
        "optimizer": "Julia"
    }
    run["parameters"] = params

    def send_data_neptune(data, plot_name):
        """ Sending data array to Neptune"""
        for epoch in range(0, len(data)):
            run[plot_name].log(data[epoch])

    def single_record(record, record_name):
        """ Sending any single record to Neptune"""
        run[record_name] = record

    def stop_run():
        """Stop running Neptune and sending data/end the program"""
        run.stop()

    # Data access
    test_data_preprocessed = np.load("project_data/processed_test_X.npy")
    train_data_preprocessed = np.load("project_data/processed_train_X.npy")
    train_labels_preprocessed = np.load("project_data/processed_train_y.npy")

    X_train, X_test, y_train, y_test = train_test_split(train_data_preprocessed, train_labels_preprocessed,
                                                        test_size=0.25, random_state=seed,
                                                        stratify=train_labels_preprocessed)

    # Validation
    def grid_search_pipeline(X_training: np.array, X_testing: np.array, y_training: np.array) -> np.array:
        """
        GridSearch with LinearSVC, KNeighborsClassifier, ExtraTreesClassifier
        searching for best model
        Shows best_score_ and best_params_
        Save best model for future use
        """

        pipe = Pipeline([("classifier", KNeighborsClassifier())])

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
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
        gridsearch = GridSearchCV(pipe, search_space, cv=rskf, scoring="f1", verbose=1, n_jobs=-1)

        best_model = gridsearch.fit(X_training, y_training)
        y_predict = best_model.predict(X_testing)

        print(f"\nBest model params: \n{best_model.best_params_}")
        print(f"\nModel scorer: \n{best_model.scorer_}")
        print(f"\nModel score: \n{best_model.best_score_}")

        filename = "clf_model.sav"
        joblib.dump(best_model, filename)

        single_record(best_model.best_score_, "model_score")
        send_data_neptune(y_predict.tolist(), 'y_pred')

        return y_predict

    y_pred = grid_search_pipeline(X_train, X_test, y_train)

    # Confusion matrix, classification report and f1 score itself
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    f1 = f1_score(y_test, y_pred)
    print(f"f1 score: {f1:.3f}")

    # Execution
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

    stop_run()


if __name__ == '__main__':
    main()
