def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    from sklearn.model_selection import train_test_split
    from sklearn.dummy import DummyClassifier

    from sklearn.metrics import f1_score
    from sklearn.metrics import ConfusionMatrixDisplay



    train_data = pd.read_csv("data/train_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    train_data_preprocessed = np.load("project_data/processed_train_X.npy")
    train_labels_preprocessed = np.load("project_data/processed_train_y.npy")

    seed = np.random.seed(147)


    def baseline_cals(train_data,train_labels):
        # baseline
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.25, random_state=seed)

        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(X_train, y_train)
        y_predict = dummy_clf.predict(X_test)
        sc = dummy_clf.score(X_test, y_predict)
        print(f"Dummy classifier score: {sc}")

        # metric checked on baseline
        ConfusionMatrixDisplay.from_estimator(dummy_clf, y_test, y_predict)
        plt.show()

        f1 = f1_score(y_test, y_predict)
        print(f"f1 score: {f1:.3f}")


    print("Data baseline")
    baseline_cals(train_data, train_labels)
    print("\n")

    print("Preprocessed data baseline")
    baseline_cals(train_data_preprocessed, train_labels_preprocessed)
    print("\n")

if __name__ == '__main__':
    main()
