def main():
    from sklearn import svm, datasets
    from sklearn.dummy import DummyClassifier
    import pandas as pd
    from numpy import genfromtxt
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix


    test_data = pd.read_csv("data/test_data.csv", header=None)
    train_data = pd.read_csv("data/train_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_data,train_labels)

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.25, random_state=seed)

    y_predict = dummy.predict(X_train)
    df = pd.DataFrame(y_predict)
    dummy.score(X_train, y_train)

    f1_score(X_train, y_train)

    plot_confusion_matrix(dummy, train_data, y_train)
    plt.show()

    accuracy_score(X_train, y_predict)

    fpr, tpr, thresholds = roc_curve(X_train, y_predict)

    plt.figure(figsize=(15, 7))
    plt.plot(fpr, tpr, alpha=0.5, color="blue", label="Elements")
    plt.title("ROC Curve", fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.legend()



















if __name__ == '__main__':
    main()