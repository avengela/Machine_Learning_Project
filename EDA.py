def main():

    ##Statistic
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    test_data = pd.read_csv("data/test_data.csv", header=None)
    train_data = pd.read_csv("data/train_data.csv", header=None)
    train_labels = pd.read_csv("data/train_labels.csv", header=None)

    data = [test_data, train_data, train_labels]
    data_names = ["test_data", "train_data", "train_labels"]


    def data_info(df: list, name: list) -> None:
        """
        Shows data_info of given data frames
        """
        for d, n in zip(df, name):
            print(n, '\n')
            d.info()
            print('\n')


    print("Describing data with info() \n")
    data_info(data, data_names)


    def data_describe(df: list, name: list) -> None:
        """
        Shows data_describe of given data frames
        """
        for d, n in zip(df, name):
            print(n, '\n')
            print(d.describe())
            print('\n')


    print("Describing data with describe() \n")
    data_describe(data, data_names)


    def data_duplicates(df: list, name: list) -> None:
        """
        Shows if data sets have duplicate values
        """
        for d, n in zip(df, name):
            print(n, '\n')
            print(d[d.duplicated()].shape)
            print('\n')


    print("Duplicates \n")
    data_duplicates(data, data_names)


    def data_null_values(df: list, name: list) -> None:
        """
        Shows if data sets have null values
        """

        for d, n in zip(df, name):
            print(n, '\n')
            print(round(d.isnull().sum() / len(d) * 100, 2)[0], '%')
            print('\n')


    print("Null values \n")
    data_null_values(data, data_names)

    print("Values in train_labels \n")
    print(pd.unique(train_labels[0]))

    train_labels[0].value_counts().plot(kind='bar')
    plt.show()


    sc = StandardScaler()
    X_train_standarized = sc.fit_transform(train_data)
    X_test_standarized = sc.transform(test_data)

    pca = PCA(n_components=2, whiten=True)

    X_pca = pca.fit_transform(X_train_standarized)
    df_pca = pd.DataFrame(X_pca, columns = ['Principal Component 1','Principal Component 2'])
    df_pca = df_pca.assign(labels=train_labels.values)
    df_pca

if __name__ == '__main__':
    main()