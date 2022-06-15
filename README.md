# Machine_Learning_Project
Final project Machine Learning - CDV 2022


# Task - description
The data consists of 2 splits train_data.csv and test_data.csv. There are numeric observations labelled either -1 or +1. Labels for the training data are located in in train_labels.csv. 
The aim of the project is to predict labels for the testing data.

Desired output:
Save predictions in test_labels.csv,
Prepare a report (saved as report.md)with the explanations on how you came up with the solution. What obstacles you faced and how did you solve them. Add also information about data quality and so on.

# Dataset
The size of the dataset for this project is to large for github repo (exceeds GitHub's file size limit of 100.00 MB). The dataset can be obtained from [here](https://drive.google.com/drive/folders/1K4IQxSH--gfMZdovzGfuYjDXUF1r2TqF?usp=sharing)



________
Do poprawy:

1) Podzielić EDA, preprocesing z przerobieniem danych, Grid search, validację, efekt końcowy z porównaniem danych wejściowych przechodzących przez najlepsza metodę
    jako osobne .py (EDA może być jupiter notebookiem) - screen
    












1) EDA - Wizualizacja - do eda, PCA, wprowadzić tu standaryzację <- Jules
     <\n>Zmienić uproszczony fit transform dla standaryzacji .fit. .transform() dla train i test <- Angela
     3) W preprocesingu użyć stratify from sklearn.model_selection import StratifiedKFold (?) <- Angela
     
3) Wybór metryka - Angela
    Zastanowić się nad metryką f1 score <- nie jest zła, ale może coś jeszcze przerobić
    
4) Dummy <- Jules
    5) Dołączyć baseline - może być dummy classifier ale musimy wiedzieć jak przedstawia sie score przed i móc porównać potem


6) Datatest split <- Jules
     Sprawdzić czy nie tniemyh za dużo datasetu
    4) 6) Sprawdzić shapy przy samplingu i zastanowić się nad tą częscią (moze lepiej będzie uzyć tylko oversamplingu
    5) 9) Train test split chyba za bardzo okraja ilość danych - zastanowić się nad tym podziałem, zwiększyć udział

8) Klasyfikatory <- Angela
    7) Dodać hiperparametry
    8) 8) Zastanowić sie nad grid searchem mozna okroić ilość sprawdzanych modeli ale dodać parametry
    9) 10) Grid search powinien posiadać parametr sugerujacy dopasowanie modelu pod interesująca nas metodykę


9) Finalny kod
    12) KONFJUŻYN MATRIXY w wynikach
10) Raport
