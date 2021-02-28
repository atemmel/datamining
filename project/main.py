#!/usr/bin/python
import dota

import sys

import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def split_x_y(df):
    return df.iloc[:, 2:], df.iloc[:, 1]

def labels():
    return ["Team 1 wins", "Team 1 loses"]

def eval_model(model, x_train, y_train, x_test, y_test, name):
    # Training
    print("Training ", name)
    model.fit(x_train, y_train.values.ravel())

    # Confusion matrix
    print("Plotting confusion matrix for", name)
    disp = metrics.plot_confusion_matrix(model, x_test, y_test.values.ravel(), 
            normalize="true", display_labels=labels())
    disp.ax_.set_title("Confusion matrix for " + name)

    # ROC curve
    print("Plotting ROC curve for", name)
    y_pred_probability = model.predict_proba(x_test)[::, 1]
    a, b, _ = metrics.roc_curve(y_test, y_pred_probability,pos_label=2)
    area_under_curve = metrics.roc_auc_score(y_test, y_pred_probability)
    _, ax = plt.subplots()
    ax.plot(a, b, label="area under curve="+str(area_under_curve))
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(loc=4)

    # Cross validation


if len(sys.argv) == 2 and sys.argv[1] == "preprocess":
    df_train = pd.read_csv("./dota2Train.csv")
    df_test = pd.read_csv("./dota2Test.csv")

    df_train = dota.preprocess_df(df_train, "./heroes.json", "./hero_stats.csv")
    df_test = dota.preprocess_df(df_test, "./heroes.json", "./hero_stats.csv")

    df_train.to_csv("./preprocessed_dota2_train.csv")
    df_test.to_csv("./preprocessed_dota2_test.csv")
elif len(sys.argv) > 1:
    print("Unrecognized argument, use 'preprocess' flag to emit new datasheet")
    exit(1)

df_train = pd.read_csv("./preprocessed_dota2_train.csv")
df_test = pd.read_csv("./preprocessed_dota2_test.csv")

x_train, y_train = split_x_y(df_train)
x_test, y_test = split_x_y(df_test)

nb_model = CategoricalNB()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

eval_model(nb_model, x_train, y_train, x_test, y_test, "Naive Bayes Classifier")
eval_model(rf_model, x_train, y_train, x_test, y_test, "Random Forest")
plt.show()
eval_model(knn_model, x_train, y_train, x_test, y_test, "KNN")
