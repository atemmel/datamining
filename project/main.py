#!/usr/bin/python
import dota

import sys

import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB

def split_x_y(df):
    return df.iloc[:, 2:], df.iloc[:, 1]

if len(sys.argv) == 2 and sys.argv[1] == "preprocess":
    df_train = pd.read_csv("./dota2Train.csv")
    df_test = pd.read_csv("./dota2Train.csv")

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

nbmodel = CategoricalNB()
nbmodel.fit(x_train, y_train.values.ravel())

y_pred = nbmodel.predict(x_test)

disp = metrics.plot_confusion_matrix(nbmodel, x_train, y_train.values.ravel(), 
        normalize='true', display_labels=["Team 1 wins", "Team 2 wins"])
disp.ax_.set_title("Confusion matrix for Naive Bayes Classifier")
plt.show()
