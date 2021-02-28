#!/usr/bin/python
import dota

import sys

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold 
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

def eval_model_using_cv(model, x, y, ax):
    # Change this!
    skf = StratifiedKFold(n_splits=8)

    final_matrix = np.array([[0,0],[0,0]])
    arr_roc_a = []
    arr_roc_b = []
    a_base = np.linspace(0, 1, 101)
    #arr_roc_a = np.array([[]])
    #arr_roc_b = np.array([[]])
    arr_area_under_curve = np.array([])

    for train, test in skf.split(x, y):
        print("Train:", train, "Test:", test)

        x_train = x.loc[train]
        y_train = y.loc[train]

        x_test = x.loc[test]
        y_test = y.loc[test]

        print("Training", type(model))
        model.fit(x_train, y_train)
        print("Training done!")

        # Confusion matrix
        y_pred = model.predict(x_test)
        matrix = metrics.confusion_matrix(y_test, y_pred)
        # ROC
        y_pred_probability = model.predict_proba(x_test)[::, 1]
        a, b, _ = metrics.roc_curve(y_test, y_pred_probability, pos_label=2)
        ax.plot(a, b, 'b', alpha=0.15)
        b = np.interp(a_base, a, b)
        b[0] = 0.0
        area_under_curve = metrics.roc_auc_score(y_test, y_pred_probability)

        final_matrix += matrix
        #arr_roc_a = np.append(arr_roc_a, [a], axis=0)
        arr_roc_a.append(a)
        arr_roc_b.append(b)
        #arr_roc_b = np.append(arr_roc_b, [b], axis=0)
        arr_area_under_curve = np.append(arr_area_under_curve, area_under_curve)

    return {
            "confusion_matrix": final_matrix,
            "roc_a": arr_roc_a,
            "roc_b": np.asarray(arr_roc_b, dtype=np.float32),
            "roc_a_base": a_base,
            "area_under_curve": area_under_curve,
        }



if len(sys.argv) == 2 and sys.argv[1] == "preprocess":
    df_train = pd.read_csv("./dota2Train.csv")
    df_test = pd.read_csv("./dota2Test.csv")

    df_train = dota.preprocess_df(df_train, "./heroes.json", "./hero_stats.csv")
    df_test = dota.preprocess_df(df_test, "./heroes.json", "./hero_stats.csv")

    df_train.to_csv("./preprocessed_dota2_train.csv")
    df_test.to_csv("./preprocessed_dota2_test.csv")
    df_train.append(df_test).to_csv("./preprocessed_dota2_all.csv")
elif len(sys.argv) > 1:
    print("Unrecognized argument, use 'preprocess' flag to emit new datasheet")
    exit(1)

df_all = pd.read_csv("./preprocessed_dota2_all.csv")
x, y = split_x_y(df_all)

nb_model = CategoricalNB()
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier()

_, ax = plt.subplots()
nb_result = eval_model_using_cv(nb_model, x, y, ax)
#print(nb_result)

roc_a = nb_result["roc_a"]
roc_b = nb_result["roc_b"]
roc_a_base = nb_result["roc_a_base"]
area_under_curve = nb_result["area_under_curve"]

#print(roc_b)
mean_roc_b = roc_b.mean(0)
#print(mean_roc_b)
std_roc_b = roc_b.std(1)

mean_area_under_curve = np.mean(area_under_curve)

#ax.plot(mean_roc_a, mean_roc_b, label="mean area under curve="+str(mean_area_under_curve))
ax.plot(roc_a_base, mean_roc_b, 'b')
ax.set_xlabel("mean false positive rate")
ax.set_ylabel("mean true positive rate")
ax.legend(loc=4)

confusion_matrix = nb_result["confusion_matrix"]
cmd = metrics.ConfusionMatrixDisplay(nb_result["confusion_matrix"], display_labels=labels())
cmd.plot()

plt.show()

# deprecated, code now uses cv
#df_train = pd.read_csv("./preprocessed_dota2_train.csv")
#df_test = pd.read_csv("./preprocessed_dota2_test.csv")
#x_train, y_train = split_x_y(df_train)
#x_test, y_test = split_x_y(df_test)
#eval_model(nb_model, x_train, y_train, x_test, y_test, "Naive Bayes Classifier")
#eval_model(rf_model, x_train, y_train, x_test, y_test, "Random Forest")
#eval_model(knn_model, x_train, y_train, x_test, y_test, "KNN")
#plt.show()

# deprecated, code is now free from seaborn use
"""
confusion_matrix_resolutions = ["True negative", "False positive", "False negative", "True positive"]
counts = ["{0}".format(value) for value in confusion_matrix.flatten()]
percentages = ["{0:.2%}".format(value) for value in 
        confusion_matrix.flatten()/np.sum(confusion_matrix)]

#print(nb_result["confusion_matrix"])
labels = np.array([f"{a}\n{b}\n{c}" for a, b, c in
        zip(confusion_matrix_resolutions, counts, percentages)]).reshape(2,2)
print(labels)
sb.heatmap(confusion_matrix, annot=labels, fmt="")
plt.show()
"""
