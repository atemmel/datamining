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

dota.init("./heroes.json", "./hero_stats.csv")

def split_x_y(df):
    return df.iloc[:, 1:], df.iloc[:, 0]

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
    skf = StratifiedKFold(n_splits=10)

    final_matrix = np.array([[0,0],[0,0]])
    arr_roc_b = []
    a_base = np.linspace(0, 1, 101)
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
        # plot each individual curve
        ax.plot(a, b, 'b', alpha=0.15, label="K="+str(len(arr_area_under_curve)+1))
        b = np.interp(a_base, a, b)
        b[0] = 0.0
        area_under_curve = metrics.roc_auc_score(y_test, y_pred_probability)

        final_matrix += matrix
        arr_roc_b.append(b)
        arr_area_under_curve = np.append(arr_area_under_curve, area_under_curve)

    final_matrix = final_matrix.astype("float") / final_matrix.sum(axis=1)[:, np.newaxis]

    return {
            "confusion_matrix": final_matrix,
            "roc_b": np.asarray(arr_roc_b, dtype=np.float32),
            "roc_a_base": a_base,
            "area_under_curve": area_under_curve,
        }

def plot_results(blob, roc_ax, cm_ax):
    confusion_matrix = blob["confusion_matrix"]
    roc_b = blob["roc_b"]
    roc_a_base = blob["roc_a_base"]
    area_under_curve = blob["area_under_curve"]

    mean_roc_b = roc_b.mean(0)
    std_roc_b = roc_b.std(0)

    mean_area_under_curve = np.mean(area_under_curve)

    # avoid going out of bounds
    upper_bound = np.minimum(mean_roc_b + std_roc_b, 1)
    lower_bound = mean_roc_b - std_roc_b

    # plot mean curve
    roc_ax.plot(roc_a_base, mean_roc_b, label="mean auc="+"{0:0.3%}".format(mean_area_under_curve))
    roc_ax.fill_between(roc_a_base, lower_bound, upper_bound, color="grey", 
            alpha=0.3, label=r'$\pm$ 1 std dev')
    roc_ax.set_xlabel("mean false positive rate")
    roc_ax.set_ylabel("mean true positive rate")
    roc_ax.legend(loc=4)

    cmd = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=labels())
    cmd.plot(ax=cm_ax)




if len(sys.argv) == 2 and sys.argv[1] == "preprocess":
    df_train = pd.read_csv("./dota2Train.csv")
    df_test = pd.read_csv("./dota2Test.csv")

    df_train = dota.preprocess_df(df_train)
    df_test = dota.preprocess_df(df_test)

    df_train.to_csv("./preprocessed_dota2_train.csv", index=False)
    df_test.to_csv("./preprocessed_dota2_test.csv", index=False)
    df_train.append(df_test).to_csv("./preprocessed_dota2_all.csv", index=False)
    exit()
elif len(sys.argv) == 3 and sys.argv[1] == "evaluate":
    heroes = sys.argv[2].split(',')
    print(heroes)
    if len(heroes) != 10:
        print("Error in formatting, 10 heroes not present in example")
        exit(1)

    t1_heroes = heroes[:5]
    t2_heroes = heroes[5:]

    attrs = list(map(dota.hero_attr.get, heroes))

    if None in attrs:
        print("Error, incorrect hero names at indicies:", 
                np.where(np.array(attrs) == None)[0].tolist())
        exit(1)

    t1_attr = attrs[:5]
    t2_attr = attrs[5:]

    t1_attr_str = t1_attr.count("STR")
    t1_attr_agi = t1_attr.count("AGI")
    t1_attr_int = t1_attr.count("INT")

    t2_attr_str = t2_attr.count("STR")
    t2_attr_agi = t2_attr.count("AGI")
    t2_attr_int = t2_attr.count("INT")

    t1_id = list(map(dota.hero_to_id.get, t1_heroes))
    t2_id = list(map(dota.hero_to_id.get, t2_heroes))

    df_all = pd.read_csv("./preprocessed_dota2_all.csv")
    # ignore target column
    features = np.zeros(df_all.shape[1] - 1)
    i = 0
    for i in t1_id:
        features[i - 1] = 1
    for i in t2_id:
        features[i - 1] = 2

    hero_offset = len(dota.id_to_hero)
    features[hero_offset + 0] = t1_attr_str
    features[hero_offset + 1] = t1_attr_agi
    features[hero_offset + 2] = t1_attr_int

    features[hero_offset + 3] = t2_attr_str
    features[hero_offset + 4] = t2_attr_agi
    features[hero_offset + 5] = t2_attr_int

    x, y = split_x_y(df_all)
    # NB performed best, it is therefore used
    nb = CategoricalNB()
    nb = nb.fit(x, y)

    res = nb.predict(features.reshape(1, -1))

    print(res)

    exit()
elif len(sys.argv) > 1:
    print("Unrecognized argument, use 'preprocess' flag to emit new datasheet")
    exit(1)

df_all = pd.read_csv("./preprocessed_dota2_all.csv")
print(df_all)
exit()

x, y = split_x_y(df_all)

nb_model = CategoricalNB()
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()

_, nb_ax_1 = plt.subplots()
_, nb_ax_2 = plt.subplots()
_, rf_ax_1 = plt.subplots()
_, rf_ax_2 = plt.subplots()
_, knn_ax_1 = plt.subplots()
_, knn_ax_2 = plt.subplots()

nb_ax_1.set_title("Naive Bayes ROC curve")
nb_ax_2.set_title("Naive Bayes Confusion Matrix")
rf_ax_1.set_title("Random Forest ROC curve")
rf_ax_2.set_title("Random Forest Confusion Matrix")
knn_ax_1.set_title("KNN ROC Curve")
knn_ax_2.set_title("KNN Confusion Matrix")

nb_result = eval_model_using_cv(nb_model, x, y, nb_ax_1)
rf_result = eval_model_using_cv(rf_model, x, y, rf_ax_1)
knn_result = eval_model_using_cv(knn_model, x, y, knn_ax_1)


plot_results(nb_result, nb_ax_1, nb_ax_2)
plot_results(rf_result, rf_ax_1, rf_ax_2)
plot_results(knn_result, knn_ax_1, knn_ax_2)

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
