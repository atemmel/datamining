#!/bin/python

from six import StringIO
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from IPython.display import Image
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import pydotplus
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# nya frankie
def frankie(target, names):
    return np.array(list(map(lambda x: np.where(names == x)[0][0], target)))

def extract_column(df, index):
    return df[df.columns[index]]

def build_labelencoders(df):
    arr = []
    for i in range(0, len(df.columns)):
        le = LabelEncoder()
        col = extract_column(df, i).unique()
        le.fit(col)
        arr.append(le)
    return arr

def build_onehotencoders(df):
    ohe1 = OneHotEncoder()
    ohe1.fit(df.iloc[:, :-1])
    ohe2 = OneHotEncoder()
    ohe2.fit(df.iloc[:, -1:])
    #print(ohe.transform(df)[:5, :])
    return ohe1, ohe2

def create_tree_using_labelencoders(df):
    les = build_labelencoders(df)
    res = []
    for i in range(0, len(df.columns)):
        col = df.iloc[:, i].values
        res.append(les[i].transform(col))
    res = pd.DataFrame(res)

    #print(res.transpose())
    #inverse_res = []
    #for i in range(0, len(df.columns)):
        #col = res.iloc[i, :].values
        #inverse_res.append(les[i].inverse_transform(col))
    #inverse_res = pd.DataFrame(inverse_res)
    #print(inverse_res.transpose())
    res = res.transpose()
    data = res.iloc[:,:-1]
    target = res.iloc[:,-1:]
    feature_names = list(df.columns[:-1])
    class_names = list(extract_column(df, -1).unique())
    return data, target, feature_names, class_names

def create_tree_using_onehotencoders(df):
    oheData, oheTarget = build_onehotencoders(df)
    res = pd.DataFrame(oheData.transform(df.iloc[:, :-1]).toarray())
    data = res.iloc[:, :-1]
    res = pd.DataFrame(oheTarget.transform(df.iloc[:, -1:]).toarray())
    target = res.iloc[:,-1:]
    feature_names = oheData.get_feature_names(list(df.columns[:-1]))[:-1]
    class_names = list(extract_column(df, -1).unique())
    return data, target, feature_names, class_names

def perform_bayes(df):
    les = build_labelencoders(df)
    res = []
    for i in range(len(df.columns)):
        col = df.iloc[:, i].values
        res.append(les[i].transform(col))
    res = pd.DataFrame(res).transpose()
    x = res.iloc[:, :-1]
    y = res.iloc[:, -1:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

    model = CategoricalNB()
    model.fit(x_train, y_train.values.ravel())

    y_pred = model.predict(x_test)
    y_pred_probability = model.predict_proba(x_test)[::, 1]

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(accuracy)

    # example patient
    test = ['50-59','ge40','50-54','24-26','no','1','right','left_up','yes']
    print(test)

    # transform using labelencoders
    for i in range(len(test)):
        e = test[i]
        test[i] = les[i].transform(np.array(e).reshape(1, ))
    test = np.array(test)

    # do prediction
    y = model.predict(test.reshape(1, -1))

    # translate back
    y = les[-1].inverse_transform(y)[0]
    print(y)

    a, b, _ = roc_curve(y_test, y_pred_probability)
    plt.plot(a, b, label="accuracy="+str(accuracy))
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.axis
    plt.legend(loc=4)
    plt.show()

filedata = arff.loadarff('./breast-cancer.arff')

df = pd.DataFrame(filedata[0])
for i in range(0, len(df.columns)):
    title = list(df.columns)[i]
    df[title] = df[title].apply(lambda s: s.decode("utf-8"))

#data, target, feature_names, class_names = create_tree_using_labelencoders(df)
data, target, feature_names, class_names = create_tree_using_onehotencoders(df)

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.25, random_state=10)

decision_tree = DecisionTreeClassifier(random_state=0, criterion="entropy")
path = decision_tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(x_train, y_train) for clf in clfs]
test_scores = [clf.score(x_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()

plt.show()

decision_tree = decision_tree.fit(x_train, y_train)

#r = export_text(decision_tree, feature_names)
#print(r)

dot_data = StringIO()

export_graphviz(decision_tree, 
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=feature_names,
        class_names=class_names)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("big_tree.png")
Image(graph.create_png())

perform_bayes(df)
