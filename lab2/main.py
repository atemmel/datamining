#!/bin/python

from six import StringIO
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pydotplus
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
    #print(feature_names)
    #class_names = list(map(lambda x: x.decode("utf8"), list(extract_column(df, -1).unique())))
    class_names = list(extract_column(df, -1).unique())
    return data, target, feature_names, class_names


filedata = arff.loadarff('./breast-cancer.arff')

df = pd.DataFrame(filedata[0])
for i in range(0, len(df.columns)):
    title = list(df.columns)[i]
    df[title] = df[title].apply(lambda s: s.decode("utf-8"))

#data, target, feature_names, class_names = create_tree_using_labelencoders(df)
data, target, feature_names, class_names = create_tree_using_onehotencoders(df)

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=4)
#decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(data, target)

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
