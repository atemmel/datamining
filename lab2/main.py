#!/bin/python

# nya frankie
def frankie(target, names):
    return np.array(list(map(lambda x: np.where(names == x)[0][0], target)))

from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

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

filedata = arff.loadarff('./breast-cancer.arff')

df = pd.DataFrame(filedata[0])

les = build_labelencoders(df)
res = []
for i in range(0, len(df.columns)):
    col = df.iloc[:, i].values
    res.append(les[i].transform(col))
res = pd.DataFrame(res)

#print(res.transpose())

inverse_res = []
for i in range(0, len(df.columns)):
    col = res.iloc[i, :].values
    #print(col)
    inverse_res.append(les[i].inverse_transform(col))
inverse_res = pd.DataFrame(inverse_res)

#print(inverse_res.transpose())

res = res.transpose()
data = res.iloc[:,:-1]
target = res.iloc[:,-1:]

#print(data)
#print(target)

#le = LabelEncoder()

#data = df.iloc[:,:-1]
#feature_names = df.Class.unique()
#le.fit(feature_names)
#target = df.iloc[:,-1:].Class
#target = le.transform(target.values)

#print(list(le.classes_))
#print()
#transformed = le.transform(df.iloc[:,-1].values[0:3])
#print(transformed)

#print(feature_names)
#print(target)

#index = 1
#test = extract_column(df, 1)
#print(test)
#print()
#print(list(le.inverse_transform(target)))

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(data, target)

#feature_names = df[df.columns[-1]].unique()
#feature_names = extract_column(df, -1).unique()
feature_names = list(df.columns[:-1])
class_names = list(df.columns[-1:])
#print(feature_names)

#r = export_text(decision_tree, feature_names=feature_names)
r = export_text(decision_tree)
print(r)
