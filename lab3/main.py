#!/usr/bin/python

print("Starting...")

import numpy as np
import pandas as pd
import tensorflow
import json

from tensorflow.keras import layers

def create_id_hero_mapping(jsondict):
    mapping = dict()
    inner = jsondict["heroes"]
    for i in inner:
        mapping[i["id"]] = i["localized_name"]
    return mapping

train_df = pd.read_csv("./dota2Train.csv")
test_df = pd.read_csv("./dota2Test.csv")

with open("./heroes.json") as f:
    data = json.load(f)

id_to_hero = create_id_hero_mapping(data)

cols = ["W/L", "Cluster ID", "Mode", "Type"]

delta = len(train_df.columns) -len(cols)
for col in range(1, delta + 1):
    try:
        my_hero = id_to_hero[col]
        cols.append(my_hero)
    except KeyError:
        cols.append("Unused")
train_df.columns = cols
del train_df["Unused"]
print(train_df)

# shuffle training data
train_df = train_df.reindex(np.random.permutation(train_df.index))

print("Done!")
