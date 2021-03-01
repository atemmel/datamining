import pandas as pd
import numpy as np
import json

id_to_hero = dict()
hero_to_id = dict()
hero_attr = dict()

def init(path_to_mapping, path_to_attributes):
    global id_to_hero, hero_to_id, hero_attr
    id_to_hero, hero_to_id = create_id_hero_mapping_from_file(path_to_mapping)
    hero_attr = create_hero_attr_mapping_from_file(path_to_attributes)

# this function performs the entire pipeline
def preprocess_df(df):
    global id_to_hero, hero_attr

    df = insert_labels(df, id_to_hero)
    df = cull_dataframe(df)

    df = append_attributes(df)
    df = convert_team2(df)
    return df

def columns():
    return ["W/L", "Cluster ID", "Mode", "Type"]

# naive bayes classifier cannot work with negative numbers,
# (it throws a ValueError) we therefore convert team -1 to 
# team 2
def convert_team2(df):
    return df.replace(to_replace=-1, value=2)

def create_id_hero_mapping_from_file(path):
    with open(path) as f:
        jsondict = json.load(f)

    id_name = dict()
    name_id = dict()

    inner = jsondict["heroes"]
    for i in inner:
        id_name[i["id"]] = i["localized_name"]
        name_id[i["localized_name"]] = i["id"]
    return id_name, name_id

def create_hero_attr_mapping_from_file(path):
    df_attributes = pd.read_csv(path)
    name_attr = dict()

    for index, row in df_attributes.iterrows():
        name_attr[row["Hero Name"]] = row["Primary Stat"]
    return name_attr

def cull_dataframe(df):
    return pd.DataFrame(df.iloc[:, 0]).join(df.iloc[:, 4:])

def insert_labels(df, id_to_hero):
    cols = columns()
    delta = len(df.columns) -len(cols)
    for col in range(1, delta + 1):
        try:
            my_hero = id_to_hero[col]
            cols.append(my_hero)
        except KeyError:
            cols.append("Unused")
    df.columns = cols
    del df["Unused"]
    return df

def append_attributes(df_games):
    global hero_attr

    df_attributes = pd.DataFrame(np.zeros((len(df_games.index), 3 * 2)))

    for index, row in df_games.iloc[:, 1:].iterrows():
        t1_str = 0
        t1_agi = 0
        t1_int = 0

        t2_str = 0
        t2_agi = 0
        t2_int = 0

        for col in row.iteritems():
            if col[1] == 0:
                continue
            attr = hero_attr[col[0]]

            if col[1] == 1:
                if attr == "STR":
                    t1_str += 1
                elif attr == "AGI":
                    t1_agi += 1
                elif attr == "INT":
                    t1_int += 1
            elif col[1] == -1:
                if attr == "STR":
                    t2_str += 1
                elif attr == "AGI":
                    t2_agi += 1
                elif attr == "INT":
                    t2_int += 1

        arr = np.array([t1_str, t1_agi, t1_int, t2_str, t2_agi, t2_int])

        df_attributes.iloc[index, :] = arr

    df_attributes.columns = ["1 STR", "1 AGI", "1 INT", "2 STR", "2 AGI", "2 INT"]
    
    return df_games.join(df_attributes)
