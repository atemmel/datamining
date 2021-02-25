import pandas as pd

def create_id_hero_mapping(jsondict):
    mapping = dict()
    inner = jsondict["heroes"]
    for i in inner:
        mapping[i["id"]] = i["name"]
    return mapping

def cull_dataframe(df):
    return pd.DataFrame(df.iloc[:, 0]).join(df.iloc[:, 4:])

def insert_labels(df, id_to_hero):
    cols = ["WL", "Cluster ID", "Mode", "Type"]
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

def preprocess_df(df, id_to_hero):
    df = insert_labels(df, id_to_hero)
    df = cull_dataframe(df)
    return df

