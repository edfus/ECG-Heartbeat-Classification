import numpy as np
import pandas as pd

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file)
    fields_length = len(df.columns)
    is_test = fields_length <= 2
    heartbeats_length = 0
    normalized_list = []
    for items in df.values:
        entry = [items[0]] + [float(i) for i in items[1].split(",")]
        if heartbeats_length == 0:
          heartbeats_length = len(entry) - 1
        elif heartbeats_length != len(entry) - 1:
          raise ValueError(
            "Heartbeats provided are of inconstant length ({}, {})".format(
              heartbeats_length, len(entry) - 1
            )
          )
        normalized_list.append(
          entry if is_test else entry + [items[2]]
        )
    
    df = reduce_mem_usage(pd.DataFrame(np.array(normalized_list)))
    column_labels = ["id"] + [
        "heartbeat_sample_" + str(i) for i in range(heartbeats_length)
    ]
    df.columns = column_labels if is_test else column_labels + ["label"]
    return df


# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")
    return df
