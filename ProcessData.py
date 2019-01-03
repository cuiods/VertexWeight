import pandas as pd
import re
import numpy as np
import json
import codecs


def read_data(normal_path, tag_path):
    normal = pd.read_csv(normal_path, header=None)
    abnormal = pd.read_csv(tag_path, header=None)
    normal = filter_data(normal)
    abnormal = filter_data(abnormal)
    return normal.values, abnormal.values


def read_single_data(path):
    normal = pd.read_csv(path, header=None)
    normal = filter_data(normal)
    return normal.values


def filter_data(data_frame):
    data_frame[data_frame.columns[len(data_frame.columns) - 2]] = \
        data_frame[data_frame.columns[len(data_frame.columns) - 2]].apply(lambda x: int("".join(re.findall(r'\d+', x))))
    data_frame[data_frame.columns[len(data_frame.columns) - 1]] = \
        data_frame[data_frame.columns[len(data_frame.columns) - 1]].apply(lambda x: int(x, 16))
    return data_frame


def generate_skyline_data(normal_path,index):
    normal = read_single_data(normal_path)
    time_index = 1
    timestamp1 = normal[:, time_index]
    data = normal[:, index]
    json.dump({"results": np.vstack((timestamp1, data)).T.tolist()},
              codecs.open("data"+str(index)+".json", 'w', encoding='utf-8'))


if __name__ == "__main__":
    generate_skyline_data('data/s7_1.txt', 2)