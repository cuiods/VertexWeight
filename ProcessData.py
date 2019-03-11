import pandas as pd
import re
import numpy as np


def read_single_data(path):
    """
    Read data in ndarray type
    :param path: path of data file
    :return:  data: ndarray
    """
    normal = pd.read_csv(path, header=None)
    normal = filter_data(normal)
    return normal.values


def read_origin_data(path):
    """
    Read data in data frame type
    :param path:  path of data file
    :return: data: ndarray
    """
    normal = pd.read_csv(path, header=None)
    # normal = filter_data(normal)
    return normal


def filter_data(data_frame):
    """
    Process string value
    :param data_frame: raw data in data frame type
    :return: data frame
    """
    data_frame[data_frame.columns[len(data_frame.columns) - 2]] = \
        data_frame[data_frame.columns[len(data_frame.columns) - 2]].apply(lambda x: int("".join(re.findall(r'\d+', x))))
    data_frame[data_frame.columns[len(data_frame.columns) - 1]] = \
        data_frame[data_frame.columns[len(data_frame.columns) - 1]].apply(lambda x: int(x, 16))
    return data_frame


def preprocess(path, sample_bound=-1, abnormal_bound=-1, normal_bound=-1, abnormal_rate = 0.5):
    """
    Complete data preprocess.
    :param path:  data path
    :param sample_bound:  max sample num
    :param abnormal_bound:  max abnormal tag num
    :param normal_bound:  max normal tag num
    :return: origin_data, sample_data, abnormal_index, normal_index, abnormal_data
    """
    origin_data = read_origin_data(path)
    normal_origin_data = origin_data[origin_data[0] == 0]
    abnormal_origin_data = origin_data[origin_data[0] == 1]
    if sample_bound > 0:
        abnormal_sample_data = abnormal_origin_data.sample(n=int(sample_bound*abnormal_rate))
        normal_sample_data = normal_origin_data.sample(n=int(sample_bound*(1-abnormal_rate)))
        origin_data = pd.concat([abnormal_sample_data, normal_sample_data])
    origin_data = origin_data.values

    # define specific sample data
    sample_data = origin_data[:, 2:]
    sample_data = (sample_data - np.min(sample_data, axis=0)) / (np.max(sample_data, axis=0)-np.min(sample_data, axis=0))
    sample_data = np.nan_to_num(sample_data)
    abnormal_index = np.array(np.nonzero(origin_data[:, 0] == 1))
    normal_index = np.array(np.nonzero(origin_data[:, 0] == 0))

    if (abnormal_bound > 0) and (abnormal_bound < abnormal_index.shape[1]):
        idx = np.random.randint(abnormal_index.shape[1], size=abnormal_bound)
        abnormal_index = abnormal_index[:, idx]
    if (normal_bound > 0) and (normal_bound < normal_index.shape[1]):
        idx = np.random.randint(normal_index.shape[1], size=normal_bound)
        normal_index = normal_index[:, idx]

    abnormal = sample_data[abnormal_index[0]]

    return origin_data, sample_data, abnormal_index, normal_index, abnormal
