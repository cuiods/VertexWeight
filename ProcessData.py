import pandas as pd
import re
import numpy as np
import json
import codecs
import socket
import time
import msgpack


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
    print data.shape
    json.dump({"results": np.vstack((timestamp1, data)).T.tolist()},
              codecs.open("data"+str(index)+"_tag.json", 'w', encoding='utf-8'))


def transfer_skyline_data(path, index, ip, port):
    normal = read_single_data(path)
    data = normal[:, index]
    num = data.shape[0]
    timestamp = normal[:, index]
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = (ip, port)
    start_time = float(int(time.time() - num))
    print "Analysis start at " + str(start_time)
    map_relation = []
    metric = "power.test.index" + str(index)
    for i in range(num):
        data_udp = [start_time, data[i]]
        packet = msgpack.packb((metric, data_udp))
        client_socket.sendto(packet, address)
        map_relation.append([timestamp[i], start_time])
        start_time = start_time + 1
    json.dump({"relation": map_relation},
              codecs.open("time_relation_" + str(index) + ".json", 'w', encoding='utf-8'))


if __name__ == "__main__":
    # generate_skyline_data('data/s7_1.txt', 2)
    transfer_skyline_data('data/s7.txt', 2, '127.0.0.1', 2025)
