# coding=utf-8

from scipy.spatial import KDTree
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans


def init_hyper_edge(sample, k):
    """
    Init hyper edge

    :param sample:  n * d array of samples
    :param k: K nearest neighbours are chosen to e connected by hyperedge
    :return: hyperedge, and corresponding distance
    """
    kd_tree = KDTree(sample)
    hyper_edge = []
    hyper_edge_distance = []
    for item in sample:
        k_neighbour = kd_tree.query(item, k)
        hyper_edge.append(k_neighbour[1])
        hyper_edge_distance.append(k_neighbour[0])
    hyper_edge = np.array(hyper_edge)
    hyper_edge_distance = np.array(hyper_edge_distance)
    return hyper_edge, hyper_edge_distance


def euclidean_distances(A, B):
    """
    Euclidean distance between matrix A and B

    :param A: np array
    :param B: np array
    :return: np array
    """
    BT = B.transpose()
    vec_prod = np.dot(A, BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vec_prod.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vec_prod.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vec_prod
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def calculate_H(sample, k, alpha=1):
    """
    an incidence matrix H is used to describe the relation between vertices and hyperedges

    :param sample:  n * d array of samples
    :param k: K nearest neighbours are chosen to e connected by hyperedge
    :param alpha:
    :return: H
    """
    distances = euclidean_distances(sample, sample)
    mean_d = np.mean(distances[distances.nonzero()])
    num = sample.shape[0]
    H = np.zeros((num, num))
    edge, edge_distance = init_hyper_edge(sample, k)
    for i in range(num):
        d_i = edge_distance[i, :].transpose()
        H[edge[i, :], i] = np.exp(- d_i * d_i * 1.0 / (alpha * mean_d * mean_d))
    return H


def classify_abnormal_data(abnormal_data, k):
    """
    employ cluster method to observed samples and divide them into k clusters

    :param abnormal_data: observed abnormal data
    :param k: divide into k clusters
    :return: centers
    """
    k_means = KMeans(n_clusters=k)
    k_means.fit(abnormal_data)
    return k_means.cluster_centers_


def init_vertex_weight(sample_data, centers, gamma):
    """
    The samples with TS > gamma are set as potential anomalies,
    and gamma is the average score of observed anomalies.
    We set weight according to the total score and the potential label.

    :param sample_data: sample data, n * d array of samples
    :param centers: the center of observed abnormal data cluster
    :param gamma: average score of observed anomalies
    :return: hypergraph vertex weight
    """
    TS = calculate_vertex_score(sample_data, centers, gamma)
    U = np.zeros((TS.shape[0], 1))
    abnormal_index = np.nonzero(TS > gamma)[0]
    normal_index = np.nonzero(TS <= gamma)[0]
    U[abnormal_index] = TS[abnormal_index] / np.max(TS)
    U[normal_index] = (np.max(TS) - TS[normal_index]) / (np.max(TS) - np.min(TS))
    return U


def calculate_vertex_score(samples, center, eta):
    """
    we use similarity score and isolation score to initialize vertex weight
    according to their correlations

    :param samples: all the samples
    :param eta: total score = ηIS(O) + (1 − η)SS(O)
    :return: total score of samples
    """
    clf = IsolationForest()
    clf.fit(samples)
    num = samples.shape[0]
    IS = (0.5 - clf.decision_function(samples)).reshape((num, 1))
    distance = np.min(euclidean_distances(samples, center), axis=1)
    SS = np.exp(-distance).reshape((num, 1))
    TS = eta * IS + (1 - eta) * SS
    return TS


if __name__ == '__main__':
    x, y = np.mgrid[0:5, 2:8]
    sample_data = np.array(zip(x.ravel(), y.ravel()))
    x, y = np.mgrid[0:2, 2:4]
    abnormal = np.array(zip(x.ravel(), y.ravel()))

    centers = classify_abnormal_data(abnormal, 4)
    ts_abnormal = calculate_vertex_score(abnormal, centers, 0.5)
    abnormal_mean = np.mean(ts_abnormal)
    print init_vertex_weight(sample_data, centers, abnormal_mean)
