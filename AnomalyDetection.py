import ProcessData as data
import VertexWeight as graph
import JointLearning as learn
import numpy as np

'''
CAN CHANGE
'''
# classify abnormal data into K_ABNORMAL classes
K_ABNORMAL = 5

# K_HYPER_GRAPH nearest neighbours are chosen to be connected by hyper edge
K_HYPER_GRAPH = 5

# LAMBDA value used in hyper learning
LAMBDA = 2

# MU value used in hyper learning
MU = 0.5

# ETA value
ETA = 0.5

# LEARNING_RATE of U optimization
LEARNING_RATE = 0.3

# whether tag abnormal data
USE_ABNORMAL = 1

# whether tag normal data
USE_NORMAL = 0

# whether use joint learning
USE_JOINT_LEARNING = 0

'''
CANNOT CHANGE
'''
UNKNOWN_TAG = 0.5
ABNORMAL_TAG = 1
NORMAL_TAG = 0


def init_detection_tag(sample_num, sample_abnormal_index, sample_normal_index, abnormal_result):
    """
    Set initial data tag, default set to UNKNOWN_TAG

    :param sample_num: sample num
    :param sample_abnormal_index: abnormal sample index
    :param sample_normal_index: normal sample index
    :param abnormal_result: abnormal point
    :return: Y: initial tag
    """
    Y = np.ones((sample_num, K_ABNORMAL)) - (ABNORMAL_TAG + NORMAL_TAG) / 2.0
    if USE_ABNORMAL > 0:
        for i in range(sample_abnormal_index[0].shape[0]):
            Y[sample_abnormal_index[0][i]] = NORMAL_TAG
            Y[sample_abnormal_index[0][i], abnormal_result[i]] = ABNORMAL_TAG
    if USE_NORMAL > 0:
        for i in range(sample_normal_index[0].shape[0]):
            Y[sample_normal_index[0][i]] = NORMAL_TAG
    return Y


def init_hyper_graph(sample_data, abnormal_data, abnormal_centers):
    """
    Init hyper graph with vertex weight

    :param sample_data: sample
    :param abnormal_data: abnormal data
    :param abnormal_centers: abnormal cluster center
    :return: U, Du, De, H, W of graph
    """
    sample_num = sample_data.shape[0]
    U = graph.init_vertex_weight(sample_data, abnormal_centers,
                                 np.mean(graph.calculate_vertex_score(abnormal_data, abnormal_centers, ETA)))
    W = np.zeros((sample_num, 1)) + 0.5
    H, Du, De = graph.init_hyper_graph(sample_data, K_HYPER_GRAPH, W, U)
    W_d = np.zeros((sample_num, sample_num))
    U_d = np.zeros((sample_num, sample_num))
    np.fill_diagonal(W_d, W)
    np.fill_diagonal(U_d, U)
    W = W_d
    U = U_d
    return U, Du, De, H, W


def measure_result(F, origin_data):
    """
    Result measurement
    :param F:  final tag
    :param origin_data:  origin data with tag
    :return: precision
    """
    sample_num = origin_data.shape[0]
    F[F > UNKNOWN_TAG] = ABNORMAL_TAG
    F[F <= UNKNOWN_TAG] = NORMAL_TAG
    F = np.sum(F, axis=1)
    F[F > UNKNOWN_TAG] = ABNORMAL_TAG
    true_tag = origin_data[:, 0]
    print sum(F == true_tag) * 1.0 / sample_num


def anomaly_detection(path):
    """
    Anomaly detection process.

    :return:
    """
    origin_data, sample_data, sample_abnormal_index, sample_normal_index, abnormal \
        = data.preprocess(path, sample_bound=3000, abnormal_bound=800, normal_bound=800)
    sample_num = sample_data.shape[0]

    centers, result = graph.classify_abnormal_data(abnormal, K_ABNORMAL)
    Y = init_detection_tag(sample_num, sample_abnormal_index, sample_normal_index, result)

    U, Du, De, H, W = init_hyper_graph(sample_data, abnormal, centers)

    F = learn.joint_learning(LAMBDA, LEARNING_RATE, U, Y, Du, De, H, W, MU)

    measure_result(F, origin_data)


if __name__ == '__main__':
    anomaly_detection("data/s7_1.txt")
