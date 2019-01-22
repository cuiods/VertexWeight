import ProcessData as data
import VertexWeight as graph
import JointLearning as learn
import numpy as np

'''
CAN CHANGE
'''
# classify abnormal data into K_ABNORMAL classes
K_ABNORMAL = 7

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

SAMPLE_BOUND = 5000
ABNORMAL_BOUND = 25
NORMAL_BOUND = 1250

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
    anomaly_mean_score = np.mean(graph.calculate_vertex_score(abnormal_data, abnormal_centers, ETA))
    U, abnormal_index, normal_index = graph.init_vertex_weight(sample_data, abnormal_centers, ETA, anomaly_mean_score)

    W = np.zeros((sample_num, 1)) + 0.5
    H, Du, De = graph.init_hyper_graph(sample_data, K_HYPER_GRAPH, W, U)
    W_d = np.zeros((sample_num, sample_num))
    U_d = np.zeros((sample_num, sample_num))
    np.fill_diagonal(W_d, W)
    np.fill_diagonal(U_d, U)
    W = W_d
    U = U_d
    return U, Du, De, H, W, abnormal_index, normal_index


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
    print 'Total accuracy:'+str(sum(F == true_tag) * 1.0 / sample_num)
    positive_data_index = np.nonzero(F > 0)[0].tolist()
    true_data_index = np.nonzero(true_tag > 0)[0].tolist()
    tp = set(positive_data_index).intersection(set(true_data_index))
    abnormal_precision = len(tp)*1.0 / len(positive_data_index)
    abnormal_recall = len(tp)*1.0 / len(true_data_index)
    f_value = 2.0 * abnormal_precision * abnormal_recall / (abnormal_precision + abnormal_recall)
    print 'Abnormal precision:'+str(abnormal_precision)
    print 'Abnormal recall:' + str(abnormal_recall)
    print 'F value:' + str(f_value)


def anomaly_detection(path):
    """
    Anomaly detection process.

    :return:
    """
    origin_data, sample_data, sample_abnormal_index, sample_normal_index, abnormal \
        = data.preprocess(path, sample_bound=SAMPLE_BOUND, abnormal_bound=ABNORMAL_BOUND, normal_bound=NORMAL_BOUND, abnormal_rate=0.5)
    sample_num = sample_data.shape[0]

    centers, result = graph.classify_abnormal_data(abnormal, K_ABNORMAL)
    Y = init_detection_tag(sample_num, sample_abnormal_index, sample_normal_index, result)

    U, Du, De, H, W, first_abnormal, first_normal = init_hyper_graph(sample_data, abnormal, centers)
    true_tag = origin_data[:, 0]
    true_data_index = np.nonzero(true_tag > 0)[0].tolist()
    false_data_index = np.nonzero(true_tag <= 0)[0].tolist()
    tp = set(first_abnormal).intersection(set(true_data_index))
    tn = set(first_normal).intersection(set(false_data_index))
    print len(tp)
    print len(tn)
    accuracy = (len(tp)+len(tn)) * 1.0 / true_tag.shape[0]
    abnormal_precision = len(tp) * 1.0 / len(first_abnormal)
    abnormal_recall = len(tp) * 1.0 / len(true_data_index)
    f_value = 2.0 * abnormal_precision * abnormal_recall / (abnormal_precision + abnormal_recall)
    print 'Accuracy:' + str(accuracy)
    print 'First Abnormal precision:' + str(abnormal_precision)
    print 'First Abnormal recall:' + str(abnormal_recall)
    print 'First F value:' + str(f_value)

    # F = learn.joint_learning(LAMBDA, LEARNING_RATE, U, Y, Du, De, H, W, MU, joint=USE_JOINT_LEARNING)
    #
    # measure_result(F, origin_data)


if __name__ == '__main__':
    anomaly_detection("data/s7_1.txt")
