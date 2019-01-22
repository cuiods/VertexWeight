import numpy as np
import pandas as pd


def calculate_F(lambda_value, U, Y, Du, De, H, W):
    """
    Calculate optimal F directly by U

    :return: current optimal F value
    """
    theta_u = calculate_theta_u(Du, H, W, De)
    F = lambda_value * (np.linalg.inv(U.T - U.T.dot(theta_u).dot(U) + (lambda_value*U.T).dot(U)))\
        .dot(U.T).dot(U).dot(Y)
    return F


def calculate_U(lambda_value, F, Y, Du, De, H, W, mu):
    """
    Calculate optimal U directly by F

    :return:  current optimal U
    """
    theta_u = calculate_theta_u(Du, H, W, De)
    num = H.shape[0]
    I = np.eye(num)
    U = np.linalg.inv(4*lambda_value*Y.dot(F.T)-2*lambda_value*Y.dot(Y.T)-2*lambda_value*F.dot(F.T))\
        .dot(F.dot(F.T) + 2*theta_u.dot(F).dot(F.T) + mu * I)
    return U


def calculate_U_gradient(lambda_value, l_rate, F, U, Y, Du, De, H, W, mu):
    """
    Training U by gradient descent

    :return:
    """
    temp = U - l_rate * gradient_des_step(lambda_value, F, U, Y, Du, De, H, W, mu)
    if np.sum(temp < 0) > 0:
        return U
    return temp


def gradient_des_step(lambda_value, F, U, Y, Du, De, H, W, mu):
    """
    Calculate gradient step delta[COST]/delta[U]

    :return:  delta[COST]/delta[U]
    """
    theta_u = calculate_theta_u(Du, H, W, De)
    step = F.dot(F.T) + 2*theta_u.dot(F).dot(F.T) + 2*lambda_value*U.dot(F).dot(F.T) + \
           2*lambda_value*U.dot(Y).dot(Y.T) - 4*lambda_value*U.dot(Y).dot(F.T) + mu
    dia = np.diag(np.fliplr(step))
    step_num = theta_u.shape[0]
    step_diag = np.zeros((step_num, step_num))
    np.fill_diagonal(step_diag, dia)
    return step_diag


def calculate_cost(lambda_value, F, U,  Y, Du, De, H, W, mu):
    """
    Calculate joint cost of U and F (Loss function)

    :return: cost of U and F
    """
    theta_u = calculate_theta_u(Du, H, W, De)
    cost = np.trace(F.T.dot(U.T-U.T.dot(theta_u).dot(U)).dot(F)) + \
           lambda_value * np.trace(F.T.dot(U.T).dot(U).dot(F) + Y.T.dot(U.T).dot(U).dot(Y) - 2*F.T.dot(U.T).dot(U).dot(Y))+\
           mu * np.trace(U)
    return cost


def calculate_theta_u(Du, H, W, De):
    """
    Middle value calculation

    :return:
    """
    theta_u = np.sqrt(reciprocal(Du)).dot(H).dot(W) \
        .dot(reciprocal(De)).dot(H.T).dot(np.sqrt(reciprocal(Du)))
    dia = np.diag(np.fliplr(theta_u))
    theta_num = theta_u.shape[0]
    theta = np.zeros((theta_num, theta_num))
    np.fill_diagonal(theta, dia)
    return theta


def reciprocal(matrix):
    """
    Calculate 1/n of a matrix
    notice: 1/0 is set to zero

    :param matrix:
    :return: reciprocal of the matrix
    """
    c_matrix = np.reciprocal(matrix)
    c_matrix[np.abs(c_matrix) == np.inf] = 0
    c_matrix = pd.DataFrame(c_matrix).fillna(0).values
    return c_matrix


def joint_learning(lambda_value, learning_rate, U, Y, Du, De, H, W, mu, joint=False):
    """
    Joint Learning on Vertex Relevance and Vertex Weights

    :param lambda_value: lambda
    :param learning_rate: learning rate
    :param U: vertex weight
    :param Y: observed tag
    :param Du: degree of vertex
    :param De: degree of edge
    :param H: vertex relevance
    :param W: edge weight
    :param mu: mu
    :return: final F value
    """
    F = calculate_F(lambda_value, U, Y, Du, De, H, W)
    cost = calculate_cost(lambda_value, F, U, Y, Du, De, H, W, mu)
    if joint:
        print cost
        U = calculate_U_gradient(lambda_value, learning_rate, F, U, Y, Du, De, H, W, mu)
        cost = calculate_cost(lambda_value, F, U, Y, Du, De, H, W, mu)
        print cost
        while True:
            F = calculate_F(lambda_value, U, Y, Du, De, H, W)
            U = calculate_U_gradient(lambda_value, learning_rate, F, U, Y, Du, De, H, W, mu)
            current_cost = calculate_cost(lambda_value, F, U, Y, Du, De, H, W, mu)
            print current_cost
            if cost - current_cost < 1:
                break
            cost = current_cost
    return F
