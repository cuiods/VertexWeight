import numpy as np
import pandas as pd


def calculate_F(lambda_value, U, Y, Du, De, H, W):
    theta_u = calculate_theta_u(Du, H, W, De)
    F = lambda_value * (np.linalg.inv(U.T - U.T.dot(theta_u).dot(U) + (lambda_value*U.T).dot(U)))\
        .dot(U.T).dot(U).dot(Y)
    return F


def calculate_U(lambda_value, F, Y, Du, De, H, W, mu):
    theta_u = calculate_theta_u(Du, H, W, De)
    num = H.shape[0]
    I = np.eye(num)
    U = np.linalg.inv(4*lambda_value*Y.dot(F.T)-2*lambda_value*Y.dot(Y.T)-2*lambda_value*F.dot(F.T))\
        .dot(F.dot(F.T) + 2*theta_u.dot(F).dot(F.T) + mu * I)
    return U


def calculate_cost(lambda_value, F, U,  Y, Du, De, H, W, mu):
    theta_u = calculate_theta_u(Du, H, W, De)
    cost = np.trace(F.T.dot(U.T-U.T.dot(theta_u).dot(U)).dot(F)) + \
           lambda_value * np.trace(F.T.dot(U.T).dot(U).dot(F) + Y.T.dot(U.T).dot(U).dot(Y) - 2*F.T.dot(U.T).dot(U).dot(Y))+\
           mu * np.trace(U)
    return cost


def calculate_theta_u(Du, H, W, De):
    theta_u = np.sqrt(reciprocal(Du)).dot(H).dot(W) \
        .dot(reciprocal(De)).dot(H.T).dot(np.sqrt(reciprocal(Du)))
    dia = np.diag(np.fliplr(theta_u))
    theta_num = theta_u.shape[0]
    theta = np.zeros((theta_num, theta_num))
    np.fill_diagonal(theta, dia)
    return theta


def reciprocal(matrix):
    c_matrix = np.reciprocal(matrix)
    c_matrix[np.abs(c_matrix) == np.inf] = 0
    c_matrix = pd.DataFrame(c_matrix).fillna(0).values
    return c_matrix


def joint_learning(lambda_value, U, Y, Du, De, H, W, mu):
    F = calculate_F(lambda_value, U, Y, Du, De, H, W)
    cost = calculate_cost(lambda_value, F, U, Y, Du, De, H, W, mu)
    print cost
    U = calculate_U(lambda_value, F, Y, Du, De, H, W, mu)
    cost = calculate_cost(lambda_value, F, U,  Y, Du, De, H, W, mu)
    print cost
    while True:
        F = calculate_F(lambda_value, U, Y, Du, De, H, W)
        U = calculate_U(lambda_value, F, Y, Du, De, H, W, mu)
        current_cost = calculate_cost(lambda_value, F, U,  Y, Du, De, H, W, mu)
        print current_cost
        if cost - current_cost < 0.01:
            break
        cost = current_cost
    return F
