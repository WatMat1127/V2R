import numpy as np
import copy
import math
np.set_printoptions(linewidth=10000)

def calc_f_val_grad(qm_ene_list, grad_list):
    dE_list = [(i - qm_ene_list[0]) * 627.51 for i in qm_ene_list]
    dE = np.array(dE_list[1:])

    ddE_dp_list = [(i - grad_list[0]) * 627.51 for i in grad_list]
    ddE_dp = np.array(ddE_dp_list[1:])

    mean_std = np.loadtxt('mean_std.csv', delimiter=',', skiprows=1, encoding='utf-8-sig')
    mean = mean_std.T[0]
    std = mean_std.T[1]

    U_real = np.loadtxt('U_real.csv', delimiter=',', encoding='utf-8-sig')
    X_real = np.loadtxt('X_real.csv', delimiter=',', encoding='utf-8-sig')

    dE_scaled = (dE - mean) / (std)
    X_virt = dE_scaled @ U_real.T
    f_val = np.linalg.norm(X_virt - X_real) ** 2

    ddE_dp_scaled = ddE_dp / np.tile(std.reshape(-1,1), ddE_dp.shape[1])
    dX_virt_dp = (ddE_dp_scaled.T @ U_real.T).T

    f_grad = np.zeros(ddE_dp.shape[1])
    for i in range(dX_virt_dp.shape[0]):
        f_grad += 2 * (X_virt[i] - X_real[i]) * dX_virt_dp[i]


    return f_val, f_grad
