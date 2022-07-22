#!/usr/bin/env python3
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.linalg import hankel
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from scipy import integrate
import pandas as pd
from matplotlib import rc
from optimalDMD import optdmd


def DMD(X, Xprime, r):
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)  # Step 1
    Ur = U[:, :r]
    Sigmar = np.diag(Sigma[:r])
    VTr = VT[:r, :]
    Atilde = np.linalg.solve(Sigmar.T, (Ur.T @ Xprime @ VTr.T).T).T  # Step 2
    Lambda, W = np.linalg.eig(Atilde)  # Step 3
    Lambda = np.diag(Lambda)
    Phi = Xprime @ np.linalg.solve(Sigmar.T, VTr).T @ W  # Step 4
    alpha1 = Sigmar @ VTr[:, 0]
    b = np.linalg.solve(W @ Lambda, alpha1)
    return Phi, Lambda, b


def BOP_DMD(p: int, X: np.array, time: np.array, r: int, e_init):
    Xp = np.zeros((2, p))
    tp = np.zeros(p)
    X_rows, X_cols = X.shape
    choices = np.zeros(p)

    for i in range(0, p):
        column_sel = np.random.choice(X_cols)
        if column_sel in choices:
            column_sel = np.random.choice(X_cols)
        else:
            choices[i] = column_sel
            Xp[0, i] = X[0, column_sel]
            Xp[1, i] = X[1, column_sel]
            tp[i] = time[column_sel]
    Xp.sort()
    tp.sort()
    w, e, b = optdmd(X=Xp, t=tp, r=r, imode=1, e_init=e_init)
    return w, e, b


def DEL_DMD(X: np.array, r: int):
    n_emb = 5
    #x_rows, x_cols = X.shape
    max_samples = 25
    H = X[:, :max_samples]
    dH = X[:, 1:max_samples + 1]
    for i in range(1, n_emb):
        H = np.vstack((H, X[:, i:max_samples + i]))
        dH = np.vstack((dH, X[:, i + 1:max_samples + i + 1]))
    Phi, Lambda, b = DMD(H, dH, r)
    return Phi, Lambda, b, H,max_samples


def DEL_BOP_DMD(p: int, n_emb: int, X: np.array, time: np.array, r: int, e_init):
    max_samples = 25
    H = X[:, :max_samples]
    for i in range(1, n_emb):
        H = np.vstack((H, X[:, i:max_samples + i]))
    H_rows, H_cols = H.shape
    Hp = np.zeros((H_rows, p))
    tp = np.zeros(p)
    choices = np.zeros(p)

    for i in range(0, p):
        column_sel = np.random.choice(H_cols)
        if column_sel in choices:
            column_sel = np.random.choice(H_cols)
        else:
            choices[i] = column_sel
            Hp[:, i] = H[:, column_sel]
            tp[i] = time[column_sel]

    Hp.sort()
    tp.sort()
    w, e, b = optdmd(X=Hp, t=tp, r=r, imode=1, e_init=e_init)
    return w, e, b


# Build data vectors from file
file_name = "Population_data.csv"
input_data_tab = pd.read_csv(filepath_or_buffer=file_name, sep="\t")
year = input_data_tab["Year"]
SH = input_data_tab["Snowshoe_Hare"]
CL = input_data_tab["Canada_Lynx"]
data = np.vstack((SH, CL))
xdata = np.array(data)
X = np.array(data[:, :-1])
Xprime = np.array(data[:, 1:])
rank = 2
time = np.asarray(year[:-1] - year[0])
time_idx = np.arange(0, 29)
dT = time[1] - time[0]

DELAY = False    # Set True for Delay DMDs, False for others


if not DELAY:
    # EXACT DMD
    Phi, Lambda, b = DMD(X=X, Xprime=Xprime, r=rank)
    omega = np.log(Lambda) / dT
    omega = np.diag(omega)
    x_dmd = np.zeros((rank, len(time)), dtype=omega.dtype)

    for k in range(0, len(time)):
        x_dmd[:, k] = b * np.exp(omega * time[k])

    x_dmd = np.dot(Phi, x_dmd)
    SH_dmd = np.real(x_dmd[0, :])
    CL_dmd = np.real(x_dmd[1, :])

    # Initial eigenvalues guess from optdmd run
    w_0, e_0, b_0 = optdmd(X=X, t=time, r=rank, imode=0, e_init=0)

    K_iter = 200  # choose number of iterations
    x_bop = np.zeros((rank, len(time_idx)))
    x_bop_idx = np.zeros((rank, len(time_idx)))
    x_bop_mean = np.zeros((rank, len(time_idx)))
    w_list = []
    e_list = []
    b_list = []
    x_bop_list = []

    for idx in range(0, K_iter):
        w_idx, e_idx, b_idx = BOP_DMD(p=20, X=X, time=time, r=rank, e_init=e_0)
        w_list.append(w_idx)
        e_list.append(e_idx)
        b_list.append(b_idx)

    w_vec = np.array(w_list)
    e_vec = np.array(e_list)
    b_vec = np.array(b_list)
    w_vec[::-1].sort(axis=0)
    e_vec[::-1].sort(axis=0)
    b_vec[::-1].sort(axis=0)
    w_mean = np.mean(w_vec, axis=0)
    e_mean = np.mean(e_vec, axis=0)
    b_mean = np.mean(b_vec, axis=0)
    for i in range(0, len(time_idx)):
        x_bop[:, i] = b_mean * np.exp(e_mean * time_idx[i])
    x_bop = np.dot(w_mean, x_bop)


if DELAY:
    # DELAY EXACT DMD
    n_emb = 5
    rank_del = n_emb * 2  # augment target rank for delay models
    Phi_del, Lambda_del, b_del, H, max_samples = DEL_DMD(X=xdata, r=rank_del)
    omega_del = np.log(Lambda_del) / dT
    omega_del = np.diag(omega_del)
    x_tmp = np.zeros((rank_del, len(time_idx)))
    x_del_dmd = np.zeros((2, len(time_idx)))
    K_iter_del = 100

    for k in range(0, len(time_idx)):
        x_tmp[:, k] = b_del * np.exp(omega_del * time_idx[k])
    x_tmp = np.dot(Phi_del, x_tmp)
    x_tmp_rows, x_tmp_cols = x_tmp.shape
    half_x_dmd_idx = int(x_tmp_rows/2)
    x_del_dmd[0, :] = np.sum(x_tmp[:half_x_dmd_idx, :], axis=0)
    x_del_dmd[1, :] = np.sum(x_tmp[half_x_dmd_idx:, :], axis=0)
    U_h, S_h, V_h = np.linalg.svd(H, full_matrices=False)


    # DELAY BOP DMD
    x_bop_del_mean = np.zeros((rank_del, len(time_idx)))
    x_bop_del = np.zeros((2, len(time_idx)))
    x_bop_del_list = []
    w_del_bop_list = []
    e_del_bop_list = []
    b_del_bop_list = []

    time_H = np.asarray(time[:max_samples])
    w_opt, e_opt, b_opt = optdmd(X=H, t=time_H, r=rank_del, imode=0, e_init=0)

    for idx in range(0, K_iter_del):
        w_del_bop_idx, e_del_bop_idx, b_del_bop_idx = DEL_BOP_DMD(p=20, n_emb=n_emb, X=xdata, time=time, r=rank_del, e_init=e_opt)
        w_del_bop_list.append(w_del_bop_idx)
        e_del_bop_list.append(e_del_bop_idx)
        b_del_bop_list.append(b_del_bop_idx)
    x_bop_del_vec = np.array(x_bop_del_list)
    w_del_bop_vec = np.array(w_del_bop_list)
    e_del_bop_vec = np.array(e_del_bop_list)
    b_del_bop_vec = np.array(b_del_bop_list)
    w_del_bop_vec[::-1].sort(axis=0)
    e_del_bop_vec[::-1].sort(axis=0)
    b_del_bop_vec[::-1].sort(axis=0)
    w_del_bop_mean = np.mean(w_del_bop_vec, axis=0)
    e_del_bop_mean = np.mean(e_del_bop_vec, axis=0)
    b_del_bop_mean = np.mean(b_del_bop_vec, axis=0)
    for i in range(0, len(time_idx)):
        x_bop_del_mean[:, i] = b_del_bop_mean * np.exp(e_del_bop_mean * time_idx[i])
    x_bop = np.dot(w_del_bop_mean, x_bop_del_mean)
    x_bop_del[0, :] = np.sum(x_bop_del_mean[:half_x_dmd_idx, :], axis=0)
    x_bop_del[1, :] = np.sum(x_bop_del_mean[half_x_dmd_idx:, :], axis=0)


# PLOTS
if not DELAY:
    plt.figure(1)
    plt.plot(time, SH_dmd)
    plt.plot(time, CL_dmd)
    plt.title('Exact DMD')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.grid()

    plt.figure(2)
    plt.plot(time, x_bop[0, :])
    plt.plot(time, x_bop[1, :])
    plt.title('Bagging Optimal DMD')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.grid()

if DELAY:
    plt.figure(4)
    plt.plot(time, x_del_dmd[0, :])
    plt.plot(time, x_del_dmd[1, :])
    plt.xlabel('time')
    plt.ylabel('population')
    plt.grid()

    plt.figure(5)
    plt.scatter(range(1, len(S_h)+1), S_h / np.sum(S_h))
    plt.xlabel('eigenvalues')
    plt.ylabel('energy')
    plt.grid()

    plt.figure(6)
    plt.plot(time, x_bop_del[0, :])
    plt.plot(time, x_bop_del[1, :])
    plt.xlabel('time')
    plt.ylabel('population')
    plt.grid()

plt.show()