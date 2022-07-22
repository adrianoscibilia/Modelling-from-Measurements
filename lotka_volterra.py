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
import pysindy as ps
from pysindy.feature_library import CustomLibrary

def rhs_lotkavolterra(t, x0, b, p, d, r):
    x, y = x0
    rhs = [(b - p * y) * x,
           (r * x - d) * y]
    return rhs

def bagging(X: np.array, p: int, time: np.array):
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
            tp[i] = float(time[column_sel])
    return Xp, tp

# Build data vectors from file
file_name = "Population_data.csv"
input_data_tab = pd.read_csv(filepath_or_buffer=file_name, sep="\t")
year = input_data_tab["Year"]
SH = input_data_tab["Snowshoe_Hare"]
CL = input_data_tab["Canada_Lynx"]
data = np.vstack((SH, CL))
xdata = np.array(data)
time = np.array(year[:] - year[0])
dT = int(time[1] - time[0])

test_size = int(len(time) * 0.2)
x1 = SH[:-test_size]
x2 = CL[:-test_size]
x1_test = SH[test_size:]
x2_test = CL[test_size:]
x_train = np.vstack((x1, x2))

# Generation of state derivative vectors
n = len(time) - test_size
x1dot = [0] * (n - 2)
x2dot = [0] * (n - 2)

# center difference scheme
for i in range(1, n - 1):
    x1dot[i - 1] = (x1[i + 1] - x1[i - 1]) / (2 * dT)
    x2dot[i - 1] = (x2[i + 1] - x2[i - 1]) / (2 * dT)

x1s = np.array(x1[1:-1])
x2s = np.array(x2[1:-1])

A1 = np.array([x1s, -x1s * x2s])
A2 = np.array([-x2s, x1s * x2s])

model1 = LinearRegression()
model1.fit(A1.T, x1dot)
model2 = LinearRegression()
model2.fit(A2.T, x2dot)

load1 = np.linalg.pinv(A1.T) @ x1dot
load2 = np.linalg.pinv(A2.T )@ x2dot

# Lotka-Volterra parameters estimate of regression with standard least square
b_l, p_l, d_l, r_l = model1.coef_[0], model1.coef_[1], model2.coef_[0], model2.coef_[1]
b_inv, p_inv, d_inv, r_inv = load1[0], load1[1], load2[0], load2[1]

# Inegration of system dynamics for the test times with estimate parameters
x0 = [x1s[-1], x2s[-1]]
#t_test = np.arange(time[test_size], len(time))
t_test = time[n:]
sol_tmp = integrate.solve_ivp(rhs_lotkavolterra, [time[test_size], time[len(time)-1]], x0, args=(b_inv, p_inv, d_inv, r_inv), method = 'RK45', t_eval=t_test)
sol_test = np.vstack((sol_tmp['y'][0], sol_tmp['y'][1])).T

# Integration of the system dynamics along all the time axis
x0 = [x1[0], x2[0]]
sol_tmp = integrate.solve_ivp(rhs_lotkavolterra, [time[0], time[len(time)-1]], x0, args=(b_inv, p_inv, d_inv, r_inv), method = 'RK45', t_eval=time)
sol = np.vstack((sol_tmp['y'][0], sol_tmp['y'][1])).T
x_true = np.vstack((SH, CL)).T
r2 = sklearn.metrics.r2_score(x_true, sol)


# PLOTS
plt.figure(1)
plt.plot(time, sol[:, 0])
plt.plot(time, sol[:, 1])
plt.xlabel('time')
plt.ylabel('population')
plt.title("Lotka-Volterra Approximation")
plt.grid()


# SINDY
n_of_iterations = 500
BAG = True
feature_names = ['x','y','x^2','y^2','xy','1/x','1/y','1/x^2','1/y^2','x^3','y^3','x^4','y^4']
t_train = time[:n]
t_train.sort()
ensemble_coeffs_list = []
ensemble_optimizer = ps.STLSQ(threshold=0.00005, alpha=0.5, max_iter=2000)
model3 = ps.SINDy(feature_names=feature_names, optimizer=ensemble_optimizer)

if not BAG: 
    for idx in range(0, n_of_iterations):
        model3.fit(x=xdata.transpose(), t=dT, ensemble=True, replace=False, quiet=True)
        ensemble_coeffs_list.append(model3.coef_list)
    ensemble_coeffs = np.asarray(ensemble_coeffs_list)
    mean_ensemble_coefs = np.mean(ensemble_coeffs, axis=0)
    Mmean_ensemble_coefs = np.mean(mean_ensemble_coefs, axis=0)
    ensemble_optimizer.coef_ = Mmean_ensemble_coefs
    x_sim_mean = model3.simulate(xdata[:, 0], time)


# BAGGING SINDY
if BAG:
    for idx in range(0, n_of_iterations):
        x_train_bag, time_bag = bagging(X=xdata, p=10, time=time)
        model3.fit(x=x_train_bag.transpose(), t=dT, ensemble=True, replace=False, quiet=True)
        coeff_tmp = np.asarray(model3.coef_list)
        ensemble_coeffs_list.append(coeff_tmp)
    ensemble_coeffs = np.asarray(ensemble_coeffs_list)
    mean_ensemble_coefs = np.mean(ensemble_coeffs, axis=0)
    Mmean_ensemble_coefs = np.mean(mean_ensemble_coefs, axis=0)
    ensemble_optimizer.coef_ = Mmean_ensemble_coefs
    x_sim_mean = model3.simulate(xdata[:, 0], time)

lib_functions = model3.get_feature_names()
model3.print()

plt.figure(4)
plt.plot(time, x_sim_mean[:, 0], '-.', color='r')
plt.plot(time, sol[:, 0], color='r')
plt.plot(time, x_sim_mean[:, 1], '-.', color='b')
plt.plot(time, sol[:, 1], color='b')
plt.xlabel('time')
plt.ylabel('population')
plt.title("SINDy simulated response")
plt.grid()

plt.show()
