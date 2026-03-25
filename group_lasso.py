import argparse
from typing import Optional, Dict
import code
from code import *
import json
from code.ADMM_dual import gl_ADMM_dual
from code.ADMM_primal import gl_ADMM_primal
from code.ALM_dual import gl_ALM_dual
from code.FastProximalGD import gl_FProxGD_primal
from code.GD import gl_GD_primal
from code.ProximalGD import gl_ProxGD_primal
from code.SubGradient import gl_SGD_primal
from code.cvx_gurobi import gl_cvx_gurobi
from code.cvx_mosek import gl_cvx_mosek
from code.gurobi import gl_gurobi
from code.origin_mosek import gl_mosek

import numpy as np
import time
import matplotlib.pyplot as plt
import cvxpy as cp
import mosek
import gurobipy as gp
from gurobipy import GRB


def errfun(x1, x2):
    """计算两个解的相对误差：||x1-x2||_F / (1+||x1||_F)"""
    norm_x1 = np.linalg.norm(x1, 'fro')
    norm_diff = np.linalg.norm(x1 - x2, 'fro')
    return norm_diff / (1 + norm_x1)


def errfun_exact(x):
    """计算解与真实解u的相对误差：||x-u||_F / (1+||u||_F)"""
    global u
    norm_u = np.linalg.norm(u, 'fro')
    norm_diff = np.linalg.norm(x - u, 'fro')
    return norm_diff / (1 + norm_u)


def sparisity(x):
    """计算解的稀疏性：非零元素占比（阈值1e-6*max(abs(x))）"""
    x_flat = x.flatten()
    threshold = 1e-6 * np.max(np.abs(x_flat))
    non_zero = np.sum(np.abs(x_flat) > threshold)
    return non_zero / len(x_flat)


# ===================== 2. 定义绘图函数（对应MATLAB的plot_results） =====================
def plot_results(x, title, save_path, u_true, x_cvx_mosek, x_cvx_gurobi):
    import os
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n, l = x.shape
    plt.figure(figsize=(12, 6))
    # 绘制第一列
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n + 1), u_true[:, 0], '*', label='Exact u (col1)')
    plt.plot(range(1, n + 1), x[:, 0], 'o', label=f'{title} (col1)')
    plt.plot(range(1, n + 1), x_cvx_mosek[:, 0], 'x', label='CVX-Mosek (col1)')
    plt.plot(range(1, n + 1), x_cvx_gurobi[:, 0], '+', label='CVX-Gurobi (col1)')
    plt.xlim([1, n])
    plt.legend()
    plt.title(f'{title} - Column 1')

    # 绘制第二列
    plt.subplot(2, 1, 2)
    plt.plot(range(1, n + 1), u_true[:, 1], '*', label='Exact u (col2)')
    plt.plot(range(1, n + 1), x[:, 1], 'o', label=f'{title} (col2)')
    plt.plot(range(1, n + 1), x_cvx_mosek[:, 1], 'x', label='CVX-Mosek (col2)')
    plt.plot(range(1, n + 1), x_cvx_gurobi[:, 1], '+', label='CVX-Gurobi (col2)')
    plt.xlim([1, n])
    plt.legend()
    plt.title(f'{title} - Column 2')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#生成数据
np.random.seed(97006855)
n = 512
m = 256
k = round(n * 0.1)  # 稀疏度：10%非零元素
l = 2
A = np.random.randn(m, n)

# 生成稀疏解
p = np.random.permutation(n)[:k]
u = np.zeros((n, l))
u[p, :] = np.random.randn(k, l)
b = A @ u
mu = 1e-2
x0 = np.random.randn(n, l)
opts1 = {}
start = time.time()
x1, iter1, out1 = gl_cvx_mosek(x0, A, b, mu, opts1)
t1 = time.time() - start
opts2 = {}
start = time.time()
x2, iter2, out2 = gl_cvx_gurobi(x0, A, b, mu, opts2)
t2 = time.time() - start
opts3 = {}
start = time.time()
x3, iter3, out3 = gl_mosek(x0, A, b, mu, opts3)
t3 = time.time() - start
opts4 = {}
start = time.time()
x4, iter4, out4 = gl_gurobi(x0, A, b, mu, opts4)
t4 = time.time() - start


opts5 = {}
start = time.time()
x5, iter5, out5 = gl_SGD_primal(x0, A, b, mu, opts5)
t5 = time.time() - start

# GD Primal
opts6 = {}
start = time.time()
x6, iter6, out6 = gl_GD_primal(x0, A, b, mu, opts6)
t6 = time.time() - start

# # FGD Primal
# opts7 = {}
# start = time.time()
# x7, iter7, out7 = gl_FGD_primal(x0, A, b, mu, opts7)
# t7 = time.time() - start
#
# # ProxGD Primal
opts8 = {}
start = time.time()
x8, iter8, out8 = gl_ProxGD_primal(x0, A, b, mu, opts8)
t8 = time.time() - start

# FProxGD Primal
opts9 = {}
start = time.time()
x9, iter9, out9 = gl_FProxGD_primal(x0, A, b, mu, opts9)
t9 = time.time() - start
#
# ALM Dual
opts10 = {}
start = time.time()
x10, iter10, out10 = gl_ALM_dual(x0, A, b, mu, opts10)
t10 = time.time() - start

# ADMM Dual
opts11 = {}
start = time.time()
x11, iter11, out11 = gl_ADMM_dual(x0, A, b, mu, opts11)
t11 = time.time() - start

# ADMM Primal
opts12 = {}
start = time.time()
x12, iter12, out12 = gl_ADMM_primal(x0, A, b, mu, opts12)
t12 = time.time() - start
#
# # PPA Dual
# opts13 = {}
# start = time.time()
# x13, iter13, out13 = gl_PPA_dual(x0, A, b, mu, opts13)
# t13 = time.time() - start
#
# # BCD Primal
# opts14 = {}
# start = time.time()
# x14, iter14, out14 = gl_BCD_primal(x0, A, b, mu, opts14)
# t14 = time.time() - start

# ===================== 5. 打印对比结果（复刻MATLAB的fprintf） =====================
print(
    f'     CVX-Mosek: cpu: {t1:5.2f}, iter: {iter1}, optval: {out1["fval"]:6.5E}, sparisity: {sparisity(x1):4.3f}, err-to-exact: {errfun_exact(x1):3.2E}, err-to-cvx-mosek: {errfun(x1, x1):3.2E}, err-to-cvx-gurobi: {errfun(x2, x1):3.2E}.')
print(
    f'    CVX-Gurobi: cpu: {t2:5.2f}, iter: {iter2}, optval: {out2["fval"]:6.5E}, sparisity: {sparisity(x2):4.3f}, err-to-exact: {errfun_exact(x2):3.2E}, err-to-cvx-mosek: {errfun(x1, x2):3.2E}, err-to-cvx-gurobi: {errfun(x2, x2):3.2E}.')
print(
    f'         Mosek: cpu: {t3:5.2f}, iter: {iter3}, optval: {out3["fval"]:6.5E}, sparisity: {sparisity(x3):4.3f}, err-to-exact: {errfun_exact(x3):3.2E}, err-to-cvx-mosek: {errfun(x1, x3):3.2E}, err-to-cvx-gurobi: {errfun(x2, x3):3.2E}.')
print(
    f'        Gurobi: cpu: {t4:5.2f}, iter: {iter4}, optval: {out4["fval"]:6.5E}, sparisity: {sparisity(x4):4.3f}, err-to-exact: {errfun_exact(x4):3.2E}, err-to-cvx-mosek: {errfun(x1, x4):3.2E}, err-to-cvx-gurobi: {errfun(x2, x4):3.2E}.')
print(
    f'    SGD Primal: cpu: {t5:5.2f}, iter: {iter5:5d}, optval: {out5["fval"]:6.5E}, sparisity: {sparisity(x5):4.3f}, err-to-exact: {errfun_exact(x5):3.2E}, err-to-cvx-mosek: {errfun(x1, x5):3.2E}, err-to-cvx-gurobi: {errfun(x2, x5):3.2E}.')
print(
    f'     GD Primal: cpu: {t6:5.2f}, iter: {iter6:5d}, optval: {out6["fval"]:6.5E}, sparisity: {sparisity(x6):4.3f}, err-to-exact: {errfun_exact(x6):3.2E}, err-to-cvx-mosek: {errfun(x1, x6):3.2E}, err-to-cvx-gurobi: {errfun(x2, x6):3.2E}.')
# print(
#     f'    FGD Primal: cpu: {t7:5.2f}, iter: {iter7:5d}, optval: {out7["fval"]:6.5E}, sparisity: {sparisity(x7):4.3f}, err-to-exact: {errfun_exact(x7):3.2E}, err-to-cvx-mosek: {errfun(x1, x7):3.2E}, err-to-cvx-gurobi: {errfun(x2, x7):3.2E}.')
print(
    f' ProxGD Primal: cpu: {t8:5.2f}, iter: {iter8:5d}, optval: {out8["fval"]:6.5E}, sparisity: {sparisity(x8):4.3f}, err-to-exact: {errfun_exact(x8):3.2E}, err-to-cvx-mosek: {errfun(x1, x8):3.2E}, err-to-cvx-gurobi: {errfun(x2, x8):3.2E}.')
print(
    f'FProxGD Primal: cpu: {t9:5.2f}, iter: {iter9:5d}, optval: {out9["fval"]:6.5E}, sparisity: {sparisity(x9):4.3f}, err-to-exact: {errfun_exact(x9):3.2E}, err-to-cvx-mosek: {errfun(x1, x9):3.2E}, err-to-cvx-gurobi: {errfun(x2, x9):3.2E}.')
print(
    f'      ALM Dual: cpu: {t10:5.2f}, iter: {iter10:5d}, optval: {out10["fval"]:6.5E}, sparisity: {sparisity(x10):4.3f}, err-to-exact: {errfun_exact(x10):3.2E}, err-to-cvx-mosek: {errfun(x1, x10):3.2E}, err-to-cvx-gurobi: {errfun(x2, x10):3.2E}.')
print(
    f'     ADMM Dual: cpu: {t11:5.2f}, iter: {iter11:5d}, optval: {out11["fval"]:6.5E}, sparisity: {sparisity(x11):4.3f}, err-to-exact: {errfun_exact(x11):3.2E}, err-to-cvx-mosek: {errfun(x1, x11):3.2E}, err-to-cvx-gurobi: {errfun(x2, x11):3.2E}.')
print(
    f'   ADMM Primal: cpu: {t12:5.2f}, iter: {iter12:5d}, optval: {out12["fval"]:6.5E}, sparisity: {sparisity(x12):4.3f}, err-to-exact: {errfun_exact(x12):3.2E}, err-to-cvx-mosek: {errfun(x1, x12):3.2E}, err-to-cvx-gurobi: {errfun(x2, x12):3.2E}.')
# print(
#     f'      PPA dual: cpu: {t13:5.2f}, iter: {iter13:5d}, optval: {out13["fval"]:6.5E}, sparisity: {sparisity(x13):4.3f}, err-to-exact: {errfun_exact(x13):3.2E}, err-to-cvx-mosek: {errfun(x1, x13):3.2E}, err-to-cvx-gurobi: {errfun(x2, x13):3.2E}.')
# print(
#     f'    BCD primal: cpu: {t14:5.2f}, iter: {iter14:5d}, optval: {out14["fval"]:6.5E}, sparisity: {sparisity(x14):4.3f}, err-to-exact: {errfun_exact(x14):3.2E}, err-to-cvx-mosek: {errfun(x1, x14):3.2E}, err-to-cvx-gurobi: {errfun(x2, x14):3.2E}.')

# 绘图
# plot_results(u, 'Exact', 'figures/gl_exact.png', u, x1, x2)
# plot_results(x1, 'CVX-Mosek', 'figures/gl_cvx_mosek.png', u, x1, x2)
# plot_results(x2, 'CVX-Gurobi', 'figures/gl_cvx_gurobi.png', u, x1, x2)
# plot_results(x3, 'Mosek', 'figures/gl_mosek.png', u, x1, x2)
# plot_results(x4, 'Gurobi', 'figures/gl_gurobi.png', u, x1, x2)
# plot_results(x5, 'SGD Primal', 'figures/gl_SGD_Primal.png', u, x1, x2)
# plot_results(x6, 'GD Primal', 'figures/gl_GD_primal.png', u, x1, x2)
# # plot_results(x7, 'FGD Primal', '../figures/gl_FGD_primal.png', u, x1, x2)
# plot_results(x8, 'ProxGD Primal', 'figures/gl_ProxGD_primal.png', u, x1, x2)
# plot_results(x9, 'FProxGD Primal', 'figures/gl_FProxGD_primal.png', u, x1, x2)
# plot_results(x10, 'ALM Dual', 'figures/gl_ALM_dual.png', u, x1, x2)
# plot_results(x11, 'ADMM Dual', 'figures/gl_ADMM_dual.png', u, x1, x2)
# plot_results(x12, 'ADMM Primal', 'figures/gl_ADMM_primal.png', u, x1, x2)
# plot_results(x13, 'PPA Dual', '../figures/gl_PPA_dual.png', u, x1, x2)
# plot_results(x14, 'BCD Primal', '../figures/BCD_primal.png', u, x1, x2)