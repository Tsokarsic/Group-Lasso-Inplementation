# 使用多种算法求解Group Lasso问题

#对于梯度算法采取的连续化策略，我们参考了代码仓库：https://github.com/AkexStar/Algorithms-group-LASSO-problem
，但代码的具体实现方式为独立完成。

基于python版本：3.11.1

## 运行所需环境：
- NumPy 1.26.1
- Mosek 10.1.29
- Gurobi 12.0.0
- cvxpy 1.7.3

##运行代码：运行python group_lasso.py即可返回各个算法的性能指标

##函数输出的out字典包含的额外信息：
对于基于梯度的方法，我们在优化过程中，通过out字典来输出如下信息：
    Fval：算法结束时的目标函数值
    obj_val: 记录算法迭代过程中目标函数值的变化的列表
    grad_norm: 记录算法迭代过程中梯度的Frobenius范数的变化的列表
对于基于对偶问题的方法，我们在优化过程中，通过out字典来输出如下信息：
    Fval：算法结束时的目标函数值
    obj_val: 记录算法迭代过程中目标函数值的变化的列表
    dual_gap: 记录算法迭代过程中对偶间隙的变化的列表
对于其他方法，我们在优化过程中，通过out字典来输出如下信息：
    Fval：算法结束时的目标函数值
    obj_val: 记录算法迭代过程中目标函数值的变化的列表
