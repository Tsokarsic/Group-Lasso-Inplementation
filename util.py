import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Callable, Optional


def generate_data(seed: int = 97006855) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    #调用run method的生成数据的接口
    np.random.seed(seed)
    n = 512
    m = 256
    A = np.random.randn(m, n)
    k = round(n * 0.1)
    l = 2
    p = np.random.permutation(n)
    p = p[0: k]
    u = np.zeros((n, l))
    u[p, :] = np.random.randn(k, l)
    b = A @ u
    x0 = np.random.randn(n, l)
    return A, b, x0, u


def group_lasso_loss(A: NDArray, b: NDArray, x: NDArray, mu: float):
    """
    计算损失函数
    """
    return 0.5 * np.linalg.norm(A @ x - b, "fro") ** 2 + mu * np.sum(np.linalg.norm(x, axis=1))


def extract_config(opt: Dict, key: str, default=None):
    return default if opt is None or key not in opt else opt[key]


def bench_mark(func: Callable, opts) -> float:
    import time

    A, b, x0, u = generate_data()
    mu = 1e-2
    for _ in range(50):
        func(x0, A, b, mu, opts)
    start = time.time()
    for _ in range(50):
        func(x0, A, b, mu, opts)
    end = time.time()
    return (end - start) * 20


def sparisity(x: NDArray) -> float:
    elem_num = 1
    for i in x.shape:
        elem_num *= i
    max_elem = np.max(np.abs(x))
    return np.sum(np.abs(x) > (1e-6 * max_elem)) / elem_num


def run_method(func: Callable, plot: bool = True, log_scale: bool = True, benchmark: bool = False,
               opts: Optional[Dict] = {}, seed: int = 97006855, output: bool = True, **kwargs) -> NDArray:
    #本函数用于多次运行一个优化器返回平均运行时间和打印相关信息
    import matplotlib.pyplot as plt
    import os
    img_dir = "./images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    func_name = func.__name__
    save_path = os.path.join(img_dir, f"{func_name}.png")

    A, b, x0, u = generate_data(seed)
    mu = 1e-2
    if "log" not in opts.keys():
        opts["log"] = False
    if benchmark:
        opts["benchmark"] = True
    x, iter_count, out = func(x0, A, b, mu, opts)
    if output:
        print(f"Objective value: {group_lasso_loss(A, b, x, mu):.8f}")
        gurobi_ans = kwargs.get('gurobi_ans', None)
        mosek_ans = kwargs.get('mosek_ans', None)
        if mosek_ans is not None:
            print(f"Error mosek: {np.linalg.norm(x - mosek_ans, 'fro') / (1 + np.linalg.norm(mosek_ans, 'fro')):.6e}")
        if gurobi_ans is not None:
            print(
                f"Error gurobi: {np.linalg.norm(x - gurobi_ans, 'fro') / (1 + np.linalg.norm(gurobi_ans, 'fro')):.6e}")
        print(f"Error exact: {np.linalg.norm(x - u, 'fro') / (1 + np.linalg.norm(u, 'fro')):.6e}")
        print(f"Sparsity: {sparisity(x):.4f}")
        print(f"Iteration count: {iter_count}")

    if plot:
        plt.figure(figsize=(4, 4))
        losses = out['obj_val']
        ax = plt.subplot(111)
        ax.plot(np.arange(len(losses)), losses)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title("Objective value")
        # data = out['grad_norm'] if 'grad_norm' in out else out['dual_gap']
        # ax = plt.subplot(122)
        # ax.plot(np.arange(len(data)), data)
        # if log_scale:
        #     ax.set_yscale('log')
        # ax.set_title("Gradient norm" if 'grad_norm' in out else "Dual gap")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"绘图已保存至：{save_path}")
        plt.show()
        plt.close()

    if benchmark:
        #计算平均时间
        avg_time = bench_mark(func, opts)
        print(f"Benchmark: {avg_time:.4f} ms")

    return x
