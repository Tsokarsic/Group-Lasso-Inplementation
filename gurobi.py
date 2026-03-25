import gurobipy as gp
import numpy as np

from util import run_method
from gurobipy import GRB
from numpy.typing import NDArray
from typing import Tuple, Dict, Optional


def gl_gurobi(x0: NDArray,
              A: NDArray,
              b: NDArray,
              mu: float,
              opts: Optional[Dict] = {}) -> Tuple[NDArray, int, Dict]:
    m, n = A.shape
    _, l = b.shape

    model = gp.Model("LASSO")
    x = [
        [model.addVar(name=f"x{i}{j}", lb=-GRB.INFINITY) for j in range(l)]
        for i in range(n)
    ]
    t = model.addVar(name="t", lb=0)
    s = [model.addVar(name=f"s{i}", lb=0) for i in range(n)]


    for i in range(n):
        for j in range(l):
            x[i][j].start = x0[i, j]


    if not opts.get("log", False):
        model.setParam("OutputFlag", False)

    #定义变量约束
    temp_list = []
    for i in range(m):
        for j in range(l):
            temp = model.addVar(lb=-GRB.INFINITY, name=f"temp{i}{j}")
            temp_list.append(temp)
            model.addConstr(temp == sum([A[i, k] * x[k][j] for k in range(n)]) - b[i, j])

    quad_expr = gp.QuadExpr()
    quad_expr.addTerms([-1], [t], [t])
    quad_expr.addTerms([1] * m * l, temp_list, temp_list)
    model.addQConstr(quad_expr, sense=GRB.LESS_EQUAL, rhs=0, name="Ax-b")


    for i in range(n):
        quad_expr = gp.QuadExpr()
        quad_expr.addTerms([1] * l, x[i], x[i])
        quad_expr.addTerms(-1, s[i], s[i])
        model.addQConstr(quad_expr, sense=GRB.LESS_EQUAL, rhs=0, name=f"x{i}")

    # 目标函数
    model.setObjective(0.5 * t + mu * gp.quicksum(s), GRB.MINIMIZE)

    model.optimize()


    result = [[model.getVarByName(f"x{i}{j}").X for j in range(l)] for i in range(n)]
    result = np.array(result)

    obj_val = model.ObjVal

    return result, model.BarIterCount, {"fval":obj_val,"obj": obj_val}


if __name__ == "__main__":
    run_method(gl_gurobi, plot=False)
