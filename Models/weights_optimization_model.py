import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


def optimize_weights(
    excess: pd.DataFrame,
    deficit: pd.DataFrame,
    preference_df: pd.DataFrame,
    num_rounds: int = 5,
    write: bool = False
) -> tuple[dict[tuple[str,str], float], float, float]:
    """
    Runs a MILP to allocate supply to demand over multiple rounds for the full month.

    Args:
        excess: DataFrame (time x producers)
        deficit: DataFrame (time x consumers)
        preference_tensor:  pd.DataFrame (n_producers x n_consumers)
        num_rounds: number of allocation rounds
        write: write resulting model formulation

    Returns:
        optimal_weights: pd.DataFrame (n_producers x n_consumers)
        total_unmet_demand: float
        total_unused_supply: float
        model: gp.Model
    """

    producers = list(excess.columns)
    consumers = list(deficit.columns)
    supply = excess.values
    demand = deficit.values

    T = excess.shape[0]
    n_prod = len(producers)
    n_cons = len(consumers)

    # binary link matrix
    pref = preference_df.values
    links = (pref > 0).astype(int)

    model = gp.Model("MILP Energy Allocation")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.03
    model.Params.Heuristics = 0.25

    w = model.addVars(n_prod, n_cons, lb=0, ub=1, name="w")
    for i in range(n_prod):
        model.addConstr(quicksum(w[i, j] for j in range(n_cons)) == 1)
        for j in range(n_cons):
            model.addConstr(w[i,j] <= links[i, j])

    S = model.addVars(range(T), range(n_prod), range(num_rounds + 1), lb=0, name="S")
    D = model.addVars(range(T), range(n_cons), range(num_rounds + 1), lb=0, name="D")
    f = model.addVars(range(T), range(n_prod), range(n_cons), range(1, num_rounds + 1), lb=0, name="f")


    for t in range(T):
        for i in range(n_prod):
            model.addConstr(S[t, i, 0] == supply[t, i])
        for j in range(n_cons):
            model.addConstr(D[t, j, 0] == demand[t, j])

 
    for t in range(T):
        for r in range(1, num_rounds + 1):
            for i in range(n_prod):
                for j in range(n_cons):
                    model.addConstr(f[t, i, j, r] <= w[i, j] * S[t, i, r - 1] * links[i, j],
                        name=f"flow_sup_{t}_{i}_{j}_{r}")
                    model.addConstr(f[t, i, j, r] <= D[t, j, r - 1],
                        name=f"flow_dem_{t}_{i}_{j}_{r}")
            
            for i in range(n_prod):
                model.addConstr(S[t, i, r] == S[t, i, r - 1] - quicksum(f[t, i, j, r] for j in range(n_cons)),
                    name=f"sup_update_{t}_{i}_{r}")
                
            for j in range(n_cons):
                model.addConstr(D[t, j, r] == D[t, j, r - 1] - quicksum(f[t, i, j, r] for i in range(n_prod)),
                    name=f"dem_update_{t}_{j}_{r}")

    unused_obj = quicksum(S[t, i, num_rounds] for t in range(T) for i in range(n_prod))
    unmet_obj  = quicksum(D[t, j, num_rounds] for t in range(T) for j in range(n_cons))
    model.setObjective(unused_obj + unmet_obj, GRB.MINIMIZE)
    
    if write:
        model.write("ER_weight_opt.lp")

    model.optimize()

    if model.Status != 2:
        optimal_weights = {}
        for i in range(n_prod):
            prod_id = producers[i]
            optimal_weights[prod_id] = {}
            for j in range(n_cons):
                optimal_weights[prod_id][consumers[j]] = 0.0

        total_unmet = np.inf
        total_unused = np.inf

        return pd.DataFrame.from_dict(optimal_weights), total_unmet, total_unused


    optimal_weights = {}
    for i in range(n_prod):
        prod_id = producers[i]
        optimal_weights[prod_id] = {}
        for j in range(n_cons):
            optimal_weights[prod_id][consumers[j]] = w[i, j].X
    
    total_unused = sum(S[t, i, num_rounds].X for t in range(T) for i in range(n_prod))
    total_unmet = sum(D[t, j, num_rounds].X for t in range(T) for j in range(n_cons))

    return pd.DataFrame.from_dict(optimal_weights), total_unmet, total_unused
