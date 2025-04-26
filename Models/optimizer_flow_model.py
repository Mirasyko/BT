import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd


def optimize_flow_time_series(
    supply_over_time: pd.DataFrame,
    demand_over_time: pd.DataFrame,
    preference_tensor: np.ndarray,
    num_rounds: int = 2
) -> tuple[dict[tuple[int,int], float], float, float]:
    """
    Runs a nonconvex flow-allocation model across the entire time series to find
    static allocation weights that minimize total unmet demand + unused supply,
    using separate potential and actual flow variables like the original.

    Args:
        supply_over_time: array of shape (T, n_producers)
        demand_over_time: array of shape (T, n_consumers)
        preference_tensor: array of shape (n_producers, n_consumers, n_pref_levels)
        num_rounds: number of preference rounds per time step

    Returns:
        optimal_weights: dict[(i, j)] -> weight
        total_unmet_demand: float
        total_unused_supply: float
    """

    producers = list(supply_over_time.columns)
    consumers = list(demand_over_time.columns)
    demand_over_time = demand_over_time.values
    supply_over_time = supply_over_time.values

    T, n_prod = supply_over_time.shape
    _, n_cons = demand_over_time.shape
    _, _, n_pref = preference_tensor.shape
    round_pref = [(r, p) for r in range(num_rounds) for p in range(n_pref)]

    model = gp.Model("nonconvex_flow_timeseries")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.03
    model.Params.Heuristics = 0.25
    model.Params.NonConvex = 2


    w = model.addVars(n_prod, n_cons, lb=0, ub=1, name="w")
    for i in range(n_prod):
        model.addConstr(quicksum(w[i, j] for j in range(n_cons)) == 1)

    potential_flow = model.addVars(
        range(T), range(n_prod), range(n_cons), range(num_rounds), range(n_pref),
        lb=0, name="potential_flow"
    )
    actual_flow = model.addVars(
        range(T), range(n_prod), range(n_cons), range(num_rounds), range(n_pref),
        lb=0, name="actual_flow"
    )
    unmet = model.addVars(range(T), range(n_cons), lb=0, name="unmet")
    unused = model.addVars(range(T), range(n_prod), lb=0, name="unused")

    for t in range(T):
        demand_rem = {(j, r, p): model.addVar(lb=0) for j in range(n_cons) for r, p in round_pref}
        supply_rem = {(i, r): model.addVar(lb=0) for i in range(n_prod) for r in range(num_rounds)}

        for j in range(n_cons):
            model.addConstr(demand_rem[j, 0, 0] == demand_over_time[t, j])
        for i in range(n_prod):
            model.addConstr(supply_rem[i, 0] == supply_over_time[t, i])

        for idx, (r, p) in enumerate(round_pref):

            for i in range(n_prod):
                for j in range(n_cons):
                    model.addConstr(
                        potential_flow[t, i, j, r, p]
                        == w[i, j] * supply_rem[i, r] * preference_tensor[i, j, p]
                    )
                    model.addConstr(
                        actual_flow[t, i, j, r, p]
                        == gp.min_(potential_flow[t, i, j, r, p], demand_rem[j, r, p])
                    )

            for i in range(n_prod):
                allocated = quicksum(actual_flow[t, i, j, r, p] for j in range(n_cons))
                if r < num_rounds - 1:
                    model.addConstr(
                        supply_rem[i, r+1] == supply_rem[i, r] - allocated
                    )
                else:
                    model.addConstr(
                        unused[t, i] == supply_rem[i, r] - allocated
                    )

            for j in range(n_cons):
                allocated = quicksum(actual_flow[t, i, j, r, p] for i in range(n_prod))
                if idx < len(round_pref) - 1:
                    nr, np_ = round_pref[idx+1]
                    model.addConstr(
                        demand_rem[j, nr, np_] == demand_rem[j, r, p] - allocated
                    )
                else:
                    model.addConstr(
                        unmet[t, j] == demand_rem[j, r, p] - allocated
                    )

    # Objective: minimize total unmet + unused
    model.setObjective(
        quicksum(unmet[t, j] for t in range(T) for j in range(n_cons))
        + quicksum(unused[t, i] for t in range(T) for i in range(n_prod)),
        GRB.MINIMIZE
    )
    model.optimize()

    optimal_weights = {}
    for i in range(n_prod):
        prod_id = producers[i]
        optimal_weights[prod_id] = {}
        for j in range(n_cons):
            optimal_weights[prod_id][consumers[j]] = w[i, j].X

    total_unmet = sum(unmet[t, j].X for t in range(T) for j in range(n_cons))
    total_unused = sum(unused[t, i].X for t in range(T) for i in range(n_prod))
    return optimal_weights, total_unmet, total_unused