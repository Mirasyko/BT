#!/usr/bin/env python3

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum
import utils as ut
import time
import psutil

from pprint import pprint


def nonlinear_flow(excess: pd.DataFrame,
                    deficit: pd.DataFrame,
                    preference_df: pd.DataFrame,
                    num_rounds: int = 5,
                    write: bool = False):
    """
    Optimize producer-to-consumer energy allocation weights to minimize total unused supply 
    and unmet demand across multiple redistribution rounds.

    This function formulates and solves a nonlinear programming model using Gurobi. 
    It allocates energy from producers (with known excess) to consumers (with known deficit) 
    based on a preference matrix that determines eligible connections and their priorities.

    Parameters
    ----------
    excess : pd.DataFrame
        Time-indexed DataFrame where each column represents a producer's available energy over time.
    deficit : pd.DataFrame
        Time-indexed DataFrame where each column represents a consumer's energy demand over time.
    preference_df : pd.DataFrame
        Static DataFrame of shape (n_producers, n_consumers) with integer values in {0, 1, ..., 5},
        indicating the preference level (0 = no link) between each producer-consumer pair.
    num_rounds : int, optional
        Number of redistribution rounds per time step (default is 5).
    write : bool, optional
        If True, writes the Gurobi model to an LP file ("ER_weight_opt_flow.lp").

    Returns
    -------
    optimal_weights : pd.DataFrame
        A DataFrame of shape (n_consumers, n_producers) representing the optimized 
        weight matrix (proportion of supply each producer allocates to each consumer).
    total_unmet : float
        The total unmet demand across all consumers and time steps after redistribution.
    total_unused : float
        The total unused supply across all producers and time steps after redistribution.

    Notes
    -----
    - A feasible solution may not always exist. In such cases, the function returns zero weights
      and infinite values for unmet demand and unused supply.
    - Preference levels affect connectivity but not priority during allocation. Only binary links
      (preference > 0) are used to restrict possible allocations.
    - Energy flows are constrained at each redistribution round, updating supply and demand accordingly.
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

    model = gp.Model("NP Energy Allocation")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.03
    model.Params.Heuristics = 0.25

    w = model.addVars(n_cons, n_prod, lb=0, ub=1, name="w")
    S = model.addVars(range(T), range(num_rounds + 1), range(n_prod), lb=0, name="S")                   # First round is initial value and round + 1 is after 5th round
    D = model.addVars(range(T), range(num_rounds + 1), range(n_cons), lb=0, name="D")

    F = model.addVars(range(T), range(1, num_rounds + 1), range(n_prod), range(n_cons), lb=0, name="F")
    A = model.addVars(range(T), range(1, num_rounds + 1), range(n_prod), range(n_cons), lb=0, name="A")

    # Data init
    for t in range(T):
        for i in range(n_prod):
            model.addConstr(S[t, 0, i] == supply[t, i])
        for j in range(n_cons):
            model.addConstr(D[t, 0, j] == demand[t, j])
        
    for i in range(n_prod):
        model.addConstr(quicksum(w[j, i] for j in range(n_cons)) <= 1)
        for j in range(n_cons):
            model.addConstr(w[j,i] <= links[j, i])

    for t in range(T):
        for r in range(1, num_rounds + 1):
            for i in range(n_prod):
                for j in range(n_cons):
                        model.addConstr(F[t, r, i, j] <= w[j, i] * S[t,r - 1, i] * links[j, i])
                        model.addConstr(A[t, r, i, j] == gp.min_(D[t, r - 1, j], F[t,r, i, j]))
            
            for j in range(n_cons):
                model.addConstr(D[t, r, j] == D[t, r - 1, j] - quicksum(A[t, r, i, j] for i in range(n_prod)))
            
            for i in range(n_prod):
                model.addConstr(S[t, r, i] == S[t, r - 1, i] - quicksum(A[t, r, i, j] for j in range(n_cons)))
                

    unused_obj = quicksum(S[t, num_rounds, i] for t in range(T) for i in range(n_prod))
    unmet_obj  = quicksum(D[t, num_rounds, j] for t in range(T) for j in range(n_cons))
    model.setObjective(unused_obj + unmet_obj, GRB.MINIMIZE)

    if write:
        model.write("ER_weight_opt_flow.lp")

    model.optimize()

    if model.Status != 2:
        optimal_weights = pd.DataFrame({producers[i]: [0.0 for j in range(n_cons)]
                                        for i in range(n_prod)}, index=consumers)

        total_unmet = np.inf
        total_unused = np.inf

        return optimal_weights, total_unmet, total_unused

    optimal_weights = pd.DataFrame({producers[i]: [w[j,i].X for j in range(n_cons)]
                                    for i in range(n_prod)}, index=consumers)
    total_unused = sum(S[t, num_rounds, i].X for t in range(T) for i in range(n_prod))
    total_unmet = sum(D[t, num_rounds, j].X for t in range(T) for j in range(n_cons))

    return optimal_weights, total_unmet, total_unused



def linear_flow(
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

    model = gp.Model("NP Energy Allocation")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.03
    #model.Params.Heuristics = 0.25

    w = model.addVars(n_cons, n_prod, lb=0, ub=1, name="w")
    for i in range(n_prod):
        model.addConstr(quicksum(w[j, i] for j in range(n_cons)) <= 1)
        for j in range(n_cons):
            model.addConstr(w[j,i] <= links[j, i])

    S = model.addVars(range(T), range(num_rounds + 1), range(n_prod), lb=0, name="S")
    D = model.addVars(range(T), range(num_rounds + 1), range(n_cons), lb=0, name="D")
    f = model.addVars(range(T), range(1, num_rounds + 1), range(n_prod), range(n_cons), lb=0, name="f")


    for t in range(T):
        for i in range(n_prod):
            model.addConstr(S[t, 0, i] == supply[t, i])
        for j in range(n_cons):
            model.addConstr(D[t, 0, j] == demand[t, j])

 
    for t in range(T):
        for r in range(1, num_rounds + 1):
            for i in range(n_prod):
                for j in range(n_cons):
                    model.addConstr(f[t, r, i, j,] <= w[j, i] * S[t,  r - 1, i],
                        name=f"flow_sup_{t}_{r}_{i}_{j}")
                    model.addConstr(f[t, r, i, j] <= D[t, r - 1, j],
                        name=f"flow_dem_{t}_{r}_{i}_{j}")
            
            for i in range(n_prod):
                model.addConstr(S[t, r, i] == S[t, r - 1, i] - quicksum(f[t, r, i, j] for j in range(n_cons)),
                    name=f"sup_update_{t}_{r}_{i}")
                
            for j in range(n_cons):
                model.addConstr(D[t, r, j] == D[t, r - 1, j] - quicksum(f[t, r, i, j] for i in range(n_prod)),
                    name=f"dem_update_{t}_{j}_{r}")

    unused_obj = quicksum(S[t, num_rounds, i] for t in range(T) for i in range(n_prod))
    unmet_obj  = quicksum(D[t, num_rounds, j] for t in range(T) for j in range(n_cons))
    model.setObjective(unused_obj + unmet_obj, GRB.MINIMIZE)
    
    if write:
        model.write("ER_weight_opt.lp")

    model.optimize()

    if model.Status != 2:
        optimal_weights = pd.DataFrame({producers[i]: [0.0 for j in range(n_cons)]
                                        for i in range(n_prod)}, index=consumers)

        total_unmet = np.inf
        total_unused = np.inf

        return optimal_weights, total_unmet, total_unused



    optimal_weights = pd.DataFrame({producers[i]: [w[j,i].X for j in range(n_cons)]
                                    for i in range(n_prod)}, index=consumers)

    total_unused = sum(S[t, num_rounds, i].X for t in range(T) for i in range(n_prod))
    total_unmet = sum(D[t, num_rounds, j].X for t in range(T) for j in range(n_cons))

    return optimal_weights, total_unmet, total_unused



def linear_flow_int(
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
    # model.Params.MIPGap = 0.01
    # model.Params.Heuristics = 0.01

    w = model.addVars(n_cons, n_prod, lb=0, ub=100,vtype=GRB.INTEGER, name="w")
    for i in range(n_prod):
        model.addConstr(quicksum(w[j, i] for j in range(n_cons)) <= 100)
        for j in range(n_cons):
            model.addConstr(w[j,i] <= 100*links[j, i])

    S = model.addVars(range(T), range(num_rounds + 1), range(n_prod), lb=0, name="S")
    D = model.addVars(range(T), range(num_rounds + 1), range(n_cons), lb=0, name="D")
    f = model.addVars(range(T), range(1, num_rounds + 1), range(n_prod), range(n_cons), lb=0, name="f")


    for t in range(T):
        for i in range(n_prod):
            model.addConstr(S[t, 0, i] == supply[t, i])
        for j in range(n_cons):
            model.addConstr(D[t, 0, j] == demand[t, j])

 
    for t in range(T):
        for r in range(1, num_rounds + 1):
            for i in range(n_prod):
                for j in range(n_cons):
                    model.addConstr(f[t, r, i, j,] <= w[j, i]/100 * S[t,  r - 1, i],
                        name=f"flow_sup_{t}_{r}_{i}_{j}")
                    model.addConstr(f[t, r, i, j] <= D[t, r - 1, j],
                        name=f"flow_dem_{t}_{r}_{i}_{j}")
            
            for i in range(n_prod):
                model.addConstr(S[t, r, i] == S[t, r - 1, i] - quicksum(f[t, r, i, j] for j in range(n_cons)),
                    name=f"sup_update_{t}_{r}_{i}")
                
            for j in range(n_cons):
                model.addConstr(D[t, r, j] == D[t, r - 1, j] - quicksum(f[t, r, i, j] for i in range(n_prod)),
                    name=f"dem_update_{t}_{j}_{r}")

    unused_obj = quicksum(S[t, num_rounds, i] for t in range(T) for i in range(n_prod))
    unmet_obj  = quicksum(D[t, num_rounds, j] for t in range(T) for j in range(n_cons))
    flow = quicksum(f[t, r, i, j] for t in range(T) for r in range(1, num_rounds + 1) for i in range(n_prod) for j in range(n_cons))
    model.setObjective(unused_obj + unmet_obj, GRB.MINIMIZE)
    
    if write:
        model.write("ER_weight_opt.lp")

    model.optimize()

    if model.Status != 2:
        optimal_weights = pd.DataFrame({producers[i]: [0.0 for j in range(n_cons)]
                                        for i in range(n_prod)}, index=consumers)

        total_unmet = np.inf
        total_unused = np.inf

        return optimal_weights, total_unmet, total_unused



    optimal_weights = pd.DataFrame({producers[i]: [w[j,i].X/100 for j in range(n_cons)]
                                    for i in range(n_prod)}, index=consumers)

    total_unused = sum(S[t, num_rounds, i].X for t in range(T) for i in range(n_prod))
    total_unmet = sum(D[t, num_rounds, j].X for t in range(T) for j in range(n_cons))
    flow = sum(f[t, r, i, j].X for t in range(T) for r in range(1, num_rounds + 1) for i in range(n_prod) for j in range(n_cons))

    return optimal_weights, total_unmet, total_unused#, flow

def main():
    sizes = [3000] #, 25, 50, 100, 250, 500, 1000, 2000] #, 250, 500]# 1000, 1500, 2000, 3000]
    results = []

    for size in sizes:
        excess, deficit, preference_df = ut.real_dataset(month=4, start=0, end=size)
        num_rounds = 3

        # --------------------------
        # Model 1: Nonlinear flow
        # --------------------------

        start = time.time()

        weights_nl, unmet_nl, unused_nl = nonlinear_flow(
            excess, deficit, preference_df, num_rounds, write=False)

        end = time.time()

        results.append({
            "model": "nonlinear_flow",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(unmet_nl, 6),
            "unused_energy": round(unused_nl, 6),
            "fitness": ut.fitness(weights_nl, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_nl, excess, deficit)
        })


        # --------------------------
        # Model 2: Linear flow
        # --------------------------
        start = time.time()

        weights_l, unmet_l, unused_l = linear_flow(
            excess, deficit, preference_df, num_rounds, write=False)

        end = time.time()

        results.append({
            "model": "linear_flow",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(unmet_l, 6),
            "unused_energy": round(unused_l, 6),
            "fitness": ut.fitness(weights_l, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_l, excess, deficit)
        })
    
        # --------------------------
        # Model 3: Linear INTEGER flow
        # --------------------------
        start = time.time()

        weights_il, unmet_il, unused_il = linear_flow(
            excess, deficit, preference_df, num_rounds, write=True)

        end = time.time()

        results.append({
            "model": "linear_flow_int",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(unmet_il, 6),
            "unused_energy": round(unused_il, 6),
            "fitness": ut.fitness(weights_il, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_il, excess, deficit)
        })

        pprint({
            "model": "linear_flow",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(unmet_il, 6),
            "unused_energy": round(unused_il, 6),
            "fitness": ut.fitness(weights_il, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_il, excess, deficit),
            "ideal"   : np.abs(
            excess.sum(axis=1) - deficit.sum(axis=1)
        ).sum(),
         "worst" : (excess.sum(axis=1) + deficit.sum(axis=1)).sum()
        })


    df_results = pd.DataFrame(results)

    print(df_results)
    
    print((excess.sum(axis=1) + deficit.sum(axis=1)).sum())

    print(np.abs(
            excess.sum(axis=1) - deficit.sum(axis=1)
        ).sum())
    df_results.to_csv('/home/miro/Bachelor/BT/Outputs/model_comparison.csv')



if __name__ == '__main__':
    main()
