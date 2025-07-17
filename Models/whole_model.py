#!/usr/bin/env python3

# This file contains a model that optimizes weigths and preferences

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pprint import pprint
import time
import utils as ut

def whole_model(excess: pd.DataFrame, deficit: pd.DataFrame, num_rounds: int = 5, max_pref: int = 5, write: bool = False):
    T = range(excess.shape[0])      # časové kroky
    J = list(excess.columns)        # výrobci
    I = list(deficit.columns)       # spotřebitelé
    L = range(1, max_pref + 1)      # priority levels (1-based)
    R = range(1, num_rounds + 1)    # redistribuční kola (1-based)

    model = gp.Model("Whole_Model_With_Priorities")
    model.setParam("OutputFlag", 0)

    n, m = len(I), len(J)
    E = excess.to_numpy()
    N = deficit.to_numpy()

    # Proměnné
    w = model.addVars(n, m, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="w")
    y = model.addVars(max_pref, n, m, vtype=GRB.BINARY, name="y")
    F = model.addVars(len(T), num_rounds, max_pref, n, m, lb=0, vtype=GRB.CONTINUOUS, name="F")
    S = model.addVars(len(T), num_rounds + 1, m, lb=0, vtype=GRB.CONTINUOUS, name="S")
    D = model.addVars(len(T), num_rounds + 1, n, lb=0, vtype=GRB.CONTINUOUS, name="D")

    # Počáteční podmínky
    for t in T:
        for i in range(n):
            model.addConstr(D[t, 0, i] == N[t, i])
        for j in range(m):
            model.addConstr(S[t, 0, j] == E[t, j])

    # Redistribuce
    for t in T:
        for r in R:
            r_idx = r - 1
            for i in range(n):
                for j in range(m):
                    for l in L:
                        l_idx = l - 1
                        model.addConstr(F[t, r, l_idx, i, j] <= w[i, j] * S[t, r_idx, j] * y[l_idx, i, j])

                    # Priorita 1
                    model.addConstr(F[t, r, 0, i, j] <= D[t, r, i])

                    # Priority > 1
                    for l in range(1, max_pref):
                        model.addConstr(
                            F[t, r, l, i, j] <= D[t, r, i] -
                            gp.quicksum(F[t, r, lp, i, jj] for lp in range(l) for jj in range(m))
                        )

            # Aktualizace stavu
            for j in range(m):
                model.addConstr(
                    S[t, r, j] == S[t, r - 1, j] -
                    gp.quicksum(F[t, r - 1, l, i, j] for l in range(max_pref) for i in range(n))
                )
            for i in range(n):
                model.addConstr(
                    D[t, r, i] == D[t, r - 1, i] -
                    gp.quicksum(F[t, r - 1, l, i, j] for l in range(max_pref) for j in range(m))
                )

    # Omezení na váhy
    for j in range(m):
        model.addConstr(gp.quicksum(w[i, j] for i in range(n)) <= 1)

    for i in range(n):
        for j in range(m):
            model.addConstr(w[i, j] <= gp.quicksum(y[l, i, j] for l in range(max_pref)))

    for i in range(n):
        model.addConstr(gp.quicksum(y[l, i, j] for l in range(max_pref) for j in range(m)) <= max_pref)

    # Cíl
    unmet = gp.quicksum(D[t, num_rounds, i] for t in T for i in range(n))
    unused = gp.quicksum(S[t, num_rounds, j] for t in T for j in range(m))
    model.setObjective(unmet + unused, GRB.MINIMIZE)

    if write:
        model.write("whole_model.lp")

    model.optimize()

    # Získání matic vah
    weight_df = pd.DataFrame(
        [[w[i, j].X for j in range(m)] for i in range(n)],
        index=I,
        columns=J
    )

    # Získání matic priorit
    priority_df = pd.DataFrame(0, index=I, columns=J, dtype=int)
    for i in range(n):
        for j in range(m):
            for l in range(max_pref):
                if y[l, i, j].X > 0.5:
                    priority_df.iloc[i, j] = max_pref - l + 1  # l je 0-based, ale priority jsou 1-based

    # Výstupy
    return weight_df, priority_df, model.ObjVal, model


def main():
    sizes = [10] #, 25] #, 250, 500]# 1000, 1500, 2000, 3000]
    start = 35
    results = []
    print(35)
    for size in sizes:
        excess, deficit, _ = ut.real_dataset(month=4, start=start, end=start+size)
        # num_rounds = 4
        # num_pref = 4
        start = time.time()

        weights_il, preference_df, obj, _ = whole_model(
            excess, deficit)

        end = time.time()

        pprint(weights_il)
        pprint(preference_df)

        results.append({
            "model": "linear_flow_int",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(obj, 6),
            "fitness": ut.fitness(weights_il, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_il, excess, deficit)
        })
        pprint({
            "model": "linear_flow_int",
            "size": len(excess),
            "time_sec": round(end - start, 4),
            "unmet_energy": round(obj, 6),
            "fitness": ut.fitness(weights_il, preference_df, excess, deficit, 3),
            "fitness_static": ut.static_fitness(weights_il, excess, deficit),
            "ideal"   : np.abs(
            excess.sum(axis=1) - deficit.sum(axis=1)
        ).sum(),
         "worst" : (excess.sum(axis=1) + deficit.sum(axis=1)).sum()
        })
    
    print(results)

if __name__ == '__main__':
    main()
