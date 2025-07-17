import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from typing import Tuple, Dict, List
import warnings
import time
import utils as ut
from pprint import pprint

warnings.filterwarnings('ignore')


def static_weights(excess: pd.DataFrame, deficit: pd.DataFrame, 
                time_limit: int = 600, relaxed: bool = False) -> gp.Model:
    """
    Build Gurobi optimization model for energy allocation.
    
    Args:
        excess: DataFrame with producer surplus (kWh) at each time interval
        deficit: DataFrame with consumer shortfall (kWh) at each time interval
        time_limit: Solver time limit in seconds
        relaxed: If True, treat y variables as continuous (LP relaxation)
    
    Returns:
        Configured Gurobi model
    """
    
    # Initialize model
    model = gp.Model("energy_allocation")
    # model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0  # Enable output
    model.Params.MIPGap = 0.03
    
    producers = list(excess.columns)
    consumers = list(deficit.columns)
    supply = excess.values
    demand = deficit.values

    T = excess.shape[0]
    n_prod = len(producers)
    n_cons = len(consumers)


    # Decision variables
    # w[i,j]: percentage weight (0-100) for producer i to consumer j
    w = model.addVars(n_cons, n_prod, lb=0, ub=100, vtype=GRB.INTEGER, name="w")
    
    # y[i,j]: binary indicator for link between producer i and consumer j
    if relaxed:
        y = model.addVars(n_cons, n_prod, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
    else:
        y = model.addVars(n_cons, n_prod, vtype=GRB.BINARY, name="y")
    
    for i in range(n_prod):
        model.addConstr(quicksum(w[j, i] for j in range(n_cons)) <= 100)
        for j in range(n_cons):
            model.addConstr(w[j,i] <= 100 * y[j, i])
    
    for j in range(n_cons):
        model.addConstr(gp.quicksum(y[j,i] for i in range(n_prod)) <= 5, name=f'maxLinks_{j}')
    
    f = model.addVars(range(T), range(n_prod), range(n_cons), lb=0, name="f")

    for t in range(T):
            
            for j in range(n_cons):
                for i in range(n_prod):
                    model.addConstr(f[t, i, j,] <= w[j, i]/100 * supply[t, i],
                        name=f"flow_sup_{t}_{i}_{j}")
            
            for j in range(n_cons):
                model.addConstr(quicksum(f[t, i, j] for i in range(n_prod)) <= demand[t, j],
                        name=f"flow_dem_{t}_{j}")


    flow = quicksum(f[t, i, j] for t in range(T) for i in range(n_prod) for j in range(n_cons))
    model.setObjective(flow, GRB.MAXIMIZE)
    
    # Store metadata for later use
    model._producers = producers
    model._consumers = consumers
    model._w_vars = w
    model._y_vars = y
    
    return model

def extract_weights(model: gp.Model) -> pd.DataFrame:
    P, C, w = model._producers, model._consumers, model._w_vars
    W = pd.DataFrame(index=C, columns=P, dtype=float)
    
    for j, c in enumerate(C):
        for i, p in enumerate(P):
            W.at[c, p] = round(w[j, i].X / 100, 2) if model.Status == GRB.OPTIMAL else 0.0
    return W

def extract_links(model):
    P, C, y = model._producers, model._consumers, model._y_vars
    links = pd.DataFrame(index=C, columns=P, dtype=float)
    for j, c in enumerate(C):
        for i, p in enumerate(P):
            links.at[c, p] = int(y[j, i].X)  if model.Status == GRB.OPTIMAL else 0.0
    return links



def milp_static_model(excess: pd.DataFrame, deficit: pd.DataFrame, 
                           time_limit: int = 600):
    milp_model = static_weights(excess, deficit, time_limit=time_limit, relaxed=False)
    milp_model.optimize()
    W_milp = extract_weights(milp_model)
    links = extract_links(milp_model)
    return W_milp, links, milp_model


def lp_static_model(excess: pd.DataFrame, deficit: pd.DataFrame, 
                           time_limit: int = 600):
    lp_model = static_weights(excess, deficit, time_limit=time_limit, relaxed=True)
    lp_model.optimize()
    W_lp = extract_weights(lp_model)
    links = extract_links(lp_model) 
    return W_lp, links, lp_model


def testing(month: int = 4, start: int = 0, size: int = 35040, 
            time_limit: int = 600) -> Dict:
    """
    Complete testing workflow that loads real data, optimizes weights, 
    calculates fitness and runtime.
    
    Args:
        month: Month for data loading
        start: Start index for data
        size: Size of data to load
        time_limit: Solver time limit in seconds
    
    Returns:
        Dictionary with model name, undistributed energy, fitness, and runtime
    """
    
    print("=== LOADING REAL DATA ===")
    start_time = time.time()
    
    # Load real data using utils function
    excess, deficit, *_ = ut.real_dataset(month=month, start=start, end=start+size)
    
    data_load_time = time.time() - start_time
    print(f"Data loaded in {data_load_time:.2f} seconds")
    print(f"Data shape - Excess: {excess.shape}, Deficit: {deficit.shape}")
    ideal_min_balance = np.abs(
            excess.sum(axis=1) - deficit.sum(axis=1)
        ).sum()

    worst_balance = (excess.sum(axis=1) + deficit.sum(axis=1)).sum()

    print("\n=== OPTIMIZING WEIGHTS ===")
    optimization_start = time.time()
    
    # Solve optimization problem
    W_optimal, links_milp, milp_model = milp_static_model(
        excess, deficit, time_limit=time_limit
    )
    
    optimization_time = time.time() - optimization_start
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    
    print("\n=== CALCULATING FITNESS ===")
    fitness_start = time.time()
    
    # Calculate fitness using utils function
    fitness_score = ut.static_fitness(
        weights_df=W_optimal,
        producer_energy_data=excess,
        subscriber_demand_data=deficit
    )
    
    fitness_time = time.time() - fitness_start
    print(f"Fitness calculated in {fitness_time:.2f} seconds")
    
    # Total runtime
    total_runtime = time.time() - start_time
    
    # Prepare results dictionary
    results = {
        "IDEAL BALANCE":ideal_min_balance,
        'fitness': fitness_score,
        "WORST POSSIBLE BALANCE" : worst_balance,
        'RUNTIME': round(optimization_time, 2),  # Only optimization time
        'total_runtime': round(total_runtime, 2),  # Total including data loading
        'data_load_time': round(data_load_time, 2),
        'fitness_calc_time': round(fitness_time, 2),
        'MILP optimization_status': 'OPTIMAL' if milp_model.Status == GRB.OPTIMAL else 'SUBOPTIMAL',
        'MILP objective_value': milp_model.ObjVal if milp_model.Status == GRB.OPTIMAL else None #,
        # 'LP optimization_status': 'OPTIMAL' if lp_model.Status == GRB.OPTIMAL else 'SUBOPTIMAL',
        # 'LP objective_value': lp_model.ObjVal if lp_model.Status == GRB.OPTIMAL else None
        

    }
    return results


def exper(month: int = 4, start: int = 0, size: int = 35040, 
            time_limit: int = 600):
    start_time = time.time()
    
    # Load real data using utils function
    excess, deficit, *_ = ut.real_dataset(month=month, start=start, end=start+size)
    
    data_load_time = time.time() - start_time
    print(f"Data loaded in {data_load_time:.2f} seconds")
    print(f"Data shape - Excess: {excess.shape}, Deficit: {deficit.shape}")
    ideal_min_balance = np.abs(
            excess.sum(axis=1) - deficit.sum(axis=1)
        ).sum()

    worst_balance = (excess.sum(axis=1) + deficit.sum(axis=1)).sum()

    print("\n=== OPTIMIZING WEIGHTS ===")
    optimization_start = time.time()
    
    # Solve optimization problem
    W_optimal, links_milp, milp_model = milp_static_model(
        excess, deficit, time_limit=time_limit
    )
    
    optimization_time = time.time() - optimization_start
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    pprint(W_optimal)
    pprint(links_milp)


# Example usage and testing
if __name__ == "__main__":
    # Test with real data

    # Run testing function
    for size in [10]: #, 3000, 3000, 3000]:
        test_results = testing(month=7, start=0, size=size, time_limit=50000)  # Smaller size for testing
        
        print(f"\n=== FINAL RESULTS DICTIONARY FOR SIZE {size} ===")
        for key, value in test_results.items():
            print(f"{key}: {value}")
        
    exper(month=7, start=150, size=10, time_limit=50000)