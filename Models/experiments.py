import os
import pandas as pd
import numpy as np
import time

import utils as ut
from heuristics import run_genetic_algorithm, evaluate_matrix, simulated_annealing, hill_climb, run_fixed_link_ga
from dynamic_weights import linear_flow_int, linear_flow
from static_weights import milp_static_model, lp_static_model

def experiment():
    m = 7
    start_day = 30
    sizes = [5, 25, 50, 100, 250, 500, 1000]
    results = []

    for size in sizes:
        print(f"\n--- Running experiment for size = {size} (days {start_day} to {start_day + size}) ---")
        end_day = start_day + size

        excess_df, deficit_df, _ = ut.real_dataset(m, start_day, end_day, all_data=False)

        ideal_min_balance = np.abs(
            excess_df.sum(axis=1) - deficit_df.sum(axis=1)
        ).sum()
        worst_balance = (excess_df.sum(axis=1) + deficit_df.sum(axis=1)).sum()

        print("### STATIC MILP")
        milp_start = time.time()
        W_milp, links_milp, milp_model = milp_static_model(
            excess_df, deficit_df, time_limit=500000
        )

        fitness_milp = ut.static_fitness(
            weights_df=W_milp,
            producer_energy_data=excess_df,
            subscriber_demand_data=deficit_df
        )

        print("### FIXED-LINK GA (fast test, optional)")
        best_M, best_fit = run_fixed_link_ga(
            link_df=links_milp,
            weights=W_milp,
            excess_df=excess_df,
            deficit_df=deficit_df,
            generations=3,
            population_size=10,
            mutation_rate=0.2,
            crossover_rate=0.9
        )
        milp_end = time.time()
        milp_time = np.round((milp_end - milp_start) / 60, 4)

        print("### GA+MILP (priority optimization)")
        ga_common = dict(
            generations      = 5,
            population_size  = 10,
            mutation_rate    = 0.1,
            crossover_rate   = 0.9,
            tournament_size  = 4,
            elitism_count    = 2,
            mutation_type    = "row_replace",
            initial_non_zero = 5,
        )

        ga_start = time.time()
        ga_priorities, ga_fit, ga_best_weights = run_genetic_algorithm(
            excess_df, deficit_df, **ga_common
        )
        ga_end = time.time()
        ga_time = np.round((ga_end - ga_start) / 60, 2)

        static_fitness_score = ut.static_fitness(
            weights_df=ga_best_weights,
            producer_energy_data=excess_df,
            subscriber_demand_data=deficit_df
        )

        results.append(dict(
            Size                      = len(deficit_df),
            TimeSteps                 = len(excess_df),
            Producers                 = excess_df.shape[1],
            Consumers                 = deficit_df.shape[1],
            ideal_min_balance         = ideal_min_balance,
            worst_balance             = worst_balance,
            STATIC_MILP_SCORE         = fitness_milp,
            SEQ_MILP_SCORE            = best_fit,
            MILP_time_min             = milp_time,
            STATIC_GA_SCORE           = static_fitness_score,
            SEQ_MA_SCORE              = ga_fit,
            GA_time_min               = ga_time,
        ))

    df_results = pd.DataFrame(results)
    print(df_results)
    df_results.to_csv('/home/miro/Bachelor/BT/Outputs/EXPERIMENT_5.csv')
    return df_results

if __name__ == '__main__':
    
    experiment()