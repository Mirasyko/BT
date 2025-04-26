#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import utils as ut
from optimizer_flow_model import optimize_flow_time_series
from weights_optimization_model import optimize_weights
import time

def main():
    ut.set_seed()


    data_dir = '/home/miro/Bachelor/BT/data/outputs'
    output_dir = '/home/miro/Bachelor/BT/Models/outputs/'

    excess_monthly, deficit_monthly = ut.load_excess_deficit(data_dir)
    excess_df, deficit_df = ut.prepare_data(excess_monthly, deficit_monthly, month=4, start=0, end=250)
    producers = list(excess_df.columns)
    consumers = list(deficit_df.columns)
    # excess_df = excess_df[producers]
    # deficit_df= deficit_df[consumers]

    # Build preference tensor
    preference_df = ut.generate_preferences(producers, consumers, 5)
    preference_df = preference_df.loc[producers,consumers]

    pref_tensor = ut.create_preference_tensor(preference_df)

    # --- Model 1: nonconvex flow, per time slice ---
    start1 = time.time()
    # w1, unmet1, unused1 = optimize_flow_time_series(
    # excess_df,
    # deficit_df,
    # pref_tensor,
    # num_rounds=5
    # )
    end1 = time.time()
    # --- Model 2: MILP on full time series --

    start2 = time.time()
    w2, unmet2, unused2, model2 = optimize_weights(
        excess=excess_df,
        deficit=deficit_df,
        preference_df=preference_df,
        num_rounds=5
    )
    end2 = time.time()

    # fitness1 = ut.fitness(pd.DataFrame.from_dict(w1), preference_df, excess_df, deficit_df, 5)
    fitness2 = ut.fitness(w2, preference_df, excess_df, deficit_df, 5)


    print("--- Weight Comparison ---")
    print("\n--- Aggregate Performance ---")
    # print(f"Total unmet demand (flow model):  {unmet1:.4f}")
    # print(f"Total unused supply (flow model): {unused1:.4f}")
    # print(f"Total unused energy (flow model): {unused1 + unmet1:.4f}")
    # print(f"Total fitness (flow model):       \n{fitness1=}")
    # print(f"Total time (flow):                {end1 - start1:.4f}")
    # print('-'*50)
    print(f"Model status:                     {model2.status}")
    print(f"Total unmet demand (MILP):        {unmet2:.4f}")
    print(f"Total unused supply (MILP):       {unused2:.4f}")
    print(f"Total unused energy (flow model): {unused2 + unmet2:.4f}")
    print(f"Total fitness (MILP):             \n{fitness2}")
    print(f"Total time (flow):                {end2 - start2:.4f}")

    preference_df.to_csv(output_dir + 'preferences.csv')
    # pd.DataFrame.from_dict(w1).to_csv(output_dir + 'weights_1.csv')
    pd.DataFrame.from_dict(w2).to_csv(output_dir + 'weights_2.csv')

def test():
    
    time_index = pd.date_range("2025-01-01", periods=1, freq="15T")
    excess_df = pd.DataFrame(
        [[90, 50]],# [40, 70], [30, 80]],
        index=time_index,
        columns=["P0", "P1"]
    )
    deficit_df = pd.DataFrame(
        [[30, 50, 50]], #[60, 20, 40], [30, 60, 20]],
        index=time_index,
        columns=["C0", "C1", "C2"]
    )
    preference_df = pd.DataFrame(
        [[5, 3, 0],
        [0, 0, 1]],
        index=["P0", "P1"],
        columns=["C0", "C1", "C2"]
    )

    opt_weights, unmet, unused = optimize_weights(
        excess_df, deficit_df, preference_df, num_rounds=2, write=True
    )
    model_obj = unmet + unused

    weight_df = pd.DataFrame.from_dict(opt_weights, orient="index")

    producer_energy_data = pd.concat([
        pd.Series(excess_df.index.astype(str), name="timestamp"),
        excess_df.reset_index(drop=True)
    ], axis=1)

    subscriber_demand_data = pd.concat([
        pd.Series(deficit_df.index.astype(str), name="timestamp"),
        deficit_df.reset_index(drop=True)
    ], axis=1)

    fitness_score = ut.fitness(
        weight_df, preference_df,
        producer_energy_data,
        subscriber_demand_data
    )

    print(f"Model objective (unmet+unused): {model_obj:.6f}")
    print(f"Fitness function score       : {fitness_score:.6f}")



if __name__ == '__main__':
    main()
