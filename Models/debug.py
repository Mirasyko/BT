import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm
import os 

EXCESS_CSV_PATH = '/home/miro/Bachelor/BT/data/outputs/excess.csv'
DEFICIT_CSV_PATH = '/home/miro/Bachelor/BT/data/outputs/deficit.csv'

excess = pd.read_csv(EXCESS_CSV_PATH)
deficit = pd.read_csv(DEFICIT_CSV_PATH)

excess['timestamp'] = pd.to_datetime(excess['timestamp'])
deficit['timestamp'] = pd.to_datetime(deficit['timestamp'])
excess['year_month'] = excess['timestamp'].dt.to_period('M')
deficit['year_month'] = deficit['timestamp'].dt.to_period('M')

excess_monthly = {}
deficit_monthly = {}

for i, month in enumerate(deficit['year_month'].unique()):

    excess_monthly[i] = excess[excess['year_month'] == month].drop('year_month', axis=1)
    deficit_monthly[i] = deficit[deficit['year_month'] == month].drop('year_month', axis=1)

month = 6

deficit_df = deficit_monthly[month].iloc[150:160,:]
excess_df = excess_monthly[month].iloc[150:160,:]

excess_df = excess_df.apply(pd.to_numeric, errors='coerce').fillna(0)
deficit_df = deficit_df.apply(pd.to_numeric, errors='coerce').fillna(0)

producers = list(excess_df.columns[1:])
consumers = list(deficit_df.columns[1:])

def generate_weights(producers, consumers):
    """
    select random consumer for each producer and then se
    """
    weights = {}
    for producer in producers:
        picked_cons = np.random.choice([c for c in consumers if c != producer], size=5, replace=False)
        assigned_weights = np.random.dirichlet(np.ones(len(picked_cons)), size=1)[0]  # Ensures sum = 1
        weights[producer] = dict(zip(picked_cons, assigned_weights))
    return pd.DataFrame(weights).fillna(0)

def generate_preferences(producers, consumers, n=5):
    preferences = {}
    for consumer in consumers:
        preferred_producers = np.random.choice([p for p in producers if p != consumer], size=n, replace=False)
        preference_values = np.random.permutation([i for i in range(1, n+1)])
        preferences[consumer] = dict(zip(preferred_producers, preference_values))
    return pd.DataFrame(preferences).fillna(0)

weights_df = generate_weights(producers, consumers)
preferences_df = generate_preferences(producers, consumers)


def step(available_energy, remaining_demand, weights_df, preferences_df, rounds=5):
    producers = available_energy.index
    consumers = remaining_demand.index

    weights = weights_df.loc[consumers, producers].T.values
    preferences = preferences_df.loc[producers, consumers].values

    for _ in range(rounds):
        allocation = available_energy.values[:, None] * weights

        for j in range(len(consumers)):
            pref = preferences[:, j]
            sorted_idx = np.argsort(-pref)
            for i in sorted_idx:
                if pref[i] <= 0 or remaining_demand.iloc[j] <= 0:
                    continue
                transfer = min(allocation[i, j], remaining_demand.iloc[j])
                remaining_demand.iloc[j] -= transfer
                available_energy.iloc[i] -= transfer
                allocation[i, j] -= transfer

    return remaining_demand, available_energy

def allocation(weights_df, preferences_df, producer_energy_data, subscriber_demand_data):
    
    num_timesteps = len(producer_energy_data)
    allocation_matrix = np.zeros((num_timesteps, 3))

    for t in range(num_timesteps):
        available_producer_energy = producer_energy_data.iloc[t, 1:].copy()
        remaining_subscriber_demand = subscriber_demand_data.iloc[t, 1:].copy()

        available_producer_energy[available_producer_energy < 1e-9] = 0
        remaining_subscriber_demand[remaining_subscriber_demand < 1e-9] = 0
        
        remaining_demand_t, available_energy_t  = step(available_producer_energy, remaining_subscriber_demand, weights_df, preferences_df)
        allocation_matrix[t, :] = [t, remaining_demand_t.sum(), available_energy_t.sum()]

    alloc_df = pd.DataFrame(data=allocation_matrix, columns = ['timestamp', 'unsatisfied_demand', 'available_energy'])

    return alloc_df

def fitness(alloc_df):
    return alloc_df[alloc_df['unsatisfied_demand'] > 0 & alloc_df['available_energy'] > 0][['unsatisfied_demand', 'available_energy']].sum()

alloc_df = allocation(weights_df, preferences_df, excess_df, deficit_df) 
