import os
import numpy as np
import pandas as pd
import random

def set_seed(seed: int = 42):
    """
    Set seed for testing.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def prepare_data(excess_monthly, deficit_monthly, month , start=0, end=10, all_data=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for testing. Automatically gets rid of zero entries,
    because their is nothing to optimize. Returns excess and deficit in range start-end.

    Args:
        excess_monthly (dict[int, pd.DataFrame]): Dictionary of monthly dataframes
        deficit_monthly (dict[int, pd.DataFrame]): Dictionary of monthly dataframes
        month (int): integer value of month starting from 0.
        start (int, optional): Defaults to 0.
        end (int, optional): Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Returns excess and deficit in range start-end.
    """
    deficit_df = deficit_monthly[month]
    excess_df = excess_monthly[month]

    excess_df = excess_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    deficit_df = deficit_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    excess_df = excess_df.drop(columns='timestamp').reset_index(drop=True)
    deficit_df = deficit_df.drop(columns='timestamp').reset_index(drop=True)

    if all_data:
        return excess_df, deficit_df

    excess_df['sum'] = excess_df.sum(axis=1)
    deficit_df['sum'] = deficit_df.sum(axis=1)

    deficit_df = deficit_df.iloc[excess_df[excess_df['sum'] > 0.0].index].reset_index(drop=True)
    excess_df = excess_df[excess_df['sum'] > 0.0].reset_index(drop=True)

    deficit_df = deficit_df.drop(columns='sum')
    excess_df = excess_df.drop(columns='sum')

    end = np.min([end, len(deficit_df)])

    return excess_df.iloc[start:end], deficit_df.iloc[start:end]


def load_excess_deficit(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load time-series of excess supply and deficit demand from CSV files.

    Expects files 'excess.csv' and 'deficit.csv' in data_dir, with a time index.

    Returns:
        excess: DataFrame (time x producers)
        deficit: DataFrame (time x consumers)
    """
    excess_path = os.path.join(data_dir, "excess.csv")
    deficit_path = os.path.join(data_dir, "deficit.csv")
    excess = pd.read_csv(excess_path)
    deficit = pd.read_csv(deficit_path)
    excess['timestamp'] = pd.to_datetime(excess['timestamp'])
    deficit['timestamp'] = pd.to_datetime(deficit['timestamp'])
    excess['year_month'] = excess['timestamp'].dt.to_period('M')
    deficit['year_month'] = deficit['timestamp'].dt.to_period('M')

    excess_monthly = {}
    deficit_monthly = {}

    for i, month in enumerate(deficit['year_month'].unique()):

        excess_monthly[i] = excess[excess['year_month'] == month].drop('year_month', axis=1)
        deficit_monthly[i] = deficit[deficit['year_month'] == month].drop('year_month', axis=1)
        
    return excess_monthly, deficit_monthly


def create_preference_tensor(
    preference_df
) -> np.ndarray:
    """
    Build a simple preference tensor of shape (n_producers, n_consumers, pref_levels).

    Returns:
        tensor: np.ndarray[int] of shape (n_producers, n_consumers, pref_levels),
                where tensor[i,j,p] = 1 if (i+j)%pref_levels == p else 0.
    """
    n_producers, n_consumers = preference_df.shape
    max_level = 5

    p = np.zeros((n_producers, n_consumers, max_level), dtype=int)

    for level in range(1, max_level + 1):
        mask = (preference_df.values == level)
        p[:, :, level - 1] = mask.astype(int)

    return p

def generate_preferences(producers, consumers, n=5):
    """
    Return random preferences.

    Args:
        producers (list): list of producer names
        consumers (list): list of consumer names
        n (int, optional): non-zero preferences. Defaults to 5.

    Returns:
        pd.DataFrame: dataframe with producers names(columns) and consumer names(index) and priorities
    """
    preferences = {}

    for consumer in consumers:
        preferred_producers = np.random.choice([p for p in producers if p != consumer], size=n, replace=False)
        preference_values = np.random.permutation([i for i in range(1, n+1)])
        preferences[consumer] = dict(zip(preferred_producers, preference_values))

    preferences = pd.DataFrame(preferences).fillna(0)
    try:
        preferences = preferences.loc[producers,consumers]
    except Exception as e:
        print(f'ERROR -> {e}')
        missing_key = [col for col in consumers if col not in preferences.index]
        for key in missing_key:
            preferences.iloc[key] = [0.0 for _ in range(len(consumers))]
        preferences = preferences.loc[producers,consumers]

    return preferences.T

def step(available_energy, remaining_demand, weights_df, preferences_df, rounds=5):
    """
    One step in fitness function. One timestamp allocation.

    Args:
        available_energy (pd.Series): Supple of each producer
        remaining_demand (pd.Series): Demand for each consumer
        weights_df (pd.DataFrame): weights dataframe
        preferences_df (pd.DataFrame): Preference dataframe
        rounds (int, optional): number of allocation rounds. Defaults to 5.

    Returns:
        tuple[pd.Series, pd.Series]: series of unallocated energy.
    """
    producers = available_energy.index
    consumers = remaining_demand.index

    weights = weights_df.loc[producers, consumers].values
    preferences = preferences_df.loc[producers,consumers].values

    for _ in range(rounds):
        allocation = weights.T * available_energy.values[:, None]

        for i in range(len(producers)):
            if available_energy.iloc[i] <= 0:
                continue
            pref = preferences[i, :]
            sorted_idx = np.argsort(-pref)
            for j in sorted_idx:
                if pref[j] <= 0 or remaining_demand.iloc[j] <= 0 or allocation[i, j] <= 0:
                    continue
                transfer = min(allocation[i, j], remaining_demand.iloc[j])
                remaining_demand.iloc[j] -= transfer
                available_energy.iloc[i] -= transfer
                allocation[i, j] -= transfer

    return remaining_demand, available_energy

def allocation(weights_df, preferences_df, producer_energy_data, subscriber_demand_data, rounds=5):
    """
    Allocation process for all timestamps.

    Args:
        weights_df (pd.DataFrame): weights dataframe
        preferences_df (pd.DataFrame): Preference dataframe
        producer_energy_data (pd.DataFrame): Supply dataframe
        subscriber_demand_data (pd.DataFrame): Demand dataframe

    Returns:
        pd.Dataframe: Unallocated energy per timestamp
    """
    
    num_timesteps = len(producer_energy_data)
    results = np.zeros((num_timesteps, 3))

    prod_ids = producer_energy_data.columns
    cons_ids = subscriber_demand_data.columns

    for t in range(num_timesteps):
        ae = producer_energy_data.loc[t].to_numpy(copy=True)
        rd = subscriber_demand_data.loc[t].to_numpy(copy=True)

        ae_series = pd.Series(ae, index=prod_ids)
        rd_series = pd.Series(rd, index=cons_ids)

        rd_t, ae_t = step(ae_series, rd_series, weights_df, preferences_df, rounds)
        results[t] = [t, rd_t.sum(), ae_t.sum()]

    return pd.DataFrame(results, columns=['timestamp', 'unsatisfied_demand', 'available_energy'])


def fitness(weights_df, preferences_df, producer_energy_data, subscriber_demand_data, rounds=5):
    """
    fitness function. How much energy stays unallocated given weights and preferences.

    Args:
        weights_df (pd.DataFrame): weights dataframe
        preferences_df (pd.DataFrame): Preference dataframe
        producer_energy_data (pd.DataFrame): Supply dataframe
        subscriber_demand_data (pd.DataFrame): Demand dataframe

    Returns:
        float: sum of all unallocated energy.
    """

    alloc_df = allocation(weights_df, preferences_df, producer_energy_data, subscriber_demand_data, rounds) 
    mask = (alloc_df['unsatisfied_demand'] > 0) | (alloc_df['available_energy'] > 0)
    alloc_df = alloc_df.loc[mask, ['unsatisfied_demand', 'available_energy']].sum()
    alloc_df['Total'] = alloc_df['unsatisfied_demand'] + alloc_df['available_energy']
    return alloc_df
