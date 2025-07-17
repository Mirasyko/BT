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

def prepare_data(excess_monthly, deficit_monthly, month, start=0, end=10, all_data=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for testing. Returns excess and deficit of specified length (end - start),
    filtered to only include time steps with non-zero total supply and demand.

    Args:
        excess_monthly (dict[int, pd.DataFrame]): Monthly excess data.
        deficit_monthly (dict[int, pd.DataFrame]): Monthly deficit data.
        month (int): Index of month.
        start (int, optional): Start index in the filtered data. Defaults to 0.
        end (int, optional): End index in the filtered data. Defaults to 10.
        all_data (bool, optional): If True, do not filter out zero rows. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Filtered excess and deficit of shape (end - start, n_entities)
    """
    # Load raw data for given month
    excess_df = excess_monthly[month].copy()
    deficit_df = deficit_monthly[month].copy()

    # Clean: convert to numeric and drop timestamp
    excess_df = excess_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    deficit_df = deficit_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    excess_df = excess_df.drop(columns='timestamp').reset_index(drop=True)
    deficit_df = deficit_df.drop(columns='timestamp').reset_index(drop=True)

    if all_data:
        # Return raw slice (no filtering)
        end = min(end, len(excess_df))
        return excess_df.iloc[start:end], deficit_df.iloc[start:end]

    # Identify valid time steps (non-zero excess and non-zero deficit)
    valid_idx = (excess_df.sum(axis=1) > 0) & (deficit_df.sum(axis=1) > 0)

    # Filter only those time steps
    filtered_excess = excess_df[valid_idx].reset_index(drop=True)
    filtered_deficit = deficit_df[valid_idx].reset_index(drop=True)

    # Select slice of length = end - start
    size = end - start
    if size > len(filtered_excess):
        raise ValueError(f"Not enough valid time steps ({len(filtered_excess)}) to return slice of size {size}.")

    return filtered_excess.iloc[start:start+size], filtered_deficit.iloc[start:start+size]



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


def _floor_two_decimals(value: float) -> float:
    """Round *down* to two decimal places (0.01-kWh granularity)."""
    return np.floor(value * 100000.0) / 100000.0

###############################################################################
# Core allocation logic                                                      #
###############################################################################

def step(
    available_energy: pd.Series,
    remaining_demand: pd.Series,
    weights_df: pd.DataFrame,
    preferences_df: pd.DataFrame,
    rounds: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Single-timestep allocation following the corrected EDC algorithm."""

    ae = available_energy.copy()
    rd = remaining_demand.copy()
    producers = ae.index
    consumers = rd.index
    weights = weights_df.loc[consumers, producers].to_numpy()
    preferences = preferences_df.loc[consumers, producers].to_numpy()
    #MAX_PRIORITY

    used_by_consumer: dict[str, set[str]] = {c: set() for c in consumers}

    for _ in range(rounds):
        if rd.sum() <= 0 or ae.sum() <= 0:
            break

        # Track how much energy each producer will give in this round
        producer_allocations = {p: 0.0 for p in producers}

        # Track how much each consumer receives from each producer in this round
        for c_idx, consumer in enumerate(consumers):
            if rd.iloc[c_idx] <= 0 or len(used_by_consumer[consumer]) >= 5:
                continue

            pref_row = preferences[c_idx]
            ordered_p_idx = np.argsort(-pref_row)  # highest priority first

            for p_rank in ordered_p_idx:
                producer = producers[p_rank]
                if pref_row[p_rank] <= 0:
                    continue

                if ae.loc[producer] <= 0:
                    continue

                if producer in used_by_consumer[consumer]:
                    continue

                alloc_key = weights[c_idx, p_rank]
                if alloc_key <= 0:
                    continue

                potential_share = alloc_key * ae.loc[producer]
                share = min(rd.iloc[c_idx], potential_share)
                share = _floor_two_decimals(share)
                if share <= 0:
                    continue

                rd.iloc[c_idx] -= share
                producer_allocations[producer] += share
                used_by_consumer[consumer].add(producer)

                if rd.iloc[c_idx] <= 0:
                    break

        for producer, total_given in producer_allocations.items():
            ae.loc[producer] -= total_given
            ae.loc[producer] = max(ae.loc[producer], 0.0)

    return rd, ae

###############################################################################
# Multi-timestep wrapper                                                     #
###############################################################################

def allocation(
    weights_df: pd.DataFrame,
    preferences_df: pd.DataFrame,
    producer_energy_data: pd.DataFrame,
    subscriber_demand_data: pd.DataFrame,
    rounds: int = 5,
) -> pd.DataFrame:
    """Run the allocation routine for every 15-minute timestep.

    The returned DataFrame mirrors your original signature, while using the
    *consumer-first* EDC logic internally.

    Parameters
    ----------
    weights_df, preferences_df : see :func:`step_`.
    producer_energy_data : pd.DataFrame
        Time-indexed supply in kWh for every producer.
    subscriber_demand_data : pd.DataFrame
        Time-indexed demand in kWh for every consumer.
    rounds : int, default ``5``
        Passed straight to :func:`step_` as the *maximum* number of iterations
        per timestep.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``unsatisfied_demand``, ``available_energy``
    """
    if not producer_energy_data.index.equals(subscriber_demand_data.index):
        raise ValueError("Supply and demand dataframes must share the same index (timestamps!).")

    num_timesteps = len(producer_energy_data)
    results = np.zeros((num_timesteps, 3), dtype=float)

    for t, (ts, ae_row) in enumerate(producer_energy_data.iterrows()):
        rd_row = subscriber_demand_data.loc[ts]

        rd_after, ae_after = step(
            ae_row.astype(float),
            rd_row.astype(float),
            weights_df,
            preferences_df,
            rounds,
        )

        results[t] = [ts, rd_after.sum(), ae_after.sum()]

    return pd.DataFrame(results, columns=["timestamp", "unsatisfied_demand", "available_energy"]).set_index("timestamp")


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
    #mask = (alloc_df['unsatisfied_demand'] > 0) | (alloc_df['available_energy'] > 0)
    #alloc_df = alloc_df.loc[mask, ['unsatisfied_demand', 'available_energy']].sum()
    alloc_df['Total'] = alloc_df['unsatisfied_demand'] + alloc_df['available_energy']
    return alloc_df['Total'].sum()




def static_step(
    available_energy: pd.Series,
    remaining_demand: pd.Series,
    weights_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Single-timestep static allocation with deferred energy update.

    Energy is distributed based on allocation keys, and producers' available energy
    is only updated after all consumer allocations are computed.

    Parameters
    ----------
    available_energy : pd.Series
        Energy available from each producer (index = producer IDs).
    remaining_demand : pd.Series
        Energy requested by each consumer (index = consumer IDs).
    weights_df : pd.DataFrame
        Allocation keys (0-1 range). Rows = consumer IDs, columns = producer IDs.

    Returns
    -------
    tuple(pd.Series, pd.Series)
        (remaining_demand_after_allocation, remaining_available_energy)
    """
    ae = available_energy.copy().astype(float)
    rd = remaining_demand.copy().astype(float)
    producers = ae.index
    consumers = rd.index

    # Track how much energy each producer gives in this step
    producer_allocations = {p: 0.0 for p in producers}

    for consumer in consumers:
        if rd[consumer] <= 0:
            continue

        for producer in producers:
            if rd[consumer] <= 0 or ae[producer] <= 0:
                break

            alloc_key = weights_df.loc[consumer, producer]
            if alloc_key <= 0:
                continue

            # Use producer's current available energy for this round
            potential_share = alloc_key * ae[producer]
            share = min(rd[consumer], potential_share)
            share = _floor_two_decimals(share)
            if share <= 0:
                continue

            rd[consumer] -= share
            producer_allocations[producer] += share

    # Apply producer allocations after all consumers are processed
    for producer, total_shared in producer_allocations.items():
        ae[producer] -= total_shared
        ae[producer] = max(ae[producer], 0.0)

    return rd, ae



def static_allocation(
    weights_df: pd.DataFrame,
    producer_energy_data: pd.DataFrame,
    subscriber_demand_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run static allocation for every 15-minute timestep.

    For each timestamp:
      1. Take supply row (producer_energy_data) and demand row (subscriber_demand_data).
      2. Call `step` to allocate energy in one pass.
      3. Record total unsatisfied demand and leftover supply.

    Parameters
    ----------
    weights_df : pd.DataFrame
        Allocation keys (0-1 range). Rows = consumer IDs, columns = producer IDs.
    producer_energy_data : pd.DataFrame
        Time-indexed supply in kWh for every producer (index = timestamps, cols = producer IDs).
    subscriber_demand_data : pd.DataFrame
        Time-indexed demand in kWh for every consumer (index = timestamps, cols = consumer IDs).

    Returns
    -------
    pd.DataFrame
        Indexed by timestamp, with columns:
          - unsatisfied_demand: sum of remaining demand after allocation.
          - available_energy: sum of remaining supply after allocation.
    """
    if not producer_energy_data.index.equals(subscriber_demand_data.index):
        raise ValueError("Supply and demand dataframes must share the same index (timestamps!).")

    timestamps = producer_energy_data.index
    results = np.zeros((len(timestamps), 2), dtype=float)

    for i, ts in enumerate(timestamps):
        ae_row = producer_energy_data.loc[ts]
        rd_row = subscriber_demand_data.loc[ts]

        rd_after, ae_after = static_step(ae_row, rd_row, weights_df)
        results[i, 0] = rd_after.sum()
        results[i, 1] = ae_after.sum()

    out = pd.DataFrame(
        results,
        index=timestamps,
        columns=["unsatisfied_demand", "available_energy"],
    )
    return out


def static_fitness(
    weights_df: pd.DataFrame,
    producer_energy_data: pd.DataFrame,
    subscriber_demand_data: pd.DataFrame,
) -> float:
    """
    Compute total undistributed energy (unsatisfied demand + leftover supply) 
    across all timesteps for given weights.
    """
    alloc_df = static_allocation(weights_df, producer_energy_data, subscriber_demand_data)
    # Total at each timestep = unsatisfied_demand + available_energy
    total = (alloc_df["unsatisfied_demand"] + alloc_df["available_energy"]).sum()
    return total




def generate_preferences(producers, consumers, n: int = 5) -> pd.DataFrame:
    """
    Create a (consumers × producers) preference matrix.

    Parameters
    ----------
    producers : list[str]
        Column labels.
    consumers : list[str]
        Row labels.
    n : int, default 5
        Maximum number of producers a consumer can rank (must be ≥ 1).

    Returns
    -------
    pd.DataFrame
        Integer preference scores where 0 means “not ranked”.
        Shape = (len(consumers), len(producers)).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    # Never sample more producers than exist.
    n = min(n, len(producers))

    rows = {}
    for consumer in consumers:
        # Choose up to n producers for this consumer (no duplicates).
        chosen = np.random.choice(producers, size=n, replace=False)
        # Randomly permute the ranks 1..n so each appears at most once.
        ranks = np.random.permutation(np.arange(1, n + 1))

        # Build a row: start with zeros, then fill the chosen producers.
        row = dict.fromkeys(producers, 0)
        row.update(zip(chosen, ranks))
        rows[consumer] = row

    # Orient as consumers (index) × producers (columns).
    return pd.DataFrame.from_dict(rows, orient="index", columns=producers)


def create_preference_tensor(preference_df: pd.DataFrame) -> np.ndarray:
    """
    Convert a consumer x producer preference matrix into a 3-D tensor.

    Parameters
    ----------
    preference_df : pd.DataFrame
        Rows = consumers, columns = producers, cells = rank
        (0 means “not ranked”, 1 = best, 2 = next best, …).

    Returns
    -------
    np.ndarray
        Shape = (n_consumers, n_producers, L + 1) where
        L = highest rank that appears in the DataFrame.
        Slice [:, :, 0] is always 0 (unranked).
        For k > 0, slice [:, :, k] is a binary mask indicating
        which consumer–producer pairs have rank (L − k + 1), so the
        *last* slice holds rank 1 (top preference).
    """
    # dimensions
    n_consumers, n_producers = preference_df.shape
    max_level = int(preference_df.to_numpy().max())

    # tensor initialised with zeros
    tensor = np.zeros((n_consumers, n_producers, max_level + 1), dtype=int)

    # populate: iterate over consumers (rows) and producers (cols)
    for i in range(n_consumers):
        for j in range(n_producers):
            pref = int(preference_df.iat[i, j])
            if pref > 0:                       # skip unranked (=0)
                k = max_level - pref + 1       # reverse order, leave 0th slice empty
                tensor[i, j, k] = 1

    return tensor


def preference_matrix_from_tensor(
    pref_tensor: np.ndarray,
    producers: list[str] | None = None,
    consumers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Inverse of `create_preference_tensor` (consumers × producers × levels).

    Parameters
    ----------
    pref_tensor : np.ndarray
        Boolean/int tensor with shape (n_consumers, n_producers, L + 1)
        produced by `create_preference_tensor`.  Slice 0 is “unranked”,
        last slice holds rank 1, etc.
    producers : list[str] | None
        Optional column labels; auto-generated if omitted.
    consumers : list[str] | None
        Optional row labels; auto-generated if omitted.

    Returns
    -------
    pd.DataFrame
        Shape = (n_consumers, n_producers) containing integer ranks
        (0 = unranked, 1 = top choice, …).
    """
    n_consumers, n_producers, levels = pref_tensor.shape
    max_level = levels - 1

    # Initialise matrix with zeros (unranked)
    pref_mat = np.zeros((n_consumers, n_producers), dtype=int)

    # Recover ranks
    for i in range(n_consumers):
        for j in range(n_producers):
            idx = np.flatnonzero(pref_tensor[i, j])
            if idx.size:
                l = idx[0]                      # first (only) level with a 1
                pref_mat[i, j] = max_level - l + 1

    # Default labels if none supplied
    if consumers is None:
        consumers = [f"C{i}" for i in range(n_consumers)]
    if producers is None:
        producers = [f"P{j}" for j in range(n_producers)]

    return pd.DataFrame(pref_mat, index=consumers, columns=producers)



def toy_dataset():
    excess_df = pd.DataFrame(
        [[100.0, 100.0], [40, 70], [30, 80]],
        columns=["P0", "P1"]
    )
    deficit_df = pd.DataFrame(
        [[50., 50., 50.], [60, 20, 40], [30, 60, 20]],
        columns=["C0", "C1", "C2"]
    )

    consumers = ["C0", "C1", "C2"]
    producers = ["P0", "P1"]

    excess = excess_df.values[0]
    deficit = deficit_df.values[0]

    preference_df = generate_preferences(producers, consumers, 2)
    preference_df = preference_df.loc[producers,consumers]

    return excess_df, deficit_df, preference_df

def real_dataset(month, start, end, all_data):

    data_dir = '/home/miro/Bachelor/BT/data/outputs/'
    excess_monthly, deficit_monthly = load_excess_deficit(data_dir)
    excess, deficit = prepare_data(excess_monthly, deficit_monthly, month, start, end, all_data)
    producers = excess.columns
    consumers = deficit.columns
    preference_df = generate_preferences(producers, consumers)
    preference_df = preference_df.loc[producers,consumers]

    return excess, deficit, preference_df

def test_dataset():
    excess_df = pd.DataFrame(
        [[100, 100]],#, [40, 70], [30, 80]],
        index=[0, 1, 2],
        columns=["P0", "P1"]
    )
    deficit_df = pd.DataFrame(
        [[50., 50., 50.]],# [60, 20, 40], [30, 60, 20]],
        index=[0, 1, 2],
        columns=["C0", "C1", "C2"]
    )
    preference_df = pd.DataFrame(
        [[0, 5],
         [5, 3],
         [4, 4]],
        index=["C0", "C1", "C2"],
        columns=["P0", "P1"]
    )

    weights_df = pd.DataFrame(
        [[0.0, 1/2],
         [2/3, 1/6],
         [1/3, 1/3]],
         index=["C0", "C1", "C2"],
         columns=["P0", "P1"]
    )

    return excess_df, deficit_df, preference_df, weights_df

def _run_basic_tests() -> None:
    """
    Quick self-test for generate_preferences, create_preference_tensor,
    and preference_matrix_from_tensor.

    Verifies:
      1. Every supplied consumer & producer appears in the DataFrame.
      2. Each consumer ranks ≤ n producers and each rank 1…n is unique.
      3. DataFrame → tensor → DataFrame round-trip is loss-free.

    Raises
    ------
    AssertionError
        If any invariant is violated.
    """

    # ---------- reproducible toy data ---------------------------------
    np.random.seed(42)
    producers = [f"P{i}" for i in range(6)]   # 6 producers
    consumers = [f"C{j}" for j in range(4)]   # 4 consumers
    n = 3                                     # max ranks per consumer

    # ---------- 1. generate & basic shape checks -----------------------
    df = generate_preferences(producers, consumers, n)

    assert set(df.index)   == set(consumers), "ERROR Missing consumer(s) in index"
    assert set(df.columns) == set(producers), "ERROR Missing producer(s) in columns"

    # ---------- 2. per-consumer rank rules -----------------------------
    for c in consumers:
        ranks = df.loc[c]
        non_zero = ranks[ranks > 0]

        assert len(non_zero) <= n,              f"ERROR {c} ranks more than {n} producers"
        assert non_zero.is_unique,             f"ERROR {c} has duplicate rank values"
        assert (non_zero >= 1).all() and (non_zero <= n).all(), \
            f"ERROR {c} has rank outside 1…{n}"

    # ---------- 3. round-trip through tensor ---------------------------
    tensor  = create_preference_tensor(df)
    df_back = preference_matrix_from_tensor(tensor,
                                            producers=producers,
                                            consumers=consumers)

    pd.testing.assert_frame_equal(df.sort_index(axis=0).sort_index(axis=1),
                                  df_back.sort_index(axis=0).sort_index(axis=1),
                                  check_dtype=False)

    # ---------- all good! ----------------------------------------------
    print("✅ All basic preference-generation tests passed.")



if __name__ == "__main__":
    _run_basic_tests()
