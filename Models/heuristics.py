import utils as ut
import numpy as np
import pandas as pd
import random
from dynamic_weights import linear_flow
from copy import deepcopy
import time
import math
import os


def evaluate_matrix(M: pd.DataFrame, excess_df: pd.DataFrame, deficit_df: pd.DataFrame, weights=None) -> float:
    """
    Evaluate total leftover (supplies + unmet demands) over all time periods
    using greedy allocation derived from optimized weights.

    Args:
        M pd.DataFrame          : preference data-frame
        excess_df pd.DataFrame  : dataframe with excess
        deficit_df pd.DataFrame : dataframe with deficit

    Returns:
        float: fitness score (total undistributed energy). Lower is better.
    """

    if weights is None:
        weights, unmet_demand, unused_supply = linear_flow(excess_df, deficit_df, M)


    try:
        fitness_score = ut.fitness(weights, M, excess_df, deficit_df)
        return fitness_score
    
    except Exception as e:
        print(f"Error calculating fitness with ut.fitness: {e}. Falling back to unmet+unused.")
        try:
            return unmet_demand + unused_supply
        except Exception as e:
            print(f"Error calculating using unmet_demand + unused_supply {e}. return infity")
        return np.inf


def generate_sparse_vector(n_producers: int, max_prefs: int) -> np.ndarray:
    """
    Produce ONE consumer row that satisfies all rules:

    • length  = n_producers
    • at most `max_prefs` non-zero entries
    • the non-zeros are the unique integers {1 … k} (k ≤ max_prefs)
    """
    row = np.zeros(n_producers, dtype=int)

    # how many producers this consumer actually ranks (1 … max_prefs)
    k = np.random.randint(1, max_prefs + 1)

    cols  = np.random.choice(n_producers, size=k, replace=False)
    ranks = np.random.permutation(np.arange(1, k + 1))     # unique 1…k
    row[cols] = ranks
    return row


def _swap_producer_prefs_in_row(row: pd.Series, n_prods: int) -> pd.Series:
    """
    Swap the preference values of two randomly chosen producers in *one* row.
    Keeps the two “slots” unique and ≤ n.
    """
    mutated = row.copy()
    col1, col2 = np.random.choice(n_prods, size=2, replace=False)
    mutated.iloc[col1], mutated.iloc[col2] = mutated.iloc[col2], mutated.iloc[col1]
    return mutated


def mutate_preference_matrix(individual: pd.DataFrame, 
                             mutation_rate: float = 0.05,
                             n_cons: int = 14, 
                             n_prods: int = 14,
                             non_zero_count: int = 5,
                             mutation_type: str = 'granular'
) -> pd.DataFrame:
    """
    Mutate **consumer rows** (NOT producer rows!) while preserving all constraints.
    """
    mutated = individual.copy()
    n_cons, n_prods = mutated.shape   # rows = consumers, columns = producers

    for r in range(n_cons):           # iterate over *rows*  <-- fixed
        if random.random() < mutation_rate:

            if mutation_type == "granular":
                mutated.iloc[r] = _swap_producer_prefs_in_row(mutated.iloc[r], n_prods)

            elif mutation_type == "row_replace":
                mutated.iloc[r] = generate_sparse_vector(n_prods, non_zero_count)

            else:
                raise ValueError(f"Unknown mutation type: {mutation_type}")

    return mutated


def crossover(parent1: pd.DataFrame, parent2: pd.DataFrame) -> pd.DataFrame:
    """
    Uniform crossover: for each *consumer row* choose which parent to copy.
    """
    child   = parent1.copy()
    n_cons  = len(child)              # rows

    for r in range(n_cons):           # rows, not columns  <-- fixed
        if random.random() < 0.5:
            child.iloc[r] = parent2.iloc[r]

    return child


# ======================================================
#                   GENETIC ALGORITHM
# ======================================================

def tournament_selection(population_with_fitness: list, k: int, num_winners: int) -> list:
    """
    Performs tournament selection on a population.

    Args:
        population_with_fitness (list): List of tuples (fitness, individual_df).
        k (int): Size of each tournament.
        num_winners (int): Number of individuals to select.

    Returns:
        list: Selected individuals (pd.DataFrame).
    """
    selected = []
    pop_size = len(population_with_fitness)
    for _ in range(num_winners):

        tournament_indices = random.sample(range(pop_size), k)
        tournament_participants = [population_with_fitness[i] for i in tournament_indices]

        winner = min(tournament_participants, key=lambda x: x[0])
        selected.append(winner[1])

    return selected


def run_genetic_algorithm(excess_df: pd.DataFrame,
                          deficit_df: pd.DataFrame,
                          generations: int = 100,
                          population_size: int = 50,
                          mutation_rate: float = 0.1,
                          crossover_rate: float = 0.8,
                          tournament_size: int = 3,
                          elitism_count: int = 1,
                          mutation_type: str = 'granular',
                          initial_non_zero: int = 5,
                          weights = None
                          ) -> tuple[pd.DataFrame, float, list[pd.DataFrame]]:
    """
    Runs the Genetic Algorithm to find the optimal preference matrix.

    Args:
        excess_df pd.DataFrame  : dataframe with excess
        deficit_df pd.DataFrame : dataframe with deficit
        generations (int, optional): number of generation. Defaults to 100.
        population_size (int, optional): number of individuals in a population. Defaults to 50.
        mutation_rate (float, optional): probability of mutation per individual row. Defaults to 0.1.
        crossover_rate (float, optional): probability of crossover between two parents. Defaults to 0.8.
        tournament_size (int, optional): size of the tournament for selection. Defaults to 3.
        elitism_count (int, optional): number of best individuals to carry over. Defaults to 1.
        mutation_type (str): Type of mutation ('granular' or 'row_replace'). Defaults to 'granular'.
        initial_non_zero (int): Number of non-zero preferences for initial population/row_replace mutation. Defaults to 5.


    Returns:
        tuple[pd.DataFrame, float, list[pd.DataFrame]]: best individual, its fitness and final population.
    """
    producers = excess_df.columns.tolist()
    consumers = deficit_df.columns.tolist()
    n_prods = len(producers)
    n_cons = len(consumers)

    # print("GA: Initializing population...")
    population = [ut.generate_preferences(producers, consumers, n=initial_non_zero)
                  for _ in range(population_size)]

    best_individual = None
    best_fitness = float('inf')

    for gen in range(generations):
        start_time = time.time()
        print(f"GA: Starting Generation {gen+1}/{generations}")

        population_with_fitness = []
        for i, ind in enumerate(population):
            fit = evaluate_matrix(ind, excess_df, deficit_df, weights)
            population_with_fitness.append((fit, ind))

        population_with_fitness.sort(key=lambda x: x[0])
        current_best_fit, current_best_ind = population_with_fitness[0]

        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_individual = current_best_ind.copy()
            # print(f"GA: New best fitness found: {best_fitness:.4f}")

        next_population = [ind.copy() for fit, ind in population_with_fitness[:elitism_count]]
        parents = tournament_selection(population_with_fitness, k=tournament_size,
                                        num_winners=population_size - elitism_count)

        while len(next_population) < population_size:

            if len(parents) < 2: 
                 p1 = random.choice(population_with_fitness)[1]
                 p2 = random.choice(population_with_fitness)[1]
            else:
                 p1, p2 = random.sample(parents, 2)

            child = deepcopy(p1)

            if random.random() < crossover_rate:
                child = crossover(p1, p2)

            child = mutate_preference_matrix(child, mutation_rate=mutation_rate,
                                              n_cons=n_cons,
                                              n_prods=n_prods,
                                              non_zero_count = initial_non_zero,
                                              mutation_type=mutation_type) 
            next_population.append(child)

        population = next_population
        end_time = time.time()
        # print(f"GA: Generation {gen+1} finished in {end_time - start_time:.2f} seconds. Best fitness: {best_fitness:.4f}")


    final_best_fitness = evaluate_matrix(best_individual, excess_df, deficit_df, weights)
    # print(f"GA: Final best fitness: {final_best_fitness:.4f}")
    best_weights, _, _ = linear_flow(excess_df, deficit_df, best_individual)

    return best_individual, final_best_fitness, best_weights



# ======================================================
#                STATIC GENETIC ALGORITHM
# ======================================================



def generate_initial_population_from_links(link_df: pd.DataFrame, population_size: int) -> list[pd.DataFrame]:
    """
    Generates initial population from a fixed binary link matrix.
    Each individual differs only in the assigned preference values at link=1 locations.
    """
    population = []
    for _ in range(population_size):
        ind = link_df.copy()
        for i, (_, row) in enumerate(link_df.iterrows()):
            ones = np.where(row == 1)[0]
            k = len(ones)
            values = np.random.permutation(np.arange(1, k+1)) if k > 0 else []
            ind.iloc[i, ones] = values
        population.append(ind)
    return population


def mutate_fixed_link_matrix(ind: pd.DataFrame, link_df: pd.DataFrame, mutation_rate: float = 0.1) -> pd.DataFrame:
    mutated = ind.copy()
    for i in range(mutated.shape[0]):
        if random.random() < mutation_rate:
            row = mutated.iloc[i]
            mask = link_df.iloc[i] == 1
            values = row[mask].values
            if len(values) >= 2:
                np.random.shuffle(values)
                mutated.iloc[i][mask] = values
    mutated.index = link_df.index
    mutated.columns = link_df.columns
    return mutated



def crossover_fixed_link_matrix(parent1: pd.DataFrame, parent2: pd.DataFrame, link_df: pd.DataFrame) -> pd.DataFrame:
    child = parent1.copy()

    for i in range(len(child)):
        row_mask = link_df.iloc[i] == 1
        if row_mask.sum() == 0:
            continue

        p1_vals = parent1.iloc[i][row_mask].values
        p2_vals = parent2.iloc[i][row_mask].values

        selected = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(p1_vals, p2_vals)]
        child.iloc[i][row_mask] = selected

    child.index = link_df.index
    child.columns = link_df.columns
    return child


def run_fixed_link_ga(link_df: pd.DataFrame,
                      weights: pd.DataFrame,
                      excess_df: pd.DataFrame,
                      deficit_df: pd.DataFrame,
                      generations: int = 100,
                      population_size: int = 50,
                      mutation_rate: float = 0.1,
                      crossover_rate: float = 0.8,
                      tournament_size: int = 3,
                      elitism_count: int = 1
                      ) -> tuple[pd.DataFrame, float]:
    """
    GA to optimize preference values on a fixed link structure.
    """
    population = generate_initial_population_from_links(link_df, population_size)

    best_individual = None
    best_fitness = float('inf')

    for gen in range(generations):
        population_with_fitness = []
        for ind in population:
            try:
                fitness = ut.fitness(weights, ind, excess_df, deficit_df)
            except Exception as e:
                print(f"Fitness error: {e}")
                fitness = np.inf
            population_with_fitness.append((fitness, ind))

        population_with_fitness.sort(key=lambda x: x[0])
        current_best_fit, current_best_ind = population_with_fitness[0]

        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_individual = current_best_ind.copy()

        next_population = [ind.copy() for fit, ind in population_with_fitness[:elitism_count]]
        parents = tournament_selection(population_with_fitness, k=tournament_size,
                                       num_winners=population_size - elitism_count)

        while len(next_population) < population_size:
            if len(parents) < 2:
                p1 = random.choice(population_with_fitness)[1]
                p2 = random.choice(population_with_fitness)[1]
            else:
                p1, p2 = random.sample(parents, 2)

            if random.random() < crossover_rate:
                child = crossover_fixed_link_matrix(p1, p2, link_df)  # <-- PASS LINK_DF HERE
            else:
                child = p1.copy()

            child = mutate_fixed_link_matrix(child, link_df, mutation_rate)
            next_population.append(child)

        population = next_population

    return best_individual, best_fitness




# -------------------------------------------------------------------
#  SIMULATED ANNEALING 
# -------------------------------------------------------------------

def simulated_annealing(excess_df: pd.DataFrame,
                        deficit_df: pd.DataFrame,
                        iters: int = 5000,
                        init_temp: float = 100.0,
                        decay: float = 0.95,
                        mutation_type: str = 'granular',
                        mutation_rate_sa: float = 0.9,
                        initial_non_zero: int = 5,
                        weights=None) -> tuple[pd.DataFrame, float]:

    producers = excess_df.columns.tolist()
    consumers = deficit_df.columns.tolist()
    n_prods   = len(producers)
    n_cons    = len(consumers)

    M        = ut.generate_preferences(producers, consumers, n=initial_non_zero)
    cur_fit  = evaluate_matrix(M, excess_df, deficit_df, weights)
    best_M   = M.copy()
    best_fit = cur_fit
    T        = init_temp

    # print(f"SA: Initial fitness {cur_fit:.4f} | T = {init_temp} | decay = {decay}")

    for k in range(iters):
        M_neighbor = M.copy()
        row_idx = random.randrange(n_cons)
        if random.random() < mutation_rate_sa:
            if mutation_type == 'granular':
                M_neighbor.iloc[row_idx] = _swap_producer_prefs_in_row(
                                                M_neighbor.iloc[row_idx], n_prods)
            elif mutation_type == 'row_replace':
                M_neighbor.iloc[row_idx] = generate_sparse_vector(
                                                n_prods, non_zero_count=initial_non_zero)
            else:
                raise ValueError("Unknown mutation type for SA")

        fit_neighbor = evaluate_matrix(M_neighbor, excess_df, deficit_df, weights)
        accept = (fit_neighbor < cur_fit) or \
                 (random.random() < math.exp((cur_fit - fit_neighbor) / T))

        if accept:
            M, cur_fit = M_neighbor, fit_neighbor

        if cur_fit < best_fit:
            best_M, best_fit = M.copy(), cur_fit

        T *= decay

        if (k + 1) % max(1, iters // 10) == 0 or k == iters - 1:
            print(f"SA {k+1:>6}/{iters} | best {best_fit:.4f} | cur {cur_fit:.4f} | T={T:.6f}")

    print(f"SA: Final best fitness {best_fit:.4f}")
    return best_M, best_fit


# -------------------------------------------------------------------
#  HILL CLIMBING 
# -------------------------------------------------------------------

def hill_climb(excess_df: pd.DataFrame,
               deficit_df: pd.DataFrame,
               iters: int = 5000,
               mutation_type: str = 'granular',
               initial_non_zero: int = 5,
               weights=None) -> tuple[pd.DataFrame, float]:

    producers = excess_df.columns.tolist()
    consumers = deficit_df.columns.tolist()
    n_prods   = len(producers)
    n_cons    = len(consumers)

    M        = ut.generate_preferences(producers, consumers, n=initial_non_zero)
    best_M   = M.copy()
    best_fit = evaluate_matrix(M, excess_df, deficit_df, weights)

    print(f"HC: Initial fitness {best_fit:.4f}")

    for k in range(iters):
        M_neighbor = best_M.copy()

        # ----------- ROW-ORIENTED mutation ---------------------------
        row_idx = random.randrange(n_cons)
        if mutation_type == 'granular':
            M_neighbor.iloc[row_idx] = _swap_producer_prefs_in_row(
                                            M_neighbor.iloc[row_idx], n_prods)
        elif mutation_type == 'row_replace':
            M_neighbor.iloc[row_idx] = generate_sparse_vector(
                                            n_prods, non_zero_count=initial_non_zero)
        else:
            raise ValueError("Unknown mutation type for HC")

        fit_neighbor = evaluate_matrix(M_neighbor, excess_df, deficit_df, weights)

        if fit_neighbor < best_fit:
            best_M, best_fit = M_neighbor.copy(), fit_neighbor

        if (k + 1) % max(1, iters // 10) == 0 or k == iters - 1:
            print(f"HC {k+1:>6}/{iters} | best {best_fit:.4f}")

    print(f"HC: Final best fitness {best_fit:.4f}")
    return best_M, best_fit



def _assert_invariants(pref_df: pd.DataFrame, n: int) -> None:
    """
    Raise AssertionError if any consumer row violates:
      • ≤ n non-zero entries
      • unique ranks 1…n (no 0 among ranks)
    """
    for row in pref_df.itertuples(index=False):
        vals   = np.asarray(row)
        nz     = vals[vals > 0]
        assert nz.size <= n,           "More than n producers ranked in a row"
        assert nz.size == len(set(nz)), "Duplicate rank detected in a row"
        assert (1 <= nz).all() and (nz <= n).all(), "Rank outside 1…n"


def _run_algorithm_smoke_tests() -> None:
    """
    Generates tiny random input data, runs GA / SA / HC for a few iterations,
    and checks:
      1. output matrix shape = consumers x producers
      2. each consumer row obeys the invariants
    """

    # ---------- reproducible toy data --------------------------------
    np.random.seed(1)
    random.seed(1)

    producers  = [f"P{i}" for i in range(6)]   # 6 producers
    consumers  = [f"C{j}" for j in range(4)]   # 4 consumers
    n          = 3                             # max ranks / consumer
    periods    = 12                            # time steps

    excess_df  = pd.DataFrame(
        np.random.rand(periods, len(producers)),
        columns=producers
    )
    deficit_df = pd.DataFrame(
        np.random.rand(periods, len(consumers)),
        columns=consumers
    )

    # ---------- run Genetic Algorithm (very small instance) ----------
    print("\n[TEST] Genetic Algorithm …")
    best_ga, fit_ga, _ = run_genetic_algorithm(
        excess_df, deficit_df,
        generations=2,
        population_size=6,
        mutation_rate=0.5,
        crossover_rate=0.7,
        tournament_size=2,
        elitism_count=1,
        mutation_type="granular",
        initial_non_zero=n,
    )
    assert best_ga.shape == (len(consumers), len(producers))
    _assert_invariants(best_ga, n)
    print("  ↳ GA invariant check ✅")

    # ---------- run Simulated Annealing ------------------------------
    print("[TEST] Simulated Annealing …")
    best_sa, fit_sa = simulated_annealing(
        excess_df, deficit_df,
        iters=50,
        init_temp=10.0,
        decay=0.9,
        mutation_type="granular",
        mutation_rate_sa=0.8,
        initial_non_zero=n,
    )
    assert best_sa.shape == (len(consumers), len(producers))
    _assert_invariants(best_sa, n)
    print("  ↳ SA invariant check ✅")

    # ---------- run Hill-Climbing ------------------------------------
    print("[TEST] Hill Climbing …")
    best_hc, fit_hc = hill_climb(
        excess_df, deficit_df,
        iters=50,
        mutation_type="granular",
        initial_non_zero=n,
    )
    assert best_hc.shape == (len(consumers), len(producers))
    _assert_invariants(best_hc, n)
    print("  ↳ HC invariant check ✅")

    # ---------- all good ---------------------------------------------
    print("\n✅  All optimisation-algorithm smoke-tests passed.\n")




def check_static():
    # Example inputs
    link_df = pd.DataFrame([
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 0]
    ], columns=["P1", "P2", "P3", "P4"], index=["C1", "C2", "C3"])

    # Dummy time series data — doesn't affect dummy fitness
    excess_df = pd.DataFrame(np.random.rand(3, 4), columns=link_df.columns)
    deficit_df = pd.DataFrame(np.random.rand(3, 3), columns=link_df.index)

    # Random weights — not used in dummy fitness
    weights = pd.DataFrame(np.random.rand(3, 4), index=link_df.index, columns=link_df.columns)

    # Run the algorithm
    best_M, best_fit = run_fixed_link_ga(
        link_df=link_df,
        weights=weights,
        excess_df=excess_df,
        deficit_df=deficit_df,
        generations=1,
        population_size=5,
        mutation_rate=0.2,
        crossover_rate=0.9
    )

    print("=== Best Preference Matrix ===")
    print(best_M)
    print(f"Fitness: {best_fit}")

    check_structure(best_M, link_df)

# Checks
def check_structure(M, L):
    for i in range(M.shape[0]):
        m_row = M.iloc[i]
        l_row = L.iloc[i]
        active_idxs = np.where(l_row == 1)[0]
        zero_idxs = np.where(l_row == 0)[0]

        values = m_row.iloc[active_idxs].values
        assert sorted(values) == list(range(1, len(values) + 1)), f"Row {i} incorrect: {values}"

        assert (m_row.iloc[zero_idxs] == 0).all(), f"Row {i} has nonzero where link = 0"

    print("✅ Structure check passed.")




if __name__ == "__main__":
    _run_algorithm_smoke_tests()
    check_static()