import utils as ut
import numpy as np
import pandas as pd
import random
from weights_optimization_model import optimize_weights
from copy import deepcopy
import time
import math


def generate_sparse_vector(n, non_zero_count=5):
    """
    generates row of an matrix for mutation of preference. 

    Args:
        n (int): lenght of the vector
        non_zero_count (int, optional): number of preferences. Defaults to 5.

    Returns:
        np.array: resulting vector.
    """
    values = [i for i in range(non_zero_count)]
    vec = np.zeros(n, dtype=int)
    indices = np.random.choice(n, size=non_zero_count, replace=False)
    vec[indices] = np.random.choice(list(values), size=non_zero_count, replace=False)
    return vec


def _swap_producer_prefs_in_row(row: pd.Series, n_prods: int) -> pd.Series:
    """
    Performs a granular mutation by swapping the preference values
    of two random producers within a single consumer's preference row.

    Args:
        row (pd.Series): A single row (preferences for one consumer).
        n_prods (int): Number of producers.

    Returns:
        pd.Series: The mutated row.
    """
    mutated_row = row.copy()
    p1, p2 = np.random.choice(n_prods, size=2, replace=False)
    mutated_row.iloc[p1], mutated_row.iloc[p2] = mutated_row.iloc[p2], mutated_row.iloc[p1]

    return mutated_row

def mutate_preference_matrix(individual: pd.DataFrame, 
                             mutation_rate: float = 0.05,
                             n_cons: int = 14, 
                             n_prods: int = 14,
                             non_zero_count: int = 5,
                             mutation_type: str = 'granular') -> pd.DataFrame:
    """
    Mutates a preference matrix.

    Args:
        individual (pd.DataFrame): Preference matrix.
        mutation_rate (float, optional): Probability of mutation per consumer row. Defaults to 0.05.
        n_cons (int): Number of consumers.
        n_prods (int): Number of producers.
        non_zero_count (int, optional): number of preferences. Defaults to 5.
        mutation_type (str): Type of mutation ('granular' or 'row_replace').
                             'granular' swaps preferences of two producers in a row.
                             'row_replace' replaces the entire row with a new sparse vector.

    Returns:
        pd.DataFrame: Mutated preference matrix.
    """
    mutated_individual = individual.copy()
    for i in range(n_cons):
        if random.random() < mutation_rate:
            
            if mutation_type == 'granular':
                 mutated_individual.iloc[i] = _swap_producer_prefs_in_row(mutated_individual.iloc[i], n_prods)
            
            elif mutation_type == 'row_replace':
                 mutated_individual.iloc[i] = generate_sparse_vector(n_prods, non_zero_count)
            
            else:
                raise ValueError(f"Unknown mutation type: {mutation_type}")
    return mutated_individual


def evaluate_matrix(M: pd.DataFrame, excess_df: pd.DataFrame, deficit_df: pd.DataFrame) -> float:
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

    weights, unmet_demand, unused_supply = optimize_weights(excess_df, deficit_df, M)

    try:
         fitness_score = ut.fitness(weights, M, excess_df, deficit_df)['Total']
         return fitness_score
    except Exception as e:
         print(f"Error calculating fitness with ut.fitness: {e}. Falling back to unmet+unused.")
         return unmet_demand + unused_supply


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
                          initial_non_zero: int = 5
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

    print("GA: Initializing population...")
    population = [ut.generate_preferences(producers, consumers, n=initial_non_zero)
                  for _ in range(population_size)]

    best_individual = None
    best_fitness = float('inf')

    for gen in range(generations):
        start_time = time.time()
        print(f"GA: Starting Generation {gen+1}/{generations}")

        population_with_fitness = []
        for i, ind in enumerate(population):
            fit = evaluate_matrix(ind, excess_df, deficit_df)
            population_with_fitness.append((fit, ind))

        population_with_fitness.sort(key=lambda x: x[0])
        current_best_fit, current_best_ind = population_with_fitness[0]

        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_individual = current_best_ind.copy()
            print(f"GA: New best fitness found: {best_fitness:.4f}")

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
                child = crossover(p1, p2, n_cons=n_cons)

            child = mutate_preference_matrix(child, mutation_rate=mutation_rate,
                                              n_cons=n_cons,
                                              n_prods=n_prods,
                                              non_zero_count = initial_non_zero,
                                              mutation_type=mutation_type) 
            next_population.append(child)

        population = next_population
        end_time = time.time()
        print(f"GA: Generation {gen+1} finished in {end_time - start_time:.2f} seconds. Best fitness: {best_fitness:.4f}")


    final_best_fitness = evaluate_matrix(best_individual, excess_df, deficit_df)
    print(f"GA: Final best fitness: {final_best_fitness:.4f}")

    return best_individual, final_best_fitness, population

def crossover(parent1: pd.DataFrame, parent2: pd.DataFrame, n_cons: int) -> pd.DataFrame:
    """
    Performs crossover by swapping rows between two parent matrices.

    Args:
        parent1 (pd.DataFrame): preference matrix 1
        parent2 (pd.DataFrame): preference matrix 2
        n_cons (int): number of consumers (rows)

    Returns:
        pd.DataFrame: child preference matrix
    """
    child = deepcopy(parent1)
    for i in range(n_cons):
        if random.random() < 0.5:
            child.iloc[i] = parent2.iloc[i]
    return child


# ======================================================
#                SIMULATED ANNEALING
# ======================================================

def simulated_annealing(excess_df: pd.DataFrame,
                        deficit_df: pd.DataFrame,
                        iters: int = 5000,
                        init_temp: float = 100.0,
                        decay: float = 0.95,
                        mutation_type: str = 'granular',
                        mutation_rate_sa: float = 0.9,
                        initial_non_zero: int = 5) -> tuple[pd.DataFrame, float]:
    """
    Runs the Simulated Annealing algorithm.

    Args:
        excess_df pd.DataFrame  : dataframe with excess
        deficit_df pd.DataFrame : dataframe with deficit
        iters (int, optional): number of iterations. Defaults to 5000.
        init_temp (float, optional): initial temperature. Defaults to 100.0.
        decay (float, optional): temperature decay rate per iteration. Defaults to 0.95.
        mutation_type (str): Type of mutation for generating neighbors ('granular' or 'row_replace'). Defaults to 'granular'.
                             mutation_rate_sa controls the probability of mutating *a* row in a step.
        initial_non_zero (int): Number of non-zero preferences for initial solution/row_replace mutation. Defaults to 5.


    Returns:
        tuple[pd.DataFrame, float]: returns best preference data-frame and its score
    """
    producers = excess_df.columns.tolist()
    consumers = deficit_df.columns.tolist()
    n_prods = len(producers)
    n_cons = len(consumers)

    M = ut.generate_preferences(producers, consumers, n=initial_non_zero)
    cur_fit = evaluate_matrix(M, excess_df, deficit_df)

    best_M = M.copy()
    best_fit = cur_fit

    T_temp = init_temp

    print(f"SA: Initial fitness: {cur_fit:.4f}, Initial Temp: {init_temp}, Decay: {decay}")

    for k in range(iters):

        M_neighbor = M.copy()
        row_to_mutate_index = random.randrange(n_cons)

        if random.random() < mutation_rate_sa:
        
            if mutation_type == 'granular':
                M_neighbor.iloc[row_to_mutate_index] = _swap_producer_prefs_in_row(M_neighbor.iloc[row_to_mutate_index], n_prods)
            
            elif mutation_type == 'row_replace':
                M_neighbor.iloc[row_to_mutate_index] = generate_sparse_vector(n_prods, non_zero_count=initial_non_zero)

        fit_neighbor = evaluate_matrix(M_neighbor, excess_df, deficit_df)

        if fit_neighbor < cur_fit:
            M, cur_fit = M_neighbor, fit_neighbor

        else:
            delta = cur_fit - fit_neighbor
            acceptance_prob = math.exp(delta / T_temp)
            if random.random() < acceptance_prob:
                M, cur_fit = M_neighbor, fit_neighbor

        if cur_fit < best_fit:
            best_M, best_fit = M.copy(), cur_fit
        
        T_temp *= decay

        if (k+1) % (iters // 10) == 0 or k == iters - 1:
            print(f"SA: Iter {k+1}/{iters} — Current best: {best_fit:.4f}, Current fit: {cur_fit:.4f}, T={T_temp:.6f}")

    print(f"SA: Final best fitness: {best_fit:.4f}")
    return best_M, best_fit


# ======================================================
#                   LOCAL SEARCH (HILL CLIMBING)
# ======================================================

def hill_climb(excess_df: pd.DataFrame,
               deficit_df: pd.DataFrame,
               iters: int = 5000,
               mutation_type: str = 'granular',
               initial_non_zero: int = 5) -> tuple[pd.DataFrame, float]:
    """
    Runs the Hill Climbing algorithm.

    Args:
        excess_df pd.DataFrame  : dataframe with excess
        deficit_df pd.DataFrame : dataframe with deficit
        iters (int, optional): number of iterations (number of neighbor evaluations). Defaults to 5000.
        mutation_type (str): Type of mutation for generating neighbors ('granular' or 'row_replace'). Defaults to 'granular'.
                             HC typically mutates only a small part, so 'granular' or 'row_replace' on one row is preferred.
        initial_non_zero (int): Number of non-zero preferences for initial solution/row_replace mutation. Defaults to 5.

    Returns:
        tuple[pd.DataFrame, float]: returns best preference data-frame and its score
    """

    producers = excess_df.columns.tolist()
    consumers = deficit_df.columns.tolist()
    n_prods = len(producers)
    n_cons = len(consumers)

    M = ut.generate_preferences(producers, consumers, n=initial_non_zero)
    best_M = M.copy()
    best_fit = evaluate_matrix(M, excess_df, deficit_df)

    print(f"HC: Initial fitness: {best_fit:.4f}")

    for k in range(iters):
        M_neighbor = best_M.copy()
        row_to_mutate_index = random.randrange(n_cons)

        if mutation_type == 'granular':
             M_neighbor.iloc[row_to_mutate_index] = _swap_producer_prefs_in_row(M_neighbor.iloc[row_to_mutate_index], n_prods)

        elif mutation_type == 'row_replace':
             M_neighbor.iloc[row_to_mutate_index] = generate_sparse_vector(n_prods, non_zero_count=initial_non_zero)

        else:
            raise ValueError(f"Unknown mutation type for HC: {mutation_type}")


        fit_neighbor = evaluate_matrix(M_neighbor, excess_df, deficit_df)

        if fit_neighbor < best_fit:
            best_M, best_fit = M_neighbor.copy(), fit_neighbor

        if (k+1) % (iters // 10) == 0 or k == iters - 1:
             print(f"HC: Iter {k+1}/{iters} — Current best: {best_fit:.4f}")


    print(f"HC: Final best fitness: {best_fit:.4f}")
    return best_M, best_fit


def main():

    data_dir = '/home/miro/Bachelor/BT/data/outputs/' 
    output_dir = '/home/miro/Bachelor/BT/Models/outputs/'


    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from: {data_dir}")
    excess_monthly, deficit_monthly = ut.load_excess_deficit(data_dir)

    month = 4
    start_day = 0 
    end_day   = 10

    print(f"Preparing data for month {month}, days {start_day} to {end_day}")
    excess_df, deficit_df = ut.prepare_data(excess_monthly, deficit_monthly, month, start_day, end_day)

    # --- Run Genetic Algorithm ---
    print("\n" + "="*30)
    print("Running Genetic Algorithm")
    print("="*30)
    ga_params = {
        'generations': 10,
        'population_size': 30,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'tournament_size': 5,
        'elitism_count': 2,
        'mutation_type': 'granular', 
        'initial_non_zero': 5
    }
    print("GA Parameters:")
    for k, v in ga_params.items():
        print(f"  {k}: {v}")

    ga_start = time.time()
    ga_result, ga_score, ga_final_population = run_genetic_algorithm(
        excess_df, deficit_df, **ga_params
    )
    ga_end = time.time()


    # --- Run Simulated Annealing ---
    print("\n" + "="*30)
    print("Running Simulated Annealing")
    print("="*30)
    sa_params = {
        'iters': 350,          
        'init_temp': 100.0,
        'decay': 0.96,          
        'mutation_type': 'granular',
        'initial_non_zero': 5 
    }

    print("SA Parameters:")
    
    for k, v in sa_params.items():
        print(f"  {k}: {v}")

    sa_start = time.time()
    sa_result, sa_score = simulated_annealing(
        excess_df, deficit_df, **sa_params
    )
    sa_end = time.time()


    # --- Run Hill Climbing ---
    print("\n" + "="*30)
    print("Running Local Search (Hill Climbing)")
    print("="*30)
    lc_params = {
        'iters': 350,
        'mutation_type': 'granular',
        'initial_non_zero': 5
    }

    print("HC Parameters:")
    for k, v in lc_params.items():
        print(f"  {k}: {v}")

    lc_start = time.time()
    lc_result, lc_score = hill_climb(
        excess_df, deficit_df, **lc_params
    )
    lc_end = time.time()


    # --- Results ---
    print("\n" + "="*30)
    print('SUMMARY OF RESULTS')
    print("="*30)

    total_excess = excess_df.sum(axis=1).sum()
    total_deficit = deficit_df.sum(axis=1).sum()
    ideal_min_imbalance = np.abs(excess_df.sum(axis=1) - deficit_df.sum(axis=1)).sum()

    final_ga_score = evaluate_matrix(ga_result, excess_df, deficit_df)
    final_sa_score = evaluate_matrix(sa_result, excess_df, deficit_df)
    final_lc_score = evaluate_matrix(lc_result, excess_df, deficit_df)


    print(f'Reference (Sum of Abs Imbalances per Time Step): {ideal_min_imbalance:.4f}')
    print('-----------------------------------------------------------------------')
    print(f'GA Score: {final_ga_score:.4f}')
    print(f'SA Score: {final_sa_score:.4f}')
    print(f'LC Score: {final_lc_score:.4f}')

    print("\nTIMES")
    print("---------------------")
    print(f'GA Time: {(ga_end - ga_start) / 60:.4f} minutes')
    print(f'SA Time: {(sa_end - sa_start)/60:.4f} minutes')
    print(f'LC Time: {(lc_end - lc_start) / 60 :.4f} minutes')
    print(f"\nSaving results to {output_dir}")

    ga_weights, _, _ = optimize_weights(excess_df, deficit_df, ga_result)
    sa_weights, _, _ = optimize_weights(excess_df, deficit_df, sa_result)
    lc_weights, _, _ = optimize_weights(excess_df, deficit_df, lc_result)


    # ga_result.to_csv(f'{output_dir}ga_best_preference_m{month}.csv')
    # ga_weights.to_csv(f'{output_dir}ga_best_weights_m{month}.csv')
    # sa_result.to_csv(f'{output_dir}sa_best_preference_m{month}.csv')
    # sa_weights.to_csv(f'{output_dir}sa_best_weights_m{month}.csv')
    # lc_result.to_csv(f'{output_dir}lc_best_preference_m{month}.csv')
    # lc_weights.to_csv(f'{output_dir}lc_best_weights_m{month}.csv')

    # print("Results saved.")


if __name__ == '__main__':
    
    main()

