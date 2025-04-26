# Models

This directory holds preference optimization and weight optimization scripts.

## Problem formulation

### Data Inputs:

- excess: 15-minute time-series data on producer energy supply. Columns are producer IDs.
- deficit: 15-minute time-series data on consumer energy demand. Columns are consumer IDs.


**Only rows with non-zero production and consumption were selected to reduce model size.**

Each consumer can receive energy from up to 5 producers.

***Preferences are integers***:

- 5 = highest priority,
- 1 = lowest priority,
- 0 = no connection.

### Model Requirements:

1. **Decision Variables**:

- Priority matrix, Subscribers have preference vectors for up to 5 producers.
- Weight matrix, according which producers distribute energy

2. **Constraints**:

- A consumer can only receive energy from producers they are connected to (preference > 0).
- A consumer can only receive energy up to their demand.
- A producer can only distribute up to their available supply.
- Respect the fixed limit of 5 connections per consumer.
- Honor preference ordering if possible (e.g. prioritize higher-ranked producers).

3. **Objective Function**:

- Minimize the total unredistributed energy across all timestamps:
- Energy left with producers due to no eligible or matching demand.
- Unmet demand from consumers.

4. **Iteration**:

- The model must simulate distribution for each 15-minute interval (2880 time steps).
- After each time step:
- Update producer availability and consumer demand.
- Re-apply constraints and solve again (unless a single larger model across time steps is more efficient).

### Output:

- A feasible weight matrix and priority matrix
- Total unredistributed energy over the full month.
- A breakdown of which consumers didn’t receive their full demand and which producers had unused energy. All of this output to a file for later analysis.


## Weight optimization

In this step we need to find optimal weights given preference matrix.

Here are two approaches to weight optimization:

### 1. VP approach
---

#### Input data

- **$T$**
  - Set of timestamps $t$
- **$R$**
  - Set of allocation rounds, $r \in \{0, 1, \dots, 5\}$
- **$P$**
  - Set of preference levels, $p \in \{0, 1, \dots, 5\}$
- **$e$**
  - Matrix with excess energy for all producers at time $t$
- **$d$**
  - Matrix with deficit energy for all consumers at time $t$
- **$\text{Pref}$**
  - Tensor of shape $(i, j, p)$ representing producer-consumer preferences for each level

#### Variables

- **$W$**
  - Matrix of static weights $(i, j)$
  - What proportion of a producer's supply is directed to a consumer
- **$S$**
  - Dataframe with remaining supply at each time $t$ and round $r$
- **$D$**
  - Dataframe with remaining demand at each time $t$ and round $r, p$ (indexed jointly by round and preference)
- **$\text{PotFlow}$**
  - Potential energy flow at time $t$, from $i$ to $j$, in round $r$ and preference $p$
- **$\text{ActFlow}$**
  - Actual energy flow at time $t$, from $i$ to $j$, in round $r$ and preference $p$
- **$\text{unmet}$**
  - Unmet demand for consumer $j$ at time $t$ after all rounds
- **$\text{unused}$**
  - Unused supply for producer $i$ at time $t$ after all rounds


#### Constraints

- $\sum_{j \in C} W^{i,j} = 1 \quad \forall i \in P$
  - Each producer's weights must sum to 1
- $\text{S}^{i}_{t,0} = e^{i}_{t} \quad \forall i \in P, t \in T$
  - Initialize remaining supply
- $\text{D}^{j}_{t,0,0} = d^{j}_{t} \quad \forall j \in C, t \in T$
  - Initialize remaining demand

For each $t \in T$, for each ordered pair $(r, p)$:
- Potential flow defined by:
$$
\text{PotFlow}^{i,j}_{t,r,p} = W^{i,j} \times \text{S}^{i}_{t,r} \times \text{Pref}^{i,j,p}
$$
- Actual flow limited by supply and demand:
$$
\text{ActFlow}^{i,j}_{t,r,p} = \min\left( \text{PotFlow}^{i,j}_{t,r,p}, \text{D}^{j}_{t,r,p} \right)
$$

- Update remaining supply across rounds:
  - If $r < \text{num\_rounds} - 1$:
$$
\text{S}^{i}_{t,r+1} = \text{S}^{i}_{t,r} - \sum_{j \in C, p \in P} \text{ActFlow}^{i,j}_{t,r,p}
$$
  - If $r = \text{num\_rounds} - 1$:
$$
\text{unused}^{i}_{t} = \text{S}^{i}_{t,r} - \sum_{j \in C, p \in P} \text{ActFlow}^{i,j}_{t,r,p}
$$

- Update remaining demand across rounds and preferences:
  - If not last $(r, p)$:
$$
\text{D}^{j}_{t,r',p'} = \text{D}^{j}_{t,r,p} - \sum_{i \in P} \text{ActFlow}^{i,j}_{t,r,p}
$$
where $(r',p')$ is the next (round, preference) in the lexicographic order.

  - If last $(r, p)$:
$$
\text{unmet}^{j}_{t} = \text{D}^{j}_{t,r,p} - \sum_{i \in P} \text{ActFlow}^{i,j}_{t,r,p}
$$



#### Objective

Minimize total unmet demand and unused supply:

$$
\min \sum_{t \in T} \left( \sum_{i \in P} \text{unused}^{i}_{t} + \sum_{j \in C} \text{unmet}^{j}_{t} \right)
$$

#### Comments

This approach was shared with me by my advisor. It reliably copies the simulation so it is precise model to be used. The model has some drawbacks, the main being its non-convexity given minimum function and also its size.

### 2. Second approach
---

#### Input data

- **$T$**
  - set of timestamps $t$
- **$R$**
  - Round of allocation from 1 to 5
- **$e$**
  - Matrix with excess energy for all the producers at time $t$
- **$d$**
  - Matrix with deficit energy for all the consumers at time $t$
- **$X$**
  - Matrix with binary input, linking weights.
  - 1 on place $i$,$j$ means that flow is possible from $i$ to $j$

#### Variables

- **$W$**
  - Matrix of weights (producers, consumers)
  - What proportion of producer energy goes to which consumer
- **$S$**
  - Data-frame with time excess energy for each time $t$
  - Also rounds are added, for each time $t$
- **$D$**
  - Data-frame with time deficit energy for each time $t$
  - Also rounds are added, for each time $t$
- **$F$**
  - Energy flow from in time $t$, from $i$  to $j$ in round $R$

#### Constraints

- $\sum_{j \in C} W^{i, j} = 1$, $\forall i \in P$
  - No producer can give more than 100% of its energy
- $W^{i,j} \leq X^{i,j}$ $\forall i \in P$, $\forall j \in C$
  - Eliminating non-zero weight where is no link
- $S^{i}_{t, 1} = e^{i}_{t}$, $\forall t \in T$ and $\forall i \in P$
  - Initial value of supply of energy
- $D^{j}_{t,1} = d^{t}_{j}$, $\forall t \in T$ and $\forall j \in C$
  - Initial value of demand of energy
- $F^{i,j}_{t,r} \leq W^{i, j} * S^{i}_{t, r-1} * X^{i, j}$, $\forall i \in P$, $\forall j \in C$, 
  - Flow cannot exceed the supply
- $S^{i}_{t, r} = S^{i}_{t, r-1} - F^{i, j}_{t, r-1}$, $\forall i \in P~t \in T,~r \in \{2,..,5\}$
  - Update of supplied energy
- $D^{j}_{t, r} = D^{j}_{t, r-1} - F^{i, j}_{t, r-1}$, $\forall j \in C ~t \in T,~r \in \{2,..,5\}$
  - Update of demanded energy

#### Objective

- $Minimize~\sum_{i \in P, ~t \in T} S^{i}_{t,5} + \sum_{j \in C, ~t \in T,} D^{j}_{t,5}$


#### Comments

This model is not exactly the same as the simulation, value of objective function is different than fitness function which is implemented as a simulation process. The preferences are just binary variables, so it just represents link between consumer and producer. This model is smaller and also convex.

### Comparison

Comparison of these two models was done in file `compare_models.py`, but for larger datasets (more than 1000 rows roughly half a month) the first model did not finish and restart of the computer was needed. When run on smaller datasets the first approach was ~4 times slower than the second approach so it was decided to use approximate model instead of true representation.

## Preference optimization

Preferences are the order in which consumers take energy from producers, higher the number the bigger the priority. The goal here is to find preferences for each consumer to minimize the unallocated energy. We tried 3 approaches which are very similar and did not decide which one to use. The details of the algorithms can be viewed in `preference_optimization.py`.

### 1. Genetic Algorithm (GA)

Inspired by biological evolution, the GA searches for the best solution by mimicking natural selection. It maintains a population of potential solutions, evaluates their fitness, and uses evolutionary operators (selection, crossover, mutation) to create a new generation. Over many generations, the population should, on average, improve its fitness, converging towards better solutions.

#### Implementation in Code

- **Population**: A list of pandas.DataFrame objects, where each DataFrame is a preference matrix representing an individual solution.
- **Fitness Evaluation**: The evaluate_matrix function determines the fitness of each preference matrix by calling optimize_weights (to find the best energy allocation for that preference matrix) and then ut.fitness to calculate the total leftover energy. Lower fitness is better.
- **Selection**: The tournament_selection function is used. It randomly selects a small group (a "tournament") of individuals from the current population and picks the best one from that group to be a parent. This process is repeated to select enough parents for the next generation.
- **Elitism**: The run_genetic_algorithm function explicitly copies the very best individuals (elitism_count) from the current generation directly into the next generation, ensuring that the best solutions found so far are not lost.
- **Crossover**: The crossover function takes two parent preference matrices and creates a child by swapping entire rows (consumer preferences) between them with a certain probability (0.5 per row in this implementation).
- **Mutation**: The mutate_preference_matrix function introduces variation. The code offers two types: 'row_replace' (replaces a whole row with a new random sparse vector, similar to the original code) and 'granular' (swaps the preference values of two random producers within a single consumer's row). The 'granular' type is the default and preferred method as it allows for finer-tuning of solutions. This mutation is applied to individual consumer rows with a probability defined by mutation_rate.

### 1. Simulated Annealing (SA)

Based on the annealing process in metallurgy (heating and slowly cooling a material to reduce defects), SA explores the solution space by starting at a high "temperature" (allowing broad exploration) and gradually cooling down (becoming more focused). It always accepts better neighboring solutions but also accepts worse ones with a probability that decreases as the temperature drops. This allows it to escape local optima early in the search.

#### Implementation in Code:

- **Solution Representation**: A single pandas.DataFrame (the preference matrix) represents the current solution.
- **Neighbor Generation**: In each iteration, a neighboring solution is generated by taking the current solution and applying the chosen mutation type (mutation_type, typically 'granular') to a single, randomly selected consumer row.
- Temperature Schedule: Starts with init_temp and decreases multiplicatively by decay in each iteration.
- **Acceptance Criteria**: If a neighbor has better fitness than the current solution, it's accepted. If it's worse, it's accepted with a probability calculated using the formula $e = \Delta E/T$, where $\Delta E$ is the difference in fitness (current - neighbor) and T is the current temperature.

### 3. Local Search (Hill Climbing)

Hill Climbing is a simple greedy search algorithm. It starts with an initial solution and iteratively moves to a neighboring solution only if it is strictly better than the current one. It stops when no better neighbor can be found, which corresponds to reaching a local optimum. It does not have mechanisms to escape local optima once trapped.

#### Implementation in Code:

- **Solution Representation**: A single pandas.DataFrame (the preference matrix) represents the current best solution found so far.
- **Neighbor Generation**: Similar to SA, a neighboring solution is generated by applying the chosen mutation type (mutation_type, typically 'granular') to a single, randomly selected consumer row of the current best solution.
- **Acceptance Criteria**: Only accepts the neighbor if its fitness is strictly better (fit_neighbor < best_fit) than the current best fitness.
- **Stagnation**: If an iteration does not find a better neighbor, the current best solution is kept, and the algorithm continues trying other neighbors. It essentially stops making progress once a local optimum is reached and all neighbors have been implicitly checked relative to it.

