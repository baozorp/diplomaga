import math
import random
import numpy as np
from numpy.typing import NDArray
from HSATS.candidates_generator import CandidatesGenerator
from typing import List
from HSATS.data import Generator
import pandas as pd

def calculate_objective(solution, rewards: NDArray[np.float64]) -> float:
    objective = 0
    for i in solution:
        objective += rewards[i]
    return float(objective)


def simulated_annealing(travel_time_matrix, service_times, rewards, T_start, T_end, T_cool, CN, TL, Elen):

    # Initialize parameters
    t_current = T_start
    it = 0

    # Initialize solution
    initial_solution = Generator.generate_initial_route(
        travel_time_matrix, service_times, TL)
    X_current = initial_solution
    E_current = calculate_objective(X_current, rewards)

    X_best = X_current
    E_best = E_current

    tabu_list = []

    solution_generator = CandidatesGenerator(
        travel_time_matrix, service_times, TL)
    print(E_best)
    while t_current > T_end:
        while it < Elen:
            # Generate candidate solutions
            candidates = [solution_generator.generate_candidate_solution(
                X_current) for _ in range(CN)]
            # Calculate deltas
            deltas = [calculate_objective(
                candidate, rewards) - E_current for candidate in candidates]
            f_opt = max(deltas)
            # Check if the new solution is better
            if E_current + f_opt > E_best:
                X_current = candidates[deltas.index(f_opt)]
                E_current += f_opt
                X_best = X_current
                E_best = E_current
                print(E_best)
                tabu_list.append(X_current)
            else:
                # Choose a suboptimal solution from candidates
                X_opt = candidates[random.randint(0, CN - 1)]
                # Check if X_opt is not in the tabu list
                while X_opt in tabu_list:
                    X_opt = candidates[random.randint(0, CN - 1)]
                f_opt = calculate_objective(X_opt, rewards) - E_current
                p = math.exp(f_opt / t_current) if f_opt <= 0 else 1
                if random.uniform(0, 1) < p:
                    X_current = X_opt
                    E_current += f_opt
                    tabu_list = []
            it += 1

        t_current *= T_cool
        it = 0
    return [int(i) for i in X_best], E_best

def get_solution():
    count = 50
    coeff = 0.1
    TL = 50

    coordinates: NDArray[np.float64] = np.random.normal(
        0, 20, (count, 2))
    travel_time_matrix = Generator.get_travel_times(
        coordinates=coordinates,
        coeff=coeff,
        uniform_interval=(0, 2))
    service_times = Generator.get_service_times(
        coeff=coeff,
        service_time_interval=(0, 10),
        uniform_interval=(0, 2),
        length=count
    )

    rewards: NDArray[np.float64] = np.random.normal(25, 5, (count, 1))
    print(len(service_times))
    T_start = 50
    T_end = 5
    T_cool = 0.95
    CN = 25
    Elen = 100

    best_solution, best_objective = simulated_annealing(
        travel_time_matrix, service_times, rewards, T_start, T_end, T_cool, CN, TL, Elen)
    df = pd.DataFrame({'ID': best_solution})

    df.to_csv("./heuristic.csv", index=False)
    print("Best Solution:", [int(i) for i in best_solution])
    print("Best Objective Value:", best_objective)
