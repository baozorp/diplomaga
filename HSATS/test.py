import random
import math
import numpy as np


class CandidatesGenerator:
    def __init__(self, travel_time_matrix, service_times, time_limit, penalty_coefficient):
        self.travel_time_matrix = travel_time_matrix
        self.service_times = service_times
        self.time_limit = time_limit
        self.penalty_coefficient = penalty_coefficient
        # Initialize penalties to zero
        self.penalty_values = np.zeros(len(service_times))

    def generate_candidate_solution(self, current_solution):
        candidate = current_solution.copy()

        # Randomly choose a neighborhood strategy
        strategy = np.random.choice([1, 2, 3, 4])

        if strategy == 1:
            self.neighborhood_1(candidate)
        elif strategy == 2:
            self.neighborhood_2(candidate)
        elif strategy == 3:
            self.neighborhood_3(candidate)
        elif strategy == 4:
            self.neighborhood_4(candidate)

        return candidate

    def neighborhood_1(self, solution):
        unvisited_spot = self.find_unvisited_spot(solution)
        self.insert_spot(solution, unvisited_spot)

    def neighborhood_2(self, solution):
        self.remove_spot(solution)

    def neighborhood_3(self, solution):
        self.reverse_sequence(solution)

    def neighborhood_4(self, solution):
        unvisited_spot = self.find_unvisited_spot(solution)
        self.insert_and_remove(solution, unvisited_spot)

    def find_unvisited_spot(self, solution):
        locations = len(self.service_times)
        unvisited_spots = set(range(locations)) - set(solution)
        return np.random.choice(list(unvisited_spots))

    def insert_spot(self, solution, unvisited_spot):
        best_position = 0
        best_increase = float('inf')

        for i in range(1, len(solution)):
            new_solution = solution[:i] + [unvisited_spot] + solution[i:]
            increase = self.calculate_objective_time(
                new_solution) - self.calculate_objective_time(solution)

            # Check if the new solution respects the time limit
            if self.validate_solution(new_solution):
                # Update penalties based on feasibility
                self.update_penalties(new_solution)
                # Calculate total increase with penalties
                total_increase = increase + \
                    self.calculate_penalty(new_solution)
                if total_increase < best_increase:
                    best_position = i
                    best_increase = total_increase

        solution.insert(best_position, unvisited_spot)

    def remove_spot(self, solution):
        if len(solution) > 1:
            for _ in range(len(solution)):
                remove_position = np.random.choice(range(1, len(solution)))
                removed_spot = solution.pop(remove_position)

                # Check if the new solution respects the time limit
                if self.validate_solution(solution):
                    # Update penalties based on feasibility
                    self.update_penalties(solution)
                    return

                solution.insert(remove_position, removed_spot)

    def reverse_sequence(self, solution):
        if len(solution) > 2:
            start, end = np.random.choice(
                range(1, len(solution) - 1), size=2, replace=False)
            solution[start:end+1] = reversed(solution[start:end+1])
            if self.validate_solution(solution):
                # Update penalties based on feasibility
                self.update_penalties(solution)
            else:
                solution[start:end+1] = reversed(solution[start:end+1])

    def insert_and_remove(self, solution, unvisited_spot):
        if len(solution) > 1:
            self.insert_spot(solution, unvisited_spot)
            self.remove_spot(solution)

    def validate_solution(self, solution):
        total_time = self.calculate_objective_time(solution)
        print(total_time)
        return total_time <= self.time_limit

    def calculate_objective_time(self, solution):
        total_time = 0
        for i in range(len(solution) - 1):
            travel_time = self.travel_time_matrix[solution[i], solution[i + 1]]
            service_time = self.service_times[solution[i + 1]]
            total_time += travel_time + service_time
        return total_time

    def update_penalties(self, solution):
        for spot in solution[1:]:
            if self.validate_solution([0] + [spot]):
                self.penalty_values[spot] *= self.penalty_coefficient
            else:
                self.penalty_values[spot] += 1

    def calculate_penalty(self, solution):
        penalty = 0
        for spot in solution[1:]:
            penalty += self.penalty_values[spot]
        return penalty
