import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, List
import pandas as pd


class Generator:
    @staticmethod
    def generate_normal_values(count: int, mean: int, std_dev: int) -> NDArray[np.float64]:
        coordinates: NDArray[np.float64] = np.random.normal(
            mean, std_dev, (count, 2))
        return coordinates

    @staticmethod
    def get_coordinates(count: int):
        center_lat, center_lon = 55.751244, 37.618423
        np.random.seed(42)
        lat_offsets = np.random.uniform(-0.1, 0.1, size=count)
        lon_offsets = np.random.uniform(-0.15, 0.15, size=count)
        moscow_coords = pd.DataFrame({
            "lat": center_lat + lat_offsets,
            "lon": center_lon + lon_offsets
        })
        moscow_coords.to_csv("data/coordinates.csv", index=False, header=False)



    @staticmethod
    def get_travel_times(coordinates: NDArray[np.float64],
                         coeff: float,
                         uniform_interval: Tuple[float, float]
                         ):
        euclidean_distances: NDArray[np.float64] = pdist(
            coordinates, 'euclidean')
        euclidean_distances_with_coeff: NDArray[np.float64] = euclidean_distances * coeff
        uniform_values: NDArray[np.float64] = np.random.uniform(
            uniform_interval[0], uniform_interval[1], len(euclidean_distances_with_coeff))
        result_distances = euclidean_distances_with_coeff + uniform_values
        distances_squareform: NDArray[np.float64] = squareform(
            result_distances)
        df = pd.DataFrame(distances_squareform)
        df.to_csv("data/travel_times.csv", index=False, header=True)

    @staticmethod
    def get_service_times(coeff: float,
                          service_time_interval: Tuple[float, float],
                          uniform_interval: Tuple[float, float],
                          length: int
                          ):
        service_time_values: NDArray[np.float64] = np.random.uniform(
            service_time_interval[0], service_time_interval[1], length)
        coeff_service_time_values: NDArray[np.float64] = service_time_values * coeff
        uniform_values: NDArray[np.float64] = np.random.uniform(
            uniform_interval[0], uniform_interval[1], length)
        result_service_times = coeff_service_time_values + uniform_values
        pd.Series(result_service_times).to_csv("data/service_times.csv", index=False)

    @staticmethod
    def generate_initial_route(travel_time_matrix: NDArray[np.float64], service_times: NDArray[np.float64], time_limit: int) -> List[int]:
        num_locations = len(travel_time_matrix)
        remaining_locations = set(range(1, num_locations))
        current_location = 0
        current_time = 0

        route = [current_location]

        while remaining_locations:
            best_next_location = None
            best_time_increase = float('inf')

            for next_location in remaining_locations:
                travel_time = travel_time_matrix[current_location,
                                                 next_location]
                service_time = service_times[next_location]
                total_time = current_time + travel_time + service_time

                if total_time <= time_limit and total_time < best_time_increase:
                    best_next_location = next_location
                    best_time_increase = total_time

            if best_next_location is not None:
                remaining_locations.remove(best_next_location)
                current_location = best_next_location
                current_time = best_time_increase
                route.append(current_location)
            else:
                break
        return route
