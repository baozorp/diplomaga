import time
import yaml
import os

from concurrent.futures import ThreadPoolExecutor
from hybrid_recommendation_system.content_based import content_based
from hybrid_recommendation_system.collaborative_system import collaborative_system
from hybrid_recommendation_system.merge_recommendations import merge_recommendations
from HSATS.heuristic import get_solution
from maps_integration.points import get_points_for_map

if __name__ == "__main__":
    print("Start process")
    start = time.time()
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    content_based_coeff = config['coefficients']['content_based']
    collaborative_coeff = config['coefficients']['collaborative']
    if not os.path.exists("data"):
        try:
            os.makedirs("data")
        except OSError as e:
            raise OSError(f"Failed to create directory: {"data"}") from e
    with ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(content_based))
        futures.append(executor.submit(collaborative_system))

        for future in futures:
            future.result()
    get_solution()
    get_points_for_map()

    end = time.time() - start
    print(f"Process end with {end} seconds")
