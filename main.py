import time
from threading import Thread
import yaml
import os


from recommendation_systems.content_based import content_based
from recommendation_systems.collaborative_system import collaborative_system
from HSATS.heuristic import get_solution
from merge_systems.merge_recommendations import merge_recommendations
from merge_systems.interference_to_euristic import interference_to_euristic

if __name__ == "__main__":
    print("Start process")
    start = time.time()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    sources_path = config['paths']['sources']
    results_path = config['paths']['results']
    content_based_coeff = config['coefficients']['content_based']
    collaborative_coeff = config['coefficients']['collaborative']
    heuristic_coeff = config['coefficients']['heuristic']
    if not os.path.exists(sources_path):
        raise FileNotFoundError(f"Exhibit folder {sources_path} not found")
    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path)
        except OSError as e:
            raise OSError(f"Failed to create directory: {results_path}") from e
    targets = [
        (content_based, sources_path, results_path),
        (collaborative_system, sources_path, results_path),
        (get_solution, )
    ]

    processes = []

    for target in targets:
        process = Thread(target=target[0], args=target[1:])
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    merge_recommendations(sources_path, results_path, content_based_coeff, collaborative_coeff)
    interference_to_euristic(results_path, heuristic_coeff)

    end = time.time() - start
    print(f"Process end with {end} seconds")
