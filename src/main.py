from utilities import *

import numpy as np
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("error")

def calculate_tour_cost(matrix, permutation):
    cost = 0
    for i in range(len(permutation) - 1):
        # Add the distance between the current city and the next city
        cost += matrix[int(permutation[i])][int(permutation[i + 1])]

    # Back to starting point
    cost += matrix[int(permutation[-1])][int(permutation[0])]
    return cost

def two_opt_move(permutation):
    """
    Perform a 2-opt move on the current permutation.

    Args:
        permutation (np.array): Current tour permutation

    Returns:
        np.array: New permutation after 2-opt move
    """
    # Create a copy of the permutation to avoid modifying the original
    new_permutation = permutation.copy()

    # Randomly select two different indices
    n = len(permutation)
    i, j = sorted(np.random.choice(n, 2, replace=False))

    # Reverse the segment between i and j
    new_permutation[i:j+1] = new_permutation[i:j+1][::-1]

    return new_permutation

# Maybe other class
def simulated_annealing(matrix, initial_temperature, cooling_factor, cooling_schedule=lambda t, a, i: t * np.power(a, i)):
    initial_permutation = np.random.permutation(np.linspace(0, matrix.shape[0] - 1, matrix.shape[0], dtype=int))   

    current_permutation = initial_permutation
    current_cost = calculate_tour_cost(matrix, current_permutation)
    best_permutation = current_permutation
    best_cost = current_cost

    temperature = initial_temperature
    same_solution = 0

    #for i in tqdm(range(max_iterations)):
    costs = []
    iters = 0
    while same_solution < 1500 and temperature > 0:
        iters += 1
        new_permutation = two_opt_move(current_permutation)
        cost = calculate_tour_cost(matrix, new_permutation)
        delta = cost - current_cost

        if delta < 0:
            # print(f"--> Accepted move, delta: {delta}")
            current_permutation = new_permutation
            current_cost = cost
            same_solution = 0
        else:
            try:
                probability = np.exp(-delta / temperature)
            except RuntimeWarning:
                probability = 0
            if np.random.rand() <= probability:
                # print(f"--> Accepted move, probability: {probability}")
                current_permutation = new_permutation
                same_solution = 0
                current_cost = cost
            else:
                # print(f"--> Rejected move, probability: {probability}")
                same_solution += 1
                pass

        if current_cost < best_cost:
            # print(f"--> New best solution found, cost: {current_cost}")
            best_permutation = current_permutation
            best_cost = current_cost

        costs.append(current_cost)
        temperature = cooling_schedule(temperature, cooling_factor, iters)
        #print(f"Temperature: {temperature}")
        
    return best_permutation, best_cost, costs

def run_multiple_simulated_annealing(matrix, initial_temperature, cooling_factor, cooling_schedule=lambda t, a, i: t * np.power(a, i)):
    best_permutations = []
    best_costs = []
    costs_matrix = []
    for i in tqdm(range(50)):
        best_permutation, best_cost, costs = simulated_annealing(matrix, initial_temperature, cooling_factor, cooling_schedule)
        best_permutations.append(best_permutation)
        best_costs.append(best_cost)
        costs_matrix.append(np.array(costs))
    return best_permutations, best_costs, costs_matrix

# https://cs.stackexchange.com/questions/11126/initial-temperature-in-simulated-annealing-algorithm
if __name__ == "__main__":
    basedir = "../TSP-Configurations/"
   
    configs = {
        "eil51": {
            "initial_temperature": [10, 10, 5e1],
            "cooling_factor": [(1-1e-8), 5e-8, 1e-4],
            "cooling_schedule": [lambda t, a, i: t * np.power(a, i), lambda t, a, i: t - a * i, lambda t, a, i: a/np.log(i + 1)],
            "cooling_schedule_names": ["geometric", "linear", "logarithmic"],
        },
        "a280": {
            "initial_temperature": [10, 10, 50],
            "cooling_factor": [(1-5e-5), 1e-7, 1],
            "cooling_schedule": [lambda t, a, i: t * np.power(a, i), lambda t, a, i: t - a * i, lambda t, a, i: a/np.log(i + 1)],
            "cooling_schedule_names": ["geometric", "linear", "logarithmic"],
        },
        "pcb442": {
            "initial_temperature": [10, 10, 50],
            "cooling_factor": [(1-1e-4), 5e-6, 1e-2],
            "cooling_schedule": [lambda t, a, i: t * np.power(a, i), lambda t, a, i: t - a * i, lambda t, a, i: a/np.log(i + 1)],
            "cooling_schedule_names": ["geometric", "linear", "logarithmic"],
        },
    }

    for file_config in configs.keys():
        params = configs[file_config]

        # Read TSP configuration
        config = read_tsp_configuration(basedir + file_config + ".tsp.txt")
        print(f"Problem Name: {config.name}")
        print(f"Dimension: {config.dimension}")
        # print(f"First 5 Node Coordinates: {config.node_coordinates[:5]}")
        
        save_distances_matrix(config.build_distances_matrix())
        plot_nodes(config.node_coordinates, savefig=False)

        for i in range(3):
            print(f"Running experiment {i+1}")
            print(f"--> Initial temperature: {params['initial_temperature'][i]}")
            print(f"--> Cooling factor: {params['cooling_factor'][i]}")
            best_permutation, best_costs, costs_matrix = run_multiple_simulated_annealing(config.build_distances_matrix(), 
                params["initial_temperature"][i], params["cooling_factor"][i], params["cooling_schedule"][i]
            )

            min_cost_run = np.argmin(best_costs)
            print(f"--> Best cost: {best_costs[min_cost_run]}")
            print(f"--> Best permutation: {best_permutation[min_cost_run]}\n")
            box_plot_chain_length(costs_matrix, 'chain_length_' + params["cooling_schedule_names"][i] + "_" + file_config +'.pdf')
            print_cost_iterations_log(best_costs, costs_matrix, 'costs_evolution_' + params["cooling_schedule_names"][i] + "_" + file_config + '.pdf') 

            plot_tour(best_permutation[min_cost_run], config.node_coordinates, savefig=False)
        
        # Read tour solution
        tour = read_tour_solution("../TSP-Configurations/eil51.opt.tour.txt")
        tour_cost = calculate_tour_cost(config.build_distances_matrix(), tour)
        print(f"Opt tour cost: {tour_cost}")
    # print(f"First few tour positions: {tour[:5]}")



    
