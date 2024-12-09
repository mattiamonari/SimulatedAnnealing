from utilities import read_tsp_configuration, read_tour_solution, save_distances_matrix
import numpy as np

# Maybe other class
def simulated_annealing(matrix, initial_temperature, cooling_factor, max_iterations):
    # Either start from a random permutation
    #initial_permutation = np.random.permutation(matrix.shape[0])
    # Or from the diagonal one: [city1 city2 ... cityn]
    initial_permutation = np.eye(matrix.shape[0]) @ matrix   
    print(initial_permutation.shape)
    print(initial_permutation)


# Example usage
if __name__ == "__main__":
    # Read TSP configuration
    config = read_tsp_configuration("../TSP-Configurations/a280.tsp.txt")
    print(f"Problem Name: {config.name}")
    print(f"Dimension: {config.dimension}")
    save_distances_matrix(config.built_distances_matrix())
    print(f"First 5 Node Coordinates: {config.node_coordinates[:5]}")
    simulated_annealing(config.built_distances_matrix(), 100, 0.95, 1000)

    # Read tour solution
    tour = read_tour_solution("../TSP-Configurations/a280.opt.tour.txt")
    print(f"Tour length: {len(tour)}")
    print(f"First few tour positions: {dict(list(tour.items())[:5])}")
