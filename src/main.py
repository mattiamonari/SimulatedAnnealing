from utilities import *
import numpy as np

def calculate_tour_cost(matrix, permutation):
    cost = 0
    for i in range(len(permutation) - 1):
        # Add the distance between the current city and the next city
        cost += matrix[int(permutation[i])][int(permutation[i + 1])]

    # Back to starting point
    cost += matrix[int(permutation[-1])][int(permutation[0])]
    return cost

def swap_two_cities(permutation):
    i = np.random.randint(len(permutation))
    j = np.random.randint(len(permutation))
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation

# Maybe other class
def simulated_annealing(matrix, initial_temperature, cooling_factor, max_iterations):
    # Either start from a random permutation
    #initial_permutation = np.random.permutation(matrix.shape[0])
    # Or from the diagonal one: [city1 city2 ... cityn]
    initial_permutation = np.random.permutation(np.linspace(0, matrix.shape[0] - 1, matrix.shape[0], dtype=int))   
    # print(initial_permutation.shape)
    # print(initial_permutation)

    current_permutation = initial_permutation
    current_cost = calculate_tour_cost(matrix, current_permutation)
    best_permutation = current_permutation
    best_cost = current_cost

    temperature = initial_temperature

    for i in range(max_iterations):
        print(f"Iteration: {i}")
        new_permutation = swap_two_cities(current_permutation)
        cost = calculate_tour_cost(matrix, new_permutation)
        delta = cost - current_cost

        if delta < 0:
            print(f"--> Accepted move, delta: {delta}")
            current_permutation = new_permutation
            current_cost = cost
        else:
            probability = np.exp(-delta / temperature)
            if np.random.rand() < probability:
                print(f"--> Accepted move, probability: {probability}")
                current_permutation = new_permutation
                current_cost = cost
            else:
                print(f"--> Rejected move, probability: {probability}")

        if current_cost < best_cost:
            print(f"--> New best solution found, cost: {current_cost}")
            best_permutation = current_permutation
            best_cost = current_cost

        temperature *= cooling_factor
        print(f"Temperature: {temperature}")

    return best_permutation, best_cost


# Example usage
if __name__ == "__main__":
    # Read TSP configuration
    config = read_tsp_configuration("../TSP-Configurations/a280.tsp.txt")
    print(f"Problem Name: {config.name}")
    print(f"Dimension: {config.dimension}")
    save_distances_matrix(config.build_distances_matrix())
    print(f"First 5 Node Coordinates: {config.node_coordinates[:5]}")
    plot_nodes(config.node_coordinates, savefig=False)

    best_permutation, best_cost = simulated_annealing(config.build_distances_matrix(), 100, 0.95, 1000)

    print(f"Best permutation: {best_permutation}")
    print(f"Best cost: {best_cost}")
    # Read tour solution
    tour = read_tour_solution("../TSP-Configurations/a280.opt.tour.txt")
    tour_cities = np.array([c for c in tour.keys()]) - 1
    tour_cost = calculate_tour_cost(config.build_distances_matrix(), tour_cities)
    print(f"Tour length: {tour_cost}")
    print(f"First few tour positions: {dict(list(tour.items())[:5])}")
