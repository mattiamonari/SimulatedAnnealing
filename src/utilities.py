from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from configuration import TSPConfiguration

__all__ = ["read_tsp_configuration", "read_tour_solution", "save_distances_matrix", 
           "plot_nodes", "box_plot_chain_length", "print_cost_iterations_log", 
           "plot_tour"]

def read_tour_solution(tour_file_path: str) -> Dict[int, int]:
    """
    Read the optimal tour solution from a .opt.tour file.
    
    Args:
        tour_file_path (str): Path to the .opt.tour file
    
    Returns:
        List[int]: A last of nodes rapresenting the optimal tour
    """
    tour_order = []
    with open(tour_file_path, 'r') as f:
        # Skip header lines
        while True:
            line = f.readline().strip()
            if line == 'TOUR_SECTION':
                break
        
        # Read tour sequence
        position = 0
        for line in f:
            node = int(line.strip())
            if node == -1:  # End of tour marker
                break
            tour_order.insert(position, node - 1) 
            position += 1
    
    return tour_order

def read_tsp_configuration(tsp_file_path: str) -> TSPConfiguration:
    """
    Read the TSP problem configuration from a .tsp file.
    
    Args:
        tsp_file_path (str): Path to the .tsp file
    
    Returns:
        Configuration: A Configuration object with problem details
    """
    # Initialize variables to store configuration
    name = ""
    comment = ""
    type = ""
    dimension = 0
    edge_weight_type = ""
    node_coordinates = []
    
    # Read the file
    with open(tsp_file_path, 'r') as f:
        # Parse header information
        while True:
            line = f.readline().strip()
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("COMMENT"):
                comment = line.split(":")[1].strip()
            elif line.startswith("TYPE"):
                type = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                break
        
        # Read node coordinates
        for _ in range(dimension):
            line = f.readline().strip()
            # Split line and convert to integers
            parts = line.split()
            node_coordinates.append((float(parts[1]), float(parts[2])))
    
    return TSPConfiguration(
        name=name,
        comment=comment,
        type=type,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        node_coordinates=node_coordinates
    )

def save_distances_matrix(matrix):
    mat = np.array(matrix)
    with open(f'{"distances.txt"}','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

def plot_nodes(nodes_coordinates, savefig=False):
    x = [x for x, y in nodes_coordinates]
    y = [y for x, y in nodes_coordinates]
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color='blue', s=1)
    plt.title("Nodes configuration")
    plt.xlabel('X')
    plt.ylabel('Y')
    if savefig: plt.savefig(f'../images/nodes.pdf')
    else: plt.show()
    plt.close()
    
def print_cost_iterations_log(mean, costs_matrix, filename='costs_evolution.pdf'):
    min_size = min(arr.shape[0] for arr in costs_matrix)
    costs_matrix = [arr[:min_size] for arr in costs_matrix]
    
    costs_matrix = np.array(costs_matrix, dtype=np.float64)
    mean = np.mean(costs_matrix, axis=0)
    std = np.std(costs_matrix, axis=0)
    lower_bound = mean - 1.96 * std / np.sqrt(costs_matrix.shape[0])
    upper_bound = mean + 1.96 * std / np.sqrt(costs_matrix.shape[0])

    plt.semilogx(mean, label='Mean')
    plt.fill_between(range(len(mean)), lower_bound, upper_bound, alpha=0.4, label='95% confidence interval')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig(f'../images/{filename}')
    plt.show()

def box_plot_chain_length(costs_matrix, filename='chain_length.pdf'):
    plt.boxplot([arr.shape[0] for arr in costs_matrix])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(f'../images/{filename}')
    plt.show()

def plot_tour(best_permutation, node_coordinates, savefig=False):
    xs = [node_coordinates[i][0] for i in best_permutation]
    ys = [node_coordinates[i][1] for i in best_permutation]
    xs.append(xs[0])
    ys.append(ys[0])

    xs, ys = np.array(xs), np.array(ys)

    plt.figure(figsize=(10, 8))
    # quiver plot for the route
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    plt.quiver(xs[:-1], ys[:-1], xs[1:] - xs[:-1], ys[1:] - ys[:-1], scale_units='xy', angles='xy', scale=1, width=0.004)

    # plot cities
    plt.scatter(xs, ys, color='black', s=5, zorder=100)
    # add label on each city
    for i in range(len(xs) - 1):
        plt.text(xs[i], ys[i] + 0.6, str(best_permutation[i] + 1), fontsize=8, ha='center', va='center')
    # add a star to starting point
    plt.plot(xs[0], ys[0], 'ro', label='Starting Point')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.legend()
    
    if savefig: plt.savefig(f'../images/tour.pdf')
    else: plt.show()
    plt.close()