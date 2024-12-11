from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from configuration import TSPConfiguration

__all__ = ["read_tsp_configuration", "read_tour_solution", "save_distances_matrix", 
           "plot_nodes"]

def read_tour_solution(tour_file_path: str) -> Dict[int, int]:
    """
    Read the optimal tour solution from a .opt.tour file.
    
    Args:
        tour_file_path (str): Path to the .opt.tour file
    
    Returns:
        Dict[int, int]: A dictionary mapping node indices to their order in the optimal tour
    """
    tour_order = {}
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
            tour_order[node] = position
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
    mat = np.matrix(matrix)
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