from typing import Dict, List, Tuple
import math
import numpy as np

class TSPConfiguration:
    def __init__(self, name: str, comment: str, type: str, dimension: int, 
                 edge_weight_type: str, node_coordinates: List[Tuple[int, int]]):
        """
        Initialize a Configuration object for a Traveling Salesman Problem instance.
        
        Args:
            name (str): Name of the TSP problem
            comment (str): Additional comment about the problem
            type (str): Type of the problem (e.g., TSP)
            dimension (int): Number of nodes/cities
            edge_weight_type (str): Type of edge weight calculation
            node_coordinates (List[Tuple[int, int]]): List of (x, y) coordinates for each node
        """
        self.name = name
        self.comment = comment
        self.type = type
        self.dimension = dimension
        self.edge_weight_type = edge_weight_type
        self.node_coordinates = node_coordinates


    def calculate_euclidean_distance(self, index1: int, index2: int) -> float:
        """
        Calculate the Euclidean distance between two nodes given their indices.

        Args:
            index1 (int): Index of the first node (1-based indexing)
            index2 (int): Index of the second node (1-based indexing)

        Returns:
            float: Euclidean distance between the two nodes
        """
        x1, y1 = self.node_coordinates[index1]
        x2, y2 = self.node_coordinates[index2]

        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def build_distances_matrix(self):
        distances_matrix = [[0 for _ in range(self.dimension)] for _ in range(self.dimension)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                distances_matrix[i][j] = self.calculate_euclidean_distance(i, j)
        # np.array vs np.matrix 
        return np.array(distances_matrix) 