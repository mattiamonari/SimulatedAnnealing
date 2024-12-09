from typing import Dict, List, Tuple

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

