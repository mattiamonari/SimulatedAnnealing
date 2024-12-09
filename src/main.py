from utilities import read_tsp_configuration, read_tour_solution

# Example usage
if __name__ == "__main__":
    # Read TSP configuration
    config = read_tsp_configuration("../TSP-Configurations/pcb442.tsp.txt")
    print(f"Problem Name: {config.name}")
    print(f"Dimension: {config.dimension}")
    print(f"First 5 Node Coordinates: {config.node_coordinates}")
    
    # Read tour solution
    tour = read_tour_solution("../TSP-Configurations/pcb442.opt.tour.txt")
    print(f"Tour length: {len(tour)}")
    print(f"First few tour positions: {dict(list(tour.items()))}")