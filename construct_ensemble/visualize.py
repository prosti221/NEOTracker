import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from .utils import compute_distance

def visualize_pareto_front(candidate_solutions, optimal_solutions, iteration):
    """
    Visualize the pareto front of the candidate solutions in a scatter plot.
    X-axis: distance
    Y-axis: fitness
    The optimal solutions are highlighted in red.
    """
    plt.scatter([candidate_solutions[key][0] for key in candidate_solutions.keys()], [candidate_solutions[key][1] for key in candidate_solutions.keys()], label="Candidate solutions")
    plt.scatter([candidate_solutions[key][0] for key in optimal_solutions], [candidate_solutions[key][1] for key in optimal_solutions], color="red", label="Optimal solutions")
    plt.xlabel("Distance")
    plt.ylabel("Fitness")
    plt.title(f"Pareto front of candidate solutions itr={iteration}")
    plt.legend()
    plt.show()

def visualize_population(solutions):
    # Visualize the entire population using MDS
    distance_map = np.zeros((len(solutions), len(solutions)))

    for i, key1 in enumerate(solutions.keys()):
        for j, key2 in enumerate(solutions.keys()):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    mds = MDS(n_components=2, dissimilarity="precomputed")
    embedding = mds.fit_transform(distance_map)
    # Get color based on the fitness
    fitness = [solutions[key][1] for key in solutions.keys()]

    # Plot the MDS embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], c=fitness)
    plt.xlabel("mds 1")
    plt.ylabel("mds 2")
    plt.title("Parameter space embedding of all solutions")
    plt.show()


def visualize_ensemble(solutions, ensemble_set):
    # Visualize ensemble using MDS
    distance_map = np.zeros((len(ensemble_set), len(ensemble_set)))
    for i, key1 in enumerate(ensemble_set):
        for j, key2 in enumerate(ensemble_set):
            distance_map[i][j] = compute_distance(solutions[key1][0], solutions[key2][0])

    #plt.imshow(distance_map, cmap="hot", interpolation="nearest")
    #plt.show()

    mds = MDS(n_components=2, dissimilarity="precomputed")
    embedding = mds.fit_transform(distance_map)

    fitness = [solutions[key][1] for key in ensemble_set]

    # Plot the MDS embedding
    plt.scatter(embedding[:, 0], embedding[:, 1], c=fitness)
    plt.xlabel("mds 1")
    plt.ylabel("mds 2")
    plt.title("Parameter space embedding of ensemble solutions")
    plt.show()
