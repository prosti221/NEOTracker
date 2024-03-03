import numpy as np
import matplotlib.pyplot as plt
from pareto import compute_pareto

def generate_points(n, r):
    points = []
    for i in range(n):
        x = np.random.uniform(-r, r)
        y = np.random.uniform(-r, r)
        points.append((x, y))

    return points

def plot_points(points, fitness=None):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    if fitness is not None:
        plt.scatter(x, y, c=fitness)
        plt.gray()
    else:
        plt.scatter(x, y)
    plt.show()

def max_diversity(points, k, ref):
    new_set = [ref]
    l2_dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    random_fitness = lambda p: np.random.uniform(0, 100)
    fitness_set = [random_fitness(ref)]
    included = {0:True}

    while len(new_set) < k:
        max_p = None; max_d = 0; max_i = None
        candidate_solutions = {}
        for i, candidate_point in enumerate(points):
            if i in included: continue
            min_p = None; min_d = float("inf"); min_i = None
            for point in new_set:
                l2 = l2_dist(point, candidate_point)
                if l2 < min_d:
                    min_d = l2
                    min_p = candidate_point
                    min_i = i

            if min_d > max_d and min_i not in included:
                max_d = min_d
                max_p = min_p
                max_i = min_i

            candidate_solutions[min_i] = (min_d, random_fitness(min_i))

        # Compute the pareto front with respect to the diversity and fitness
        # Select the two best solutions for each iteration
        pareto_front = compute_pareto(candidate_solutions)
        for optimal_solution in [pareto_front[0]]:
            optimal_d, optimal_fitness, index = optimal_solution
            if index not in included:
                new_set.append(points[index])
                fitness_set.append(optimal_fitness)
                included[index] = True

        """
        if max_p is not None:
            new_set.append(max_p)
            included[max_i] = True
        """

    return new_set, fitness_set


if __name__ == "__main__":
    points = generate_points(250*15, 100)
    plot_points(points)
    points, fitness = max_diversity(points, 50, points[0])
    plot_points(points, fitness)
