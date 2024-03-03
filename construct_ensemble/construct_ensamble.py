from visualize import visualize_population, visualize_ensemble, visualize_pareto_front
from utils import get_solutoins, load_solutions, save_ensemble, get_reference_solution, compute_distance, get_optimal_solution

def construct_ensamble(solutions, k):
    ref_key, ref_solution = get_reference_solution(solutions) # The reference solution is the best performing solution
    included = {ref_key:True} # Hash table to keep track of already included solutions
    new_set = [ref_key] # The new set of solutions

    while len(new_set) < k:
        candidate_solutions = {}
        for candidate_key in solutions.keys():
            if candidate_key in included: continue
            min_d = float("inf"); min_key = None
            for solution_key in new_set:
                distance = compute_distance(solutions[solution_key][0], solutions[candidate_key][0])
                if distance < min_d:
                    min_d = distance
                    min_key = candidate_key

            candidate_solutions[candidate_key] = (min_d, solutions[candidate_key][1])

        optimal_solutions = get_optimal_solution(candidate_solutions, included, 1)

        #visualize_pareto_front(candidate_solutions, optimal_solutions, len(new_set))

        for key in optimal_solutions:
            new_set.append(key)
            included[key] = True

    return new_set


def generate_ensemble(root_dir, dst, generation_st, generation_end, k):
    """
    Generates an ensemble of solutions from the solutions in the root directory
    Input:
        root_dir: root directory of the solutions
        dst: destination directory to save the ensemble
        generation_st: starting generation from which to fetch the solutions
        generation_end: ending generation from which to fetch the solutions
        k: number of solutions to include in the ensemble
    """
    solution_paths = get_solutoins(root_dir, generation_st, generation_end)
    solutions = load_solutions(solution_paths)
    ensemble = construct_ensamble(solutions, k)
    visualize_population(solutions)
    visualize_ensemble(solutions, ensemble)
    #save_ensemble(ensemble, dst)


# Testing
if __name__ == "__main__":
    src_path = "./test_populations/"
    dst_path = "./test_ensemble/"
    generation_st = 15
    generation_end = 25
    k = 20

    generate_ensemble(src_path, dst_path, generation_st, generation_end, k)
