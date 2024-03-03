import os
import shutil
import re
from tqdm import tqdm
import torch
import numpy as np
from pareto import compute_pareto

def get_solutoins(root_dir, generation_st, generation_end):
    """
    Fetches the paths to the solutions from the root directory given the range of generations.
    Input:
        root_dir: The root directory of the solutions
        generation_st: The starting generation from which to fetch the solutions
        generation_end: The ending generation from which to fetch the solutions

    Output:
        list of paths to the solutions
    """
    if not os.path.exists(root_dir):
        raise ValueError("The root directory does not exist")

    solution_paths = []
    subfolders = os.listdir(root_dir)
    #subfolders.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    for i, subfolder in enumerate(tqdm(subfolders, desc="fetching genomes from populations")):
        if i >= generation_st and i <= generation_end:
            print(f"Fetching genomes from {subfolder}")
            genomes = os.listdir(os.path.join(root_dir, subfolder))
            #genomes.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
            sub_path = os.path.join(root_dir, subfolder)
            solution_paths.extend([os.path.join(sub_path, genome) for genome in genomes])

    return solution_paths

def save_ensemble(solution_paths, dst):
    """
    Saves the optimal solutions to the destination directory
    Input:
        solution_paths: list of paths to the optimal solutions
        dst: root destination directory
    """
    if not os.path.exists(dst):
        print(f"Destination folder {dst} does not exist. Creating it.")
        os.makedirs(dst)

    # If destination folder is not empty, prompt user to delete or exit function
    delete = ""
    if len(os.listdir(dst)) != 0:
        while delete != "y" and delete != "n":
            delete = input(f"Destination folder {dst} is not empty. Delete contents? (y/n)")
            if delete == "y":
                shutil.rmtree(dst)
                os.makedirs(dst)
            elif delete == "n":
                print(f"Exiting function.")
                return
            else:
                print(f"Invalid input. Please enter 'y' or 'n'.")

    for i, solution in enumerate(tqdm(solution_paths, desc="Copying genomes to destination folder")):
        shutil.copy(solution, os.path.join(dst, f"genome_{i}.pth"))

def load_solutions(solution_paths):
    """
    Load the solution weights from the solution paths
    Input:
        solution_paths: list of paths to the solutions
    Output:
        Dictionary of solutions with the index as the key and the solution weight matrix as the value
    """
    fitness_scalar = lambda x: ((x + 1) / 2) * 100
    solutions = {}
    for solution_path in solution_paths:
        layer = torch.load(solution_path, map_location=torch.device('cpu'))
        solutions[solution_path] = (layer["state_dict"]["_submodules.0.weight"], fitness_scalar(layer["fitness"]))

    return solutions

def get_reference_solution(solutions):
    """
    Gets the best performing solution from the hash table of solutions

    Input:
        solutions: dictionary of solutions
            key: path to solution
            value: tuple of (weight_matridx, fitness)

    Output:
        The best performing solution, returned as a tuple of (key, value)
    """
    key = max(solutions, key=lambda x: solutions[x][1])

    return (key, solutions[key])

def compute_distance(mat1, mat2, metric="L1"):
    """
    Compute the mean squared error between two matrices
    """
    if metric == "L1":
        return (mat1 - mat2).abs().mean().item()
    if metric == "L2":
        return ((mat1 - mat2)**2).mean().item()
    elif metric == "dot-product":
        d = np.dot(mat1.flatten(), mat2.flatten())
        #d /= (np.linalg.norm(mat1) * np.linalg.norm(mat2))
        return d
    else:
        raise ValueError("Invalid metric")

def get_optimal_solution(candidate_solutions, included, n):
    """
    Given a set of solutions, and a set of candidate solution metrics, return the next optimal solution
    input:
        candidate_solutions: dictionary of candidate solutions
            key: index of solution in solutions
            value: tuple of (distance, fitness)
        included: hash table of included solutions
        n: number of solutions to include
    output:
        list of optimal solutions
    """
    optimal_solutions = []
    pareto_front = compute_pareto(candidate_solutions)
    cnt = 0
    for optimal_solution in pareto_front:
        if cnt >= n: break
        optimal_d, optimal_fitness, key = optimal_solution
        if key not in included:
            optimal_solutions.append(key)
            cnt += 1

    return optimal_solutions
