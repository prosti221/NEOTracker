import sys
import time
import argparse
sys.path.append("pytracking")
sys.path.append("pytracking/pytracking")
sys.path.append("pytracking/ltr")
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from pytracking.evaluation import Tracker

from pytracking.tracker.adnet.utils import *
from pytracking.tracker.adnet.train_ne import *
from pytracking.tracker.adnet.train_sl import *
from pytracking.tracker.adnet.train import *
from pytracking.tracker.adnet.synthetic import *
import numpy as np

from utils import *
from construct_ensemble.constructer import generate_ensemble
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="mk2_backbone",
    group="SLRLNE_f",
    job_type="train",
    name="SLRLNE_f-1",
   config={
        "Number of samples": 97
    }
)

def evaluate_all_checkpoints(params, dataset, checkpoint_path):
    N_RUNS = 1
    force = True
    trackers = []
    debug = 0

    targets = get_all_checkpoints(checkpoint_path)
    for i, target in enumerate(targets):
        num = re.findall(r"\d+", str(target).split(".")[0])[0]
        trackers.extend(evaluate_tracker(dummy_dataset, target, f"NE_gen{num}", N_RUNS, display_name=f"$NE_{{{num}}}$", force=force, threads=0, debug=debug))
        eval_data = print_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
        #print_per_sequence_results(trackers, dummy_dataset,"NE vs SLRL - dummy dataset",  merge_results=True, force_evaluation=True, skip_missing_seq=True)

def construct_ensemble_folder(src, dst, generation_start, generation_end, top_n):
    """
    A function that takes in a source folder containing folders with genomes per generation and picks the top_n genomes to put in dst folder.
    generation_start and generation_end define the range of generations to pick from.
    """
    if not os.path.exists(src):
        raise ValueError(f"Source folder {src} does not exist.")
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

    genome_list = []
    subfolders = os.listdir(src)
    subfolders.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    for i, subfolder in enumerate(tqdm(subfolders, desc="fetching genomes from populations")):
        gen = int(re.findall(r'\d+', subfolder)[0])
        if i >= generation_start and i <= generation_end:
        #if gen >= generation_start and gen <= generation_end:
            print(f"Fetching genomes from {subfolder}")
            # Get the top_n genomes
            genomes = os.listdir(os.path.join(src, subfolder))
            genomes.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
            genomes = genomes[:top_n]
            genome_list.extend([os.path.join(subfolder, genome) for genome in genomes])

    for i, genome in enumerate(tqdm(genome_list, desc="Copying genomes to destination folder")):
        #print(f"Copying {genome} to {os.path.join(dst, f'genome_{i}.pth')}")
        shutil.copy(os.path.join(src, genome), os.path.join(dst, f"genome_{i}.pth"))

def half_sequences(dataset):
    """
    A function that halves the number of frames per sequence in a dataset.
    """
    for sequence in dataset:
        sequence.frames = sequence.frames[:40]
        sequence.ground_truth_rect = sequence.ground_truth_rect[:40]

    return dataset

if __name__ == '__main__':
    tracker = Tracker("adnet", "default", run_id=None)
    params = tracker.get_parameters()
    device = torch.device('cuda:0')
    #start_from = latest_checkpoint(params.checkpoints_path / "RL-XL")
    start_from = latest_checkpoint(params.checkpoints_path / "RL-mk3")

    dummy_dataset = load_dummy_dataset()
    dataset = load_datasets(
        train_tags=["vot2014", "vot2015","vot2017","vot-st2020","lasot","got_10k-val"],
        #train_tags=["got_10k-val"],
        val_tags=["vot-st2021"],
        #val_tags=["got_10k-train"],
        n_train_sequences=-1,
        n_val_sequences=-1,
        n_test_sequences=-1,
        remove_overlapping=True
    )

    #dummy_dataset = dataset["train"][:50]
    #dummy_dataset = half_sequences(dataset["train"][:200])
    dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=0.038) # gets us 97 samples
    #dummy_dataset = half_sequences(dummy_dataset)
    #dummy_dataset = dataset["train"]
    #dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=0.0125)  # gets us 150 samples
    #dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=0.128)  # gets us 25 samples
    #dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=-0.005)  # gets us 247 samples
    print(len(dataset["train"]))

    stats = train_ne(
        initial_model_path = start_from,
        params = params,
        dataset_train = dataset["train"],
        dataset_eval = dataset["val"],
        population_size = 256,
        n_generations = 1200,
        reward_version="PAPER",
        experiment_name="NE-evotorch", 
        device=device,
        checkpoints=[], # get_all_checkpoints(params.checkpoints_path / "SLRL")
        resume=False
    )
   

    """
    construct_ensemble_folder(
        src=Path(__file__).parent / "final_experiments_mk2/SLRLNE/models/saved_populations",
        dst=params.checkpoints_path / "ensemble",
        generation_start=10,
        generation_end=10000,
        top_n=10
    )
    """

    """
    generate_ensemble(
            root_dir=Path(__file__).parent / "final_experiments_mk2/SLRLNE/models/saved_populations", 
            #root_dir=Path(__file__).parent / "networks/adnet_checkpoints/NE-evotorch/pytorch_checkpoints/saved_populations", 
            dst=params.checkpoints_path / "ensemble", 
            generation_st=0, 
            generation_end=26, 
            k=20
    )
    """

    """
    p_target_rl1 = latest_checkpoint(params.checkpoints_path / "RL-XL")
    #p_target_rl2 = latest_checkpoint(params.checkpoints_path / "SLRL_fc6_f")
    #p_target_rl3 = latest_checkpoint(params.checkpoints_path / "SLRL+RL_fc6_f")
    p_target_ne1 = params.checkpoints_path / "SLNE_mk1" / "best_genome_medium.pth"
    p_target_ne2 = Path(__file__).parent / "final_experiments_mk2/SLRLNE/models/best_genome_540.pth"
    #p_target_ne = params.checkpoints_path / "NE-evotorch" / "pytorch_checkpoints/best_genome_540.pth" 

    N_RUNS = 1
    force = True
    trackers = []
    debug = 0
    # For evaluating all the ne checkpoint.pth"
    #evaluate_all_checkpoints(params, dataset, checkpoint_path=params.checkpoints_path / "NE-evotorch" / "pytorch_checkpoints")
    #evaluate_all_checkpoints(params, dataset, Path(__file__).parent / "final_experiments" / "SLNE_f" / "models")

    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl3, "SLRL+RL_fc6", N_RUNS, display_name="$SLRL+RL_fc6$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl2, "SLRL_fc6_f", N_RUNS, display_name="$SLRL_fc6$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl1, "SLRL", N_RUNS, display_name="$SLRL$", force=force, threads=0, debug=debug))

    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne1, "SLRLNE_f", N_RUNS, display_name="$SLRLNE_f$", force=force, threads=0, debug=debug))
    trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne2, "SLRLNE", N_RUNS, display_name="$SLRLNE$", force=force, threads=0, debug=debug))

    #plot_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    eval_data = print_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    print_per_sequence_results(trackers, dummy_dataset,"NE vs SLRL - dummy dataset",  merge_results=True, force_evaluation=True, skip_missing_seq=True)
    """
