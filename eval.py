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
    start_from = latest_checkpoint(params.checkpoints_path / "SL-XL")

    dataset = load_datasets(
        #train_tags=["vot2014", "vot2015","vot2017","vot-st2020","lasot","got_10k-val"],
        train_tags=["got_10k-val"],
        #val_tags=["vot-st2021"],
        #val_tags=["got_10k-train"],
        n_train_sequences=-1,
        n_val_sequences=-1,
        n_test_sequences=-1,
        remove_overlapping=True
    )
    #dummy_dataset = load_dummy_dataset()

    #dummy_dataset = dataset["train"][:50]
    #dataset["train"] = half_sequences(dataset["train"][:200])
    dataset["train"] = dataset["train"][:50]
    #dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=0.038) # gets us 97 samples
    dummy_dataset = dataset["train"]
    print(len(dataset["train"]))
    
    """
    generate_ensemble(
            root_dir=Path(__file__).parent / "final_experiments_mk2/SLRLNE/models/saved_populations", 
            #root_dir=Path(__file__).parent / "networks/adnet_checkpoints/NE-evotorch/pytorch_checkpoints/saved_populations", 
            dst=params.checkpoints_path / "ensemble", 
            generation_st=0, 
            generation_end=26, 
            k=85
    )
    """

    p_target_sl = latest_checkpoint(params.checkpoints_path / "SL-XL")

    p_target_rl1 = latest_checkpoint(params.checkpoints_path / "RL-XL")
    #p_target_rl2 = latest_checkpoint(params.checkpoints_path / "SLRL_fc6_f")
    #p_target_rl3 = latest_checkpoint(params.checkpoints_path / "SLRL+RL_fc6_f")
    p_target_rl4 = latest_checkpoint(params.checkpoints_path / "RL-mk2")
    p_target_rl5 = latest_checkpoint(params.checkpoints_path / "RL-mk3")

    p_target_ne1 = params.checkpoints_path / "SLNE_mk1" / "best_genome_medium.pth"
    p_target_ne2 = Path(__file__).parent / "final_experiments_mk2/SLRLNE/models/best_genome_540.pth"
    p_target_ne3 = Path(__file__).parent / "final_experiments/SLNE/models/best_genome_4462.pth"
    p_target_ne4 = Path(__file__).parent / "final_experiments/SLNE_f/models/best_genome_4462.pth"

    N_RUNS = 1
    force = True
    trackers = []
    debug = 0
    # For evaluating all the ne checkpoint.pth"
    p_target_rl5 = latest_checkpoint(params.checkpoints_path / "RL-XL")
    trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl5, "SLRL-mk3", N_RUNS, display_name="$SLRL-mk3$", force=force, threads=0, debug=debug))
    eval_data = print_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    print_per_sequence_results(trackers, dummy_dataset,"NE vs SLRL - dummy dataset",  merge_results=True, force_evaluation=True, skip_missing_seq=True)
    exit()
    #evaluate_all_checkpoints(params, dataset, checkpoint_path=params.checkpoints_path / "NE-evotorch" / "pytorch_checkpoints")
    #evaluate_all_checkpoints(params, dataset, Path(__file__).parent / "final_experiments" / "SLNE_f" / "models")

    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_sl, "SL", N_RUNS, display_name="$SL$", force=force, threads=0, debug=debug))

    trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl5, "SLRL-mk3", N_RUNS, display_name="$SLRL-mk3$", force=force, threads=0, debug=debug))
    trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl4, "SLRL-mk2", N_RUNS, display_name="$SLRL-mk2$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl3, "SLRL+RL_fc6", N_RUNS, display_name="$SLRL+RL_fc6$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl2, "SLRL_fc6_f", N_RUNS, display_name="$SLRL_fc6$", force=force, threads=0, debug=debug))
    trackers.extend(evaluate_tracker(dummy_dataset, p_target_rl1, "SLRL", N_RUNS, display_name="$SLRL$", force=force, threads=0, debug=debug))

    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne1, "SLRLNE_f", N_RUNS, display_name="$SLRLNE_f$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne2, "SLRLNE", N_RUNS, display_name="$SLRLNE$", force=force, threads=0, debug=debug))

    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne3, "SLNE", N_RUNS, display_name="$SLNE$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dummy_dataset, p_target_ne4, "SLNE_f", N_RUNS, display_name="$SLNE_f$", force=force, threads=0, debug=debug))

    #plot_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    eval_data = print_results(trackers, dummy_dataset, "NE vs SLRL - dummy dataset", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    print_per_sequence_results(trackers, dummy_dataset,"NE vs SLRL - dummy dataset",  merge_results=True, force_evaluation=True, skip_missing_seq=True)
