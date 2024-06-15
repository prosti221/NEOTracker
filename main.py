import sys

sys.path.append("pytracking")
sys.path.append("pytracking/pytracking")
sys.path.append("pytracking/ltr")
from pathlib import Path
import matplotlib.pyplot as plt

from pytracking.evaluation import Tracker

from pytracking.tracker.adnet.utils import *
from pytracking.tracker.adnet.train_ne import *
from pytracking.tracker.adnet.train_sl import *
from pytracking.tracker.adnet.train import *
from pytracking.tracker.adnet.synthetic import *

import argparse
import yaml
import os

from utils import *

def parse_yaml(path):
    with open(path, "r") as stream:
        try:
            y = yaml.safe_load(stream)
            return y
        except yaml.YAMLError as exc:
            print(exc)

def checkpoint_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError("File {} not found".format(path))
    elif os.path.isdir(path):
        initial_model_path = latest_checkpoint(Path(path))
    else:
        initial_model_path = Path(path)
    
    return initial_model_path

def run(config):
    params = config["params"]
    train_tags = config["datasets"]["train"]
    val_tags = config["datasets"]["val"]
    test_tags = config["datasets"]["test"]


    if train_tags == "dummy":
        dataset = load_dummy_dataset()
    else:
        dataset = load_datasets(
            train_tags=train_tags,
            val_tags=val_tags,
            test_tags=test_tags,
            n_train_sequences=-1,
            n_test_sequences=-1,
            remove_overlapping=True
        )

    mode = config["train_mode"]
    stats = None
    match mode:
        case "NE":
            # If initial model path is a directory, get the latest checkpoint. Otherwise, use the provided path
            # If it doesn't exist, raise an error
            initial_model_path = checkpoint_path(config["train_ne"]["initial_model_path"])
            checkpoints_root_path = Path(config["train_ne"]["checkpoints"])
            checkpoints = get_all_checkpoints(checkpoints_root_path)[:config["train_ne"]["checkpoint_num"]]
            
            stats = train_ne(
                device=config["device"],
                initial_model_path=initial_model_path,
                params=params,
                dataset_train=dataset["train"],
                population_size=config["train_ne"]["population_size"],
                n_generations=config["train_ne"]["n_generations"],
                algorithm=config["train_ne"]["algorithm"],
                experiment_name=config["train_ne"]["experiment_name"],
                checkpoints=checkpoints,
                resume=config["train_ne"]["resume"]
            )

        case "SL":
            initial_model_path = checkpoint_path(config["train_sl"]["initial_model_path"])
            
            stats = train_sl(
                device=config["device"],
                initial_model_path=initial_model_path,
                params=params,
                dataset_train=dataset["train"],
                dataset_val=dataset["val"],
                epochs=config["train_sl"]["epochs"],
                epoch_checkpoint=config["train_sl"]["epoch_checkpoint"],
                evaluate_performance_=config["train_sl"]["evaluate_performance_"],
                experiment_name=config["train_sl"]["experiment_name"]
            )

        case "RL":
            initial_model_path = checkpoint_path(config["train_rl"]["initial_model_path"])
            
            stats = train_rl(
                device=config["device"],
                initial_model_path=initial_model_path,
                params=params,
                dataset_train=dataset["train"],
                dataset_val=dataset["val"],
                epochs=config["train_rl"]["epochs"],
                epoch_checkpoint=config["train_rl"]["epoch_checkpoint"],
                evaluate_performance_=config["train_rl"]["evaluate_performance_"],
                experiment_name=config["train_rl"]["experiment_name"],
                single_layer=config["train_rl"]["single_layer"],
                reset_fc6=config["train_rl"]["reset_fc6"]
            )

        case _:
            raise ValueError(f"Invalid train mode: {mode}")

    return stats

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Training script for ADNet.')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to training config.')

    # Get yaml file path
    yaml_path = Path(parser.parse_args().config)
    config = parse_yaml(yaml_path)
    tracker = Tracker("adnet", "default", run_id=None)
    params = tracker.get_parameters()
    config["params"] = params

    stats = run(config)

    fig, ax = plt.subplots()
    #rl_plot(trainer.stats['avg_epoch_reward_per_genome'], ax, label="NE")
    rl_plot(stats['avg_epoch_reward'], ax, label="avg reward per gen")
    rl_plot(stats['max_epoch_reward'], ax, label="max reward per gen")
    ax.legend()
    plt.show()