import sys
sys.path.append("pytracking")
sys.path.append("pytracking/pytracking")
sys.path.append("pytracking/ltr")
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from pytracking.evaluation import Tracker

from pytracking.tracker.adnet.utils import *
from pytracking.tracker.adnet.train_rl import *
from pytracking.tracker.adnet.train_sl import *
from pytracking.tracker.adnet.train import *
from pytracking.tracker.adnet.synthetic import make_curriculum

from utils import *

wandb.init(
    project="reference-models",
    job_type="train",
    group="RL",
    name="RL-run1",
)

if __name__ == '__main__':
    tracker = Tracker("adnet", "default", run_id=None)
    params = tracker.get_parameters()
    device = torch.device('cuda')
    #device = torch.device('cpu')
    model_path = r'networks/imagenet-vgg-m.mat'
    model_path = Path(model_path)

    dataset = load_datasets(
        #train_tags=["vot2014", "vot2015", "vot2017","vot-st2020", "lasot"],
        train_tags=["vot2014", "vot2015","vot2017","vot-st2020","lasot","got_10k-val"],
        #val_tags=["vot-st2021"],
        val_tags=["got_10k-train"],
        test_tags=["got_10k-val"],
        n_train_sequences=-1,
        n_val_sequences=-1,
        n_test_sequences=-1,
        remove_overlapping=True
    )
    #dataset["test"] = dataset["test"][:30] 
    dataset["val"] = dataset["val"][:50] 
    dataset["train"] = filter_sequences(dataset["train"], report_path=Path.cwd()/"sample_progress.txt", cutoff=0.038) # gets us 97 samples
    print(len(dataset["train"]))
    print(len(dataset["val"]))
    """
    trainer = TrainTracker_SL(params, model_path=model_path, epochs=60, epoch_checkpoint=1, evaluate_performance_=True, experiment_name="SL-visualize2", device=device)
    trainer.load_checkpoint()
    trainer.train(dataset["train"], dataset["val"])
    """

    #start_from = latest_checkpoint(params.checkpoints_path / "SL-XL")
    #start_from = latest_checkpoint(params.checkpoints_path / "RL-XL")
    start_from = model_path

    trainer = TrainTracker_RL(params, model_path=start_from, epochs=200, epoch_checkpoint=1, experiment_name="SLRL_visualize", device=device, single_layer=False, reset_fc6=False)
    trainer.load_checkpoint()
    trainer.train(dataset["train"])

    """
    fig, ax = plt.subplots()
    rl_plot(trainer.stats['avg_epoch_reward'], ax, label="SLRL-mk2")
    ax.legend()
    plt.show()

    accs = [val[0] for val in trainer.stats['acc_list']]
    fig, ax = plt.subplots()
    rl_plot(accs, ax, label="SL", only_avg=True)
    ax.legend()
    plt.show()

    p_target_rl1 = latest_checkpoint(params.checkpoints_path / "RL-XL")
    p_target_rl2 = latest_checkpoint(params.checkpoints_path / "RL-mk2")
    #p_target_sl = latest_checkpoint(params.checkpoints_path / "SL-XL")
    #p_target_ne = params.checkpoints_path / "NE-evotorch" / "pytorch_checkpoints/two_sample_batch_size_2.pth"

    N_RUNS = 1
    force = True
    trackers = []
    debug = 0 

    #trackers.extend(evaluate_tracker(dataset["test"], p_target_rl1, "SLRL-1", N_RUNS, display_name="$SLRL-1$", force=force, threads=0, debug=debug))
    trackers.extend(evaluate_tracker(dataset["test"], p_target_rl2, "SLRL-2", N_RUNS, display_name="$SLRL-2", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dataset["train"], p_target_sl, "SL", N_RUNS, display_name="$SL$", force=force, threads=0, debug=debug))
    #trackers.extend(evaluate_tracker(dataset["train"], p_target_ne, "NE", N_RUNS, display_name="$NE$", force=force, threads=0, debug=debug))

    #plot_results(trackers, dataset["train"], "test", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    eval_data = print_results(trackers,dataset["train"], "test", merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True, skip_missing_seq=True)
    print_per_sequence_results(trackers,dataset["train"], "test", merge_results=True, force_evaluation=True, skip_missing_seq=True)
    """
