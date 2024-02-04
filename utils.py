# Append the working directory to the path
import sys
sys.path.append("pytracking")
sys.path.append("pytracking/pytracking")
sys.path.append("pytracking/ltr")

# Imports from the pytracking modules 
from ltr.data.sampler import RandomSequenceSampler
from pytracking.evaluation.running import run_sequence, run_dataset
from pytracking.evaluation.got10kdataset import GOT10KDataset
from pytracking.evaluation.environment import env_settings
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.tracker.adnet.synthetic import *
from pytracking.tracker.adnet.utils import *
from pytracking.evaluation import Tracker, trackerlist
from pytracking.evaluation.otbdataset import OTBDataset
from pytracking.tracker.adnet.models import ADNet
from pytracking.tracker.adnet.train_ne import *
from pytracking.tracker.adnet.train_sl import *
from pytracking.tracker.adnet.train_rl import *
from pytracking.tracker.adnet.train import *

# Imports related to datasets
import vot.dataset.vot as vot
from vot.dataset import Dataset, DatasetException, BaseSequence
#from vot.dataset import Sequence
from vot.utilities import Progress, localize_path, read_properties
from vot.region import parse
from vot.dataset import download_dataset
from vot.dataset.vot import load_channel

# General imports
from pathlib import Path
from collections import OrderedDict
import numpy as np
import six
import re
import os
import glob
import random

class LaSOTSequence(BaseSequence):
    def __init__(self, base, name=None, dataset=None):
        self._base = base
        if name is None:
            name = os.path.basename(base)
        super().__init__(name, dataset)

    @staticmethod
    def check(path):
        return os.path.isfile(os.path.join(path, 'sequence'))
    
    def _read_metadata(self):
        metadata = dict(fps=30, format="default")
        metadata["channel.default"] = "color"

        metadata_file = os.path.join(self._base, 'sequence')
        metadata.update(read_properties(metadata_file))

        return metadata

    def _read(self):
        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        for c in ["color", "depth", "ir"]:
            channel_path = self.metadata("channels.%s" % c, None)
            if not channel_path is None:
                channels[c] = load_channel(os.path.join(self._base, localize_path(channel_path)))

        # Load default channel if no explicit channel data available
        if len(channels) == 0:
            channels["color"] = load_channel(os.path.join(self._base, "color", "%08d.jpg"))
        else:
            self._metadata["channel.default"] = next(iter(channels.keys()))

        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        groundtruth_file = os.path.join(self._base, self.metadata("groundtruth", "groundtruth.txt"))

        with open(groundtruth_file, 'r') as filehandle:
            for region in filehandle.readlines():
                groundtruth.append(parse(region))

        self._metadata["length"] = len(groundtruth)

        tagfiles = glob.glob(os.path.join(self._base, '*.tag')) + glob.glob(os.path.join(self._base, '*.label'))

        for tagfile in tagfiles:
            with open(tagfile, 'r') as filehandle:
                tagname = os.path.splitext(os.path.basename(tagfile))[0]
                tag = [line.strip() == "1" for line in filehandle.readlines()]
                while not len(tag) >= len(groundtruth):
                    tag.append(False)
                tags[tagname] = tag

        valuefiles = glob.glob(os.path.join(self._base, '*.value'))

        for valuefile in valuefiles:
            with open(valuefile, 'r') as filehandle:
                valuename = os.path.splitext(os.path.basename(valuefile))[0]
                value = [float(line.strip()) for line in filehandle.readlines()]
                while not len(value) >= len(groundtruth):
                    value.append(0.0)
                values[valuename] = value

        for name, channel in channels.items():
            if not channel.length == len(groundtruth):
                raise DatasetException("Length mismatch for channel %s (%d != %d)" % (name, channel.length, len(groundtruth)))

        for name, tag in tags.items():
            if not len(tag) == len(groundtruth):
                tag_tmp = len(groundtruth) * [False]
                tag_tmp[:len(tag)] = tag
                tag = tag_tmp

        for name, value in values.items():
            if not len(value) == len(groundtruth):
                raise DatasetException("Length mismatch for value %s" % name)

        return channels, groundtruth, tags, values

class LaSOTDataset(Dataset):
    def __init__(self, path):
        super().__init__(path)

        # generate list.txt file
        self.__generate_list_file()
        # Rename the img directories to color
        self._rename_img_dirs()

        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(names)) as progress:
            for name in names:
                self._sequences[name.strip()] = LaSOTSequence(os.path.join(path, name.strip()), dataset=self)
                progress.relative(1)
        
    def __generate_list_file(self):
        with open(os.path.join(self.path, "list.txt"), 'w') as fd:
            # Iterate through all subdirectories as store the name
            for subdir in os.listdir(self.path): 
                if os.path.isdir(os.path.join(self.path, subdir)):
                    fd.write(subdir + "\n")

        fd.close()
    
    def _rename_img_dirs(self):
        with open(os.path.join(self.path, "list.txt"), 'r') as fd:
            names = fd.readlines()
        
        for name in names:
            name = name.strip()
            root = os.path.join(self.path, name)
            
            # Check if the root contains a directory called "img"
            # If it does, rename it to "color"
            if os.path.isdir(os.path.join(root, "img")):
                os.rename(os.path.join(root, "img"), os.path.join(root, "color"))
        
    @staticmethod
    def check(path: str):
        if not os.path.isfile(os.path.join(path, 'list.txt')):
            return False

        with open(os.path.join(path, 'list.txt'), 'r') as handle:
            sequence = handle.readline().strip()
            return LaSOTSequence.check(os.path.join(path, sequence))

    @property
    def path(self):
        return self._path

    @property
    def length(self):
        return len(self._sequences)

    def __getitem__(self, key):
        return self._sequences[key]

    def __contains__(self, key):
        return key in self._sequences

    def __iter__(self):
        return self._sequences.values().__iter__()

    def list(self):
        return list(self._sequences.keys())

             

class LaSOTDatasetWrapped(BaseDataset):
    def __init__(self, path):
        super().__init__()
        self.base_path = path
        self.dataset = LaSOTDataset(path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.dataset])

    def _construct_sequence(self, sequence):
        frames = [frame.filename() for frame in sequence]
        ground_truth_rect = []
        for frame in sequence:
            gt = frame.groundtruth().convert(RegionType.RECTANGLE)
            if gt.is_empty():
                ground_truth_rect.append([-1, -1, -1, -1])
            else:
                ground_truth_rect.append([gt.x, gt.y, gt.width, gt.height])
        ground_truth_rect = np.array(ground_truth_rect)
        return Sequence(sequence.name, frames, 'vot', ground_truth_rect)

    def __len__(self):
        return self.dataset.length

    def __getitem__(self, key):
        # Access by sequence name (e.g. 'fish')
        return self._construct_sequence(self.dataset[key])

def shorten_dataset(dataset_path, n_frames):
    """
    Iterates over all samples of the LaSOT dataset and shortens them to n_frames
    Problem: The dataset is too large and a lot of sequences are very long
    """
    VALID_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]

    # Iterate over all sequences
    for seq in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, seq)):
            continue
        frames_path = os.path.join(dataset_path, seq, "color")
        files = [file for file in os.listdir(frames_path) if any([file.endswith(ext) for ext in VALID_EXTENSIONS])]
        files = sorted(files, key= lambda x: int(x.split(".")[0]))
        if len(files) > n_frames:
            print("Shortening sequence {} from {} to {}".format(seq, len(files), n_frames))
            # Remove the last frames
            for file in files[n_frames:]:
                os.remove(os.path.join(frames_path, file))
                #print("Yeeted and deleted {}".format(os.path.join(frames_path, file)))
            # Remove the corresponding groundtruth
            gt_path = os.path.join(dataset_path, seq, "groundtruth.txt")
            with open(gt_path, 'r') as fd:
                lines = fd.readlines()
            lines = lines[:n_frames]
            with open(gt_path, 'w') as fd:
                fd.writelines(lines)
                #print("Yeeted and deleted {}".format(gt_path))



class VOTDatasetWrapped(BaseDataset):
    """
    Wrap VOTDataset from vot-toolkit.
    """
    def __init__(self, path):
        super().__init__()
        self.base_path = self.env_settings.vot_path
        self.dataset = vot.VOTDataset(path)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.dataset])

    def _construct_sequence(self, sequence):
        frames = [frame.filename() for frame in sequence]
        ground_truth_rect = []
        for frame in sequence:
            gt = frame.groundtruth().convert(RegionType.RECTANGLE)
            if gt.is_empty():
                ground_truth_rect.append([-1, -1, -1, -1])
            else:
                ground_truth_rect.append([gt.x, gt.y, gt.width, gt.height])
        ground_truth_rect = np.array(ground_truth_rect)
        return Sequence(sequence.name, frames, 'vot', ground_truth_rect)

    def __len__(self):
        return self.dataset.length

    def __getitem__(self, key):
        # Access by sequence name (e.g. 'fish')
        return self._construct_sequence(self.dataset[key])

def evaluate_tracker(dataset, model_path, experiment_name, iters, display_name=None, **kwargs):
    trackers = []
    for i in range(iters):
        tracker = Tracker('adnet', 'only_track', run_id=i, display_name=display_name, params_callback=lambda p: setattr(p, 'model_path', model_path), experiment_name=experiment_name)
        print(tracker)
        trackers.append(tracker)
    # Run run_dataset in parallel
    run_dataset(dataset, trackers, **kwargs)

    return trackers 

def rl_plot(x, ax, label, extend=None, only_avg=False, alpha=0.5, avg_alpha=1.0, n_iou=20, **kwargs):
    x = np.array(x)
    if not only_avg:
        p = ax.plot(x, alpha=alpha, **kwargs)
    avg = rolling_average(x, last_k=3)
    iou = x[-n_iou:].max()
    if not only_avg:
        p = ax.plot(avg, label=f"{label} [{iou:.3f}]", color=p[-1].get_color(), alpha=avg_alpha, **kwargs)
    else:
        p = ax.plot(avg, label=f"{label} [{iou:.3f}]", alpha=avg_alpha, **kwargs)
    if extend:
        ax.plot(np.arange(len(x), extend), [iou] * (extend - len(x)), '--', color=p[-1].get_color(), alpha=avg_alpha, **kwargs)
    return iou

def generate_composite_dataset(dataset_tags, n_sequences=10):
    """
    Generates a composite dataset from the given dataset tags (e.g. vot2014, vot2015, vot2017, got_10k-val, lasot)
    Input:
        dataset_tags : List of dataset tags
        n_sequences  : Number of sequences to sample from each dataset. -1 means all sequences

    Returns:
        A new dataset containing the sequences from the given dataset tags
    """
    sequences = []
    for tag in dataset_tags:
        # Check first if dataset is already downloaded
        dataset = None
        if "vot" in tag:
            if not Path(f"datasets/{tag}").exists():
                download_dataset(tag, f"datasets/{tag}")
            dataset = VOTDatasetWrapped(f"datasets/{tag}")
        elif "got_10k" in tag:
            dataset_name, dataset_type = tag.split('-')
            dataset = GOT10KDataset(dataset_type, f"datasets/{dataset_name}")
        elif "lasot" in tag:
            if not Path(f"datasets/{tag}").exists():
                raise ValueError(f"Dataset {tag} not found. Please download the dataset manually.")
            dataset = LaSOTDatasetWrapped(f"datasets/{tag}")
            shorten_dataset(f"datasets/{tag}", 500)
        elif "otb" in tag:
            dataset = OTBDataset(f"datasets/otb/{tag}")
            
        print(f"Loaded {len(dataset)} sequences from {tag}")
        sequences.extend(dataset.get_sequence_list())

    if n_sequences == -1:
        return sequences

    sequences = random.sample(sequences, n_sequences)
    print(f"\nGenerated composite dataset with {len(sequences)} sequences")

    return sequences


def remove_overlapping_sequences(dataset):
    """
    Makes sure that the composite dataset does not contain overlapping sequences
    Some of the VOT dataset especially contains overlapping sequences (e.g. vot2014 and vot2015)
    """
    ret = []
    for seq in dataset:
        if seq.name not in [s.name for s in ret]:
            ret.append(seq)
    
    print(f"Removed {len(dataset) - len(ret)} overlapping sequences")
    print(f"Dataset size: {len(ret)}")

    return ret
        
def get_sequence_by_name(dataset, name):
    for seq in dataset:
        if seq.name == name:
            return seq
    # Throw error if sequence not found
    raise ValueError(f"Sequence {name} not found in dataset")

def get_unique_sequences(dataset1, dataset2):
    """
    Given two datasets, returns a list of sequences that are unique to dataset2
    """
    ret = []
    for seq in dataset2:
        if seq.name not in [s.name for s in dataset1]:
            ret.append(seq)

    print(f"Removed {len(dataset2) - len(ret)} overlapping sequences")
    print(f"Dataset size: {len(ret)}")

    return ret

def get_all_checkpoints(root):
    checkpoints = []
    for path in root.iterdir():
        #if path.is_dir():
        #    checkpoints += get_all_checkpoints(path)
        if path.is_file() and path.suffix == ".pth":
            checkpoints.append(path)

    return sorted(checkpoints, reverse=True)

def load_dummy_dataset():
    dataset = {}
    dummy = DummyDataset()
    dummy_sequence = dummy.get_sequence_list()[0]
    dummy_dataset = [dummy_sequence]

    dataset["train"] = dummy_dataset
    dataset["val"] = None
    dataset["test"] = None

    return dataset

def load_datasets(train_tags,  val_tags=[], test_tags=[], n_train_sequences=-1, n_val_sequences=-1, n_test_sequences=-1, remove_overlapping=True):
    """
    Loads the datasets from the given tags and returns a dictionary containing the datasets
    The dictionary contains the following keys:
        train : Training dataset
        val   : Validation dataset
        test  : Testing dataset
    Input:
        train_tags         : List of tags for the training dataset
        val_tags           : List of tags for the validation dataset
        test_tags          : List of tags for the testing dataset
        n_train_sequences  : Number of sequences to sample from the training dataset. -1 means all sequences
        n_val_sequences    : Number of sequences to sample from the validation dataset. -1 means all sequences
        n_test_sequences   : Number of sequences to sample from the testing dataset. -1 means all sequences
        remove_overlapping : If True, removes overlapping sequences from the datasets
    """
    VALID_DATASETS = ["vot2013", "vot2014", "vot2015", 
                      "vot2017","got_10k-test", "got_10k-val", 
                      "vot-st2019", "vot-st2020", "vot-st2021",
                      "lasot", "otb50", "otb100"
                    ]
    datasets = {}
    for tag in train_tags + val_tags + test_tags:
        if tag not in VALID_DATASETS:
            raise ValueError("Invalid dataset tag: {}".format(tag))

    # Joint the datasets together and ensure there are no duplicates
    datasets["train"] = generate_composite_dataset(train_tags, n_sequences=int(n_train_sequences))
    datasets["test"] = generate_composite_dataset(test_tags, n_sequences=int(n_test_sequences))
    datasets["val"] = generate_composite_dataset(val_tags, n_sequences=int(n_val_sequences))

    if remove_overlapping:
        # Remove duplicates
        datasets["train"] = remove_overlapping_sequences(datasets["train"])
        datasets["val"] = remove_overlapping_sequences(datasets["val"])
        datasets["test"] = remove_overlapping_sequences(datasets["test"])

        # Make sure that there are no common sequences between datasets
        datasets["test"] = get_unique_sequences(datasets["train"] + datasets["val"], datasets["test"])
        datasets["val"] = get_unique_sequences(datasets["train"], datasets["val"])

    return datasets

def filter_sequences(dataset, report_path, cutoff=0.04):
    """
    Gets rid of samples from the dataset that does not train well based on the report file.
    Input:
        dataset     : List of sequences
        report_path : Path to the report file
        cutoff      : Cutoff value for the score

    Returns:
        The filtered dataset
    """
    regex = re.compile(r"\((\d+), '([^']+)'\): \((\[[^]]+\]), (-?\d+\.\d+)\)")
    with open(report_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    samples = [] # Index of the samples to save
    for line in lines:
        match = regex.match(line)
        try:
            sample_idx, seq_name, score = match.group(1), match.group(2), match.group(4)
        except Exception as e:
            print(e)
            continue

        if float(score) >= cutoff:
            samples.append(int(sample_idx))

    return [sample for i, sample in enumerate(dataset) if i in samples]

def train_ne(
        initial_model_path, 
        params, 
        dataset_train, 
        population_size, 
        n_generations, 
        algorithm="FIXED", 
        experiment_name="NE", 
        dataset_eval=None, 
        device="cpu", 
        checkpoints=[], 
        resume=False,
        **kwargs
    ):

    start_from = initial_model_path
    
    trainer = TrainTracker_NE(
        params, 
        algorithm=algorithm,
        model_path=start_from, 
        population_size=population_size,
        n_generations=n_generations,
        maxpool_size=1, 
        experiment_name=experiment_name,
        device=device, 
        reward_version="MATLAB", 
        #reward_version="PAPER", 
        resume_from_checkpoint=resume,
        checkpoint_paths=checkpoints
    )

    trainer.train(dataset_train, dataset_eval)

    return trainer.stats

def train_sl(
        initial_model_path, 
        params, 
        dataset_train, 
        epochs, 
        epoch_checkpoint=5,
        dataset_val=None, 
        evaluate_performance_=False,
        experiment_name="SL",
        device="cpu",
        **kwargs
    ):
    if evaluate_performance_ and dataset_val is None:
        raise ValueError("dataset_val must be provided if evaluate_performance_ is True")

    trainer = TrainTracker_SL(
        params=params, 
        model_path=initial_model_path, 
        epochs=epochs,
        epoch_checkpoint=epoch_checkpoint,
        evaluate_performance_=evaluate_performance_,
        experiment_name=experiment_name,
        device=device
    )
    trainer.load_checkpoint()

    if evaluate_performance_ and dataset_val is not None:
        trainer.train(dataset_train, dataset_val)
    else:
        trainer.train(dataset_train)
    
    return trainer.stats

def train_rl(
        initial_model_path, 
        params, 
        dataset_train, 
        epochs, 
        epoch_checkpoint=5,
        dataset_val=None, 
        evaluate_performance_=False,
        experiment_name="SL",
        device="cpu",
        single_layer=False,
        reset_fc6 = False,
        **kwargs
    ):

    trainer = TrainTracker_RL(
        params=params, 
        model_path=initial_model_path, 
        epochs=epochs, 
        epoch_checkpoint=epoch_checkpoint, 
        experiment_name=experiment_name, 
        device=device, 
        single_layer=single_layer, 
        reset_fc6=reset_fc6
    )

    trainer.load_checkpoint()

    if evaluate_performance_ and dataset_val is not None:
        trainer.train(dataset_train, dataset_val)
    else:
        trainer.train(dataset_train)

    return trainer.stats
