import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.distributions import Categorical
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from pytracking.evaluation.data import Sequence
from ltr.data.image_loader import opencv_loader as image_loader
from ltr.data.sampler import RandomSequenceSampler

from .models import ADNet, ADNetEnhanced
from .utils import overlap_ratio, extract_region, RegionExtractor, SampleGenerator
import torch.nn as nn

from torchsummary import summary

def identity_func(x): return x

def clamp_bbox(bbox, img_size):
    bbox[2:] = np.clip(bbox[2:], 10, img_size - 10)
    bbox[:2] = np.clip(bbox[:2], 0, img_size - bbox[2:] - 1)
    return bbox


def perform_action(bbox, action_idx, params):
    # Perform action on bounding box(es).
    # - bbox: 1d array of     [x,y,w,h] or
    #         2d array of N x [x,y,w,h]
    # - action_idx: scalar or array of size N (action for each bbox)
    is_1d = bbox.ndim == 1
    if is_1d:
        bbox = bbox[np.newaxis, :]

    opts = params.opts['action_move']
    x, y, w, h = [bbox[:, i] for i in range(4)]

    deltas = np.array([ opts['x'] * w, opts['y'] * h, opts['w'] * w, opts['h'] * h ]).T
    deltas = np.maximum(deltas, 1)
    
    aspect_ratio = w / h
    iff = w > h
    deltas[iff, 3] = deltas[iff, 2] / aspect_ratio[iff]
    deltas[~iff, 2] = deltas[~iff, 3] * aspect_ratio[~iff]
    # if w > h:
    #     deltas[3] = deltas[2] / aspect_ratio
    # else:
    #     deltas[2] = deltas[3] * aspect_ratio
    
    action_delta = opts['deltas'][action_idx, :] * deltas  # Element-wise product
    bbox_next = bbox.copy()
    bbox_next[:, :2] += 0.5 * bbox_next[:, 2:]  # Center
    bbox_next += action_delta
    bbox_next[:, :2] -= 0.5 * bbox_next[:, 2:]  # Un-center

    # NOTE: Original code performs clamping here (necessary? aspect ratio changes ...)

    if is_1d:
        return bbox_next.squeeze()
    return bbox_next


def generate_action_labels(bbox, samples, params):
    # Return the indexes of the best actions when moving samples towards bbox.

    # Calculate overlap between bbox and samples for all actions.
    overlaps = np.zeros((samples.shape[0], params.num_actions))
    for a in range(params.num_actions):
        overlaps[:, a] = overlap_ratio(bbox, perform_action(samples, a, params))
    
    # Check translation actions only
    max_values = np.max(overlaps[:, :-2], axis=1)
    max_actions = np.argmax(overlaps[:, :-2], axis=1)

    # Stop action if close enough
    stop_actions = overlaps[:, params.opts['stop_action']] > params.opts['stopIou']
    max_actions[stop_actions] = params.opts['stop_action']

    # Allow scaling actions if stop action is best
    stop_actions = max_values == overlaps[:, params.opts['stop_action']]
    max_actions[stop_actions] = np.argmax(overlaps[stop_actions], axis=1)

    return max_actions


# Outputs batches of positive/negative training samples for SL training.
# TODO precompute labels and use multiple workers if serious about training 
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, params):
        self.params = params

        self.seq = sequence
        self.frames = np.array(self.seq.frames)  # Image paths

        # Batch size = batch frames * (batch pos + batch neg)

        self.index = torch.randperm(len(self.seq.frames))  # Shuffle frames
        self.pointer = 0

        self.crop_size = params.img_size

        image = image_loader(self.seq.frames[0])
        img_size = np.array(image.shape[1::-1])
        self.pos_generator = SampleGenerator('gaussian', img_size,
                params.trans_pos, params.scale_pos)
        self.neg_generator = SampleGenerator('gaussian', img_size,
                params.trans_neg, params.scale_neg)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        idx = self.index[index]
        image = image_loader(self.frames[idx])
        bbox = self.seq.ground_truth_rect[idx]
        pos_samples = self.pos_generator(bbox, self.params.n_pos_train, overlap_range=self.params.overlap_pos_train)
        neg_samples = self.neg_generator(bbox, self.params.n_neg_train, overlap_range=self.params.overlap_neg_train)
        pos_patches = RegionExtractor(image, pos_samples, self.params)()
        neg_patches = RegionExtractor(image, neg_samples, self.params)()
        pos_action_labels = torch.from_numpy(generate_action_labels(bbox, pos_samples, self.params))
        pos_score_labels = torch.tensor(1).expand(pos_patches.size(0))
        neg_score_labels = torch.tensor(0).expand(neg_patches.size(0))
        return (pos_patches, pos_action_labels, pos_score_labels), (neg_patches, neg_score_labels)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.frames):
            self.pointer = 0
            raise StopIteration

        next_pointer = min(self.pointer + self.params.batch_frames, len(self.frames))
        idx = self.index[self.pointer:next_pointer]
        self.pointer = next_pointer
        
        data = [self[i] for i in idx]
        return SequenceDataset._collate(data)

    @staticmethod
    def _collate(data):
        pos_patches = torch.cat([d[0][0] for d in data], dim=0)
        pos_action_labels = torch.cat([d[0][1] for d in data], dim=0)
        pos_score_labels = torch.cat([d[0][2] for d in data], dim=0)
        neg_patches = torch.cat([d[1][0] for d in data], dim=0)
        neg_score_labels = torch.cat([d[1][1] for d in data], dim=0)
        return (pos_patches, pos_action_labels, pos_score_labels), (neg_patches, neg_score_labels)


def classification_accuracy(model, db, device, params):
    if len(db) == 0: return 0, 0

    model.eval()

    num_correct_actions = 0
    num_correct_scores = 0
    num_actions = 0
    num_scores = 0
    with torch.no_grad():
        for sequence in db:
            #for pos_data, neg_data in SequenceDataset(sequence, params):
            for pos_data, neg_data in DataLoader(SequenceDataset(sequence, params), shuffle=False, batch_size=params.batch_frames, pin_memory=True, collate_fn=SequenceDataset._collate, num_workers=2):
                pos_patches, pos_actions, pos_scores = pos_data
                neg_patches, neg_scores = neg_data
                
                # plot_image(pos_patches[0].permute(1, 2, 0).numpy().astype(np.uint8) + 128)
                # plot_image(neg_patches[0].permute(1, 2, 0).numpy().astype(np.uint8) + 128)
                # print(pos_actions[0])

                pos_patches = pos_patches.to(device) 
                pos_actions = pos_actions.to(device)
                pos_scores = pos_scores.to(device)

                neg_patches = neg_patches.to(device)
                neg_scores = neg_scores.to(device)

                action_history_oh_zero = torch.tensor(0.0).expand(pos_patches.size(0), model.action_history_size).to(device)
                out_actions, out_scores = model(pos_patches, action_history_oh_zero)
                num_correct_actions += (out_actions.argmax(dim=1) == pos_actions).sum().item()
                num_actions += len(pos_actions)
                num_correct_scores += (out_scores.argmax(dim=1) == pos_scores).sum().item()
                num_scores += len(pos_scores)

                action_history_oh_zero = torch.tensor(0.0).expand(neg_patches.size(0), model.action_history_size).to(device)
                _, out_scores = model(neg_patches, action_history_oh_zero)
                num_correct_scores += (out_scores.argmax(dim=1) == neg_scores).sum().item()
                num_scores += len(neg_scores)

    return num_correct_actions / num_actions, num_correct_scores / num_scores


def latest_checkpoint(path):
    checkpoints = list(path.glob("checkpoint_*.pth"))
    if len(checkpoints) > 0:
        epoch = max(map(int, (p.stem.rsplit('_')[1] for p in checkpoints)))
        return path / f"checkpoint_{epoch}.pth"
    else:
        return None


class TrackerTrainer:
    def __init__(self, params, model_path=None, epoch_checkpoint=1, epochs=10, evaluate_performance_=True, experiment_name="", device="cpu"):
        self.model_path = model_path
        self.params = params 
        self.epoch_checkpoint = epoch_checkpoint
        self.n_epochs = epochs
        self.evaluate_performance_ = evaluate_performance_
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.checkpoint_path = params.checkpoints_path / experiment_name
        self.stats = dict(total_epoch_loss=0, avg_epoch_loss=[], epoch=0)

        # Either provide model_path or use load_checkpoint()
        if params.use_efficientnet:
            self.model = ADNetEnhanced(n_actions=params.num_actions, n_action_history=params.num_action_history)
        else:
            self.model = ADNet(n_actions=params.num_actions, n_action_history=params.num_action_history)

        if model_path is not None:
            self.model.load_network(model_path, freeze_backbone=False)
            self.model.to(self.device)

        # Set different learning rates for backbone and FC layers.
        self.optimizer = torch.optim.SGD([
            {'params': self.model.backbone.parameters(), 'lr': params.lr_backbone},
            {'params': self.model.fc4_5.parameters()},
            {'params': self.model.fc6.parameters()},
            {'params': self.model.fc7.parameters()}],
            lr=params.lr_train, momentum=params.momentum, weight_decay=params.weight_decay)

        self.decay_learning_rate = False
        if self.decay_learning_rate:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.85)

    def train_epoch(self, train_db, test_db):
        pass

    def train(self, train_db, test_db=[]):
        if self.stats['epoch'] >= self.n_epochs:
            return self.model

        while self.stats['epoch'] < self.n_epochs:
            print("=" * 40, "Starting epoch %d" % (self.stats['epoch'] + 1), "=" * 40)
            self.stats['epoch'] += 1
            self.stats['total_epoch_loss'] = 0.0

            self.model.train()

            if self.decay_learning_rate:
                self.scheduler.step()

            self.train_epoch(train_db, test_db)
            self.stats['avg_epoch_loss'].append(self.stats['total_epoch_loss'] / len(train_db))

            if self.stats['epoch'] % self.epoch_checkpoint == 0:
                self.save_checkpoint()
            if self.evaluate_performance_:
                self.evaluate_performance(train_db, test_db)
            self.print_stats()

        # Save final model
        self.save_checkpoint()

        return self.model

    def evaluate_performance(self, train_db, test_db):
        pass

    def print_stats(self):
        print(self.stats)

    def load_checkpoint(self, path=None):
        if path is None:
            # Resume latest checkpoint
            path = latest_checkpoint(self.checkpoint_path)
            if path is None:
                return False

        print(path)
        state = torch.load(path, map_location=torch.device(self.device))
        """
        for key in list(state["model"].keys()):
            state["model"][key.replace("module.", "")] = state["model"].pop(key)
            #state["model"]["module." + key] = state["model"].pop(key)
        """
        
        self.stats = state['stats']

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state['model'])
            self.model.module.backbone.requires_grad_(True)
        else:
            self.model.load_state_dict(state['model'])
            self.model.backbone.requires_grad_(True)

        self.model.to(self.device)

        self.optimizer.load_state_dict(state['optimizer'])
        
        if 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])

        return True

    def save_checkpoint(self):
        model = self.model if not isinstance(self.model, nn.DataParallel) else self.module.model
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_path / f"checkpoint_{self.stats['epoch']}.pth"

        state = {
            'stats': self.stats,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            # scheduler
        }
        torch.save(state, path)
        print(path)
