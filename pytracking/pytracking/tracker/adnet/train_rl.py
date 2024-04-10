from .train import TrackerTrainer, clamp_bbox, perform_action, one_hot, identity_func, image_loader
from .utils import overlap_ratio, extract_region
from ltr.data.sampler import RandomSequenceSampler
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import colorama
import wandb
from tqdm import tqdm


class TrainTracker_RL(TrackerTrainer):
    REWARD_VERSIONS = ["PAPER", "MATLAB", "RETURNS", "ALL_OVERLAPS", "ALL_REWARDS"]

    def randomize_fc6(self, fc6):
        nn.init.normal_(fc6.weight, mean=0, std=0.01)
        nn.init.constant_(fc6.bias, 0.1)

    def __init__(self, params, reward_version="RETURNS", single_layer=False, reset_fc6 = False, **kwargs):
        super().__init__(params, **kwargs)
        
        self.n_epochs = kwargs.get('epochs', params.n_epochs_rl)
        self.epoch_checkpoint = kwargs.get('epoch_checkpoint', params.checkpoint_interval_rl)

        self.stats['reward_version'] = reward_version
        self.stats['rewards'] = []
        self.stats['actions'] = []
        self.stats['avg_epoch_reward'] = []

        if reset_fc6:
            self.randomize_fc6(self.model.fc6[0])

        # Only optimize the FC6 layer
        if single_layer:
            print("Only optimizing FC6 layer")
            self.optimizer = torch.optim.Adam([
                {'params': self.model.fc6.parameters()}],
                lr=params.lr_train)

    def extract_region(self, image, bbox):
        return extract_region(image, bbox, crop_size=self.params.img_size, 
                                           padding=self.params.padding, 
                                           means=self.params.means)

    def track_frame(self, image, bbox, action_history_oh):
        # Perform actions until convergence in current frame (for RL).
        params = self.params 
        opts = params.opts

        # self.model.eval()

        img_size = np.array(image.shape[1::-1])

        log_probs = []
        bboxes = [bbox]
        
        # For oscillation checking
        round_bboxes = set()
        round_bboxes.add(tuple(bbox.round()))

        move_counter = 0
        prev_action = -1
        curr_bbox = bbox
        while move_counter < opts['num_action_step_max'] and prev_action != opts['stop_action']:
            curr_patch = self.extract_region(image, curr_bbox).to(self.device)

            actions, conf = self.model(curr_patch, action_history_oh, log_softmax=False)
            prob_actions = Categorical(probs=actions)  # For action sampling

            # NOTE more correct to always sample actions instead of taking the max (like in matlab code)
            # However in evaluation tracking, I take just the max.
            action = prob_actions.sample()
            action_idx = action.item()
            log_probs.append(prob_actions.log_prob(action))
            self.stats['actions'].append(action_idx)

            next_bbox = perform_action(curr_bbox, action_idx, params)
            next_bbox = clamp_bbox(next_bbox, img_size)

            # Check for oscillations
            next_bbox_round = tuple(next_bbox.round())
            if move_counter > 0 and action_idx != opts['stop_action'] and next_bbox_round in round_bboxes:
                action_idx = opts['stop_action']  # Don't store this action because it wasn't picked by the model
                self.stats['actions'].append(action_idx)

            # Update one-hot action history
            n = self.params.num_actions
            # Clone to avoid in-place modification
            action_history_oh[0, n:] = action_history_oh[0, :-n].clone()
            action_history_oh[0, :n] = one_hot(torch.tensor(action_idx), num_classes=n)
    

            curr_bbox = next_bbox
            bboxes.append(curr_bbox)
            round_bboxes.add(next_bbox_round)
            prev_action = action_idx
            move_counter += 1

        # self.stats['move_counter'] = move_counter
        return np.stack(bboxes), log_probs

    def calc_weights(self, sim_overlaps, version):
        if version == "PAPER":
            # The reward z_t,l for each action is computed only from the terminal state! (in paper)
            weights = []
            for overs in sim_overlaps:
                reward = 1 if overs[-1] >= 0.7 else -1  # Final box overlap
                weights.append(reward * torch.ones(len(overs)))
            weights = torch.cat(weights)
        elif version == "MATLAB":
            # In matlab code, the reward is computed for the whole sequence from just the final (gt, bbox) pair!
            # ADNet matlab version: reward whole sequence based on final frame IoU
            n = sum(len(x) for x in sim_overlaps)
            reward = 1 if sim_overlaps[-1][-1] >= 0.7 else -1
            weights = reward * torch.ones(n)
        elif version == "RETURNS":
            # Calculate rewards-to-go
            rewards = np.concatenate(sim_overlaps)
            R = 0
            weights = []
            for r in rewards[::-1]:
                R = r + self.params.rl_gamma * R
                weights.append(R)
            weights = torch.tensor(weights[::-1])
            weights = (weights - weights.mean()) / (weights.std() + 1e-9)  # normalize discounted rewards
        elif version == "ALL_OVERLAPS":
            # reward == overlap
            weights = torch.from_numpy(np.concatenate(sim_overlaps)).float()
        elif version == "ALL_REWARDS":
            # 1, -1 based on overlap
            weights = torch.from_numpy(np.concatenate(sim_overlaps)).float()
            idx = weights >= 0.7
            weights[idx] = 1
            weights[~idx] = -1
        return weights

    def train_epoch(self, train_db, test_db):
        self.model.train()

        total_epoch_reward = 0
        total_epoch_loss = 0

        #db_idx = np.random.permutation(len(train_db))  # Shuffle sequences
        #db_idx = np.random.permutation(30)  # Shuffle sequences
        db_idx = np.arange(len(train_db))
        for i, seq_idx in enumerate(tqdm(db_idx)):
            seq = train_db[int(seq_idx)]

            # Collect experience by acting in the environment with current policy
            # Batching decreases variance
            batch_log_probs = []
            batch_weights = []
            batch_rewards = []

            # All simulations in the batch are sampled from the same sequence (correct?)
            sampler = RandomSequenceSampler([seq], self.params.rl_sequence_length, samples_per_epoch=self.params.n_batches_rl)
            for sequence, in DataLoader(sampler, batch_size=1, num_workers=0, collate_fn=identity_func, pin_memory=True):
            # for _ in range(self.params.n_batches_rl):
                # sequence = self.sequence_sampler[i]  # Random sequence of fixed length (10)
                sequence = sampler[0]
                # self.stats['sequences'].append(str(sequence))

                sim_log_probs = []
                sim_overlaps = []
                sim_total_reward = 0  # Not actually the 'reward' but IoU overlap!
                
                # Simulate sequence and gather results
                curr_bbox = sequence.ground_truth_rect[0]
                image = image_loader(sequence.frames[0])
                action_history_oh = torch.zeros(1, self.model.action_history_size).to(self.device)  # Updated in-place
                for frame_num, frame_path in enumerate(sequence.frames[1:], start=1):
                    image = image_loader(frame_path)
                    bboxes, log_probs = self.track_frame(image, curr_bbox, action_history_oh)

                    # Compute "rewards" for each action used in simulation.
                    next_bbox = bboxes[-1]
                    gt = sequence.ground_truth_rect[frame_num]
                    frame_overlaps = overlap_ratio(gt, bboxes[1:])

                    sim_total_reward += frame_overlaps[-1]  # Final state overlap
                    sim_overlaps.append(frame_overlaps)
                    sim_log_probs.extend(log_probs)

                    curr_bbox = next_bbox

                sim_weights = self.calc_weights(sim_overlaps, version=self.stats['reward_version'])
                sim_log_probs = torch.cat(sim_log_probs)

                batch_log_probs.append(sim_log_probs)
                batch_weights.append(sim_weights)

                # NOTE don't keep track of total reward by summing rewards because longer actions would yield more reward...
                sequence_reward = sim_total_reward / (len(sequence.frames) - 1)
                batch_rewards.append(sequence_reward)

            batch_log_probs = torch.cat(batch_log_probs).to(self.device)
            batch_weights = torch.cat(batch_weights).to(self.device)

            # Policy gradient (loss)
            # NOTE: scores (fc7) are ignored during RL training
            # -1 for maximization using minimization algorithm
            self.optimizer.zero_grad()
            loss = -(batch_log_probs * batch_weights).mean()
            loss.backward()
            self.optimizer.step()

            # TODO we could also train fc7 here ...

            batch_reward = sum(batch_rewards) / len(batch_rewards)
            total_epoch_reward += batch_reward
            total_epoch_loss += loss.item()
            self.stats['rewards'].append(batch_reward)
            self.stats['total_epoch_loss'] += loss.item()

        self.stats['avg_epoch_reward'].append(total_epoch_reward / len(train_db))
        wandb.log({"Reward": total_epoch_reward / len(train_db), "Loss": -(total_epoch_loss / len(train_db))})

    def evaluate_performance(self, train_db, test_db):
        pass

    def print_stats(self):
        # super().print_stats()
        print(
            colorama.Fore.GREEN
            + f"\nEpoch {self.stats['epoch']}/{self.n_epochs}, Loss={self.stats['avg_epoch_loss'][-1]:.4f}, Reward={self.stats['avg_epoch_reward'][-1]:.4f}",
            colorama.Fore.RESET
        )
