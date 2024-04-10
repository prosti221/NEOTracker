from .train import TrackerTrainer, SequenceDataset, classification_accuracy
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import colorama
from tqdm import tqdm
import os
import time
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 4, 5"

class TrainTracker_SL(TrackerTrainer):
    def __init__(self, params, model_path=None, **kwargs):
        super().__init__(params, model_path, **kwargs)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.model.backbone.parameters(), 'lr': params.lr_backbone},
        #     {'params': self.model.fc4_5.parameters()},
        #     {'params': self.model.fc6.parameters()},
        #     {'params': self.model.fc7.parameters()}],
        #     lr=params.lr_train)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.n_epochs = kwargs.get('epochs', params.n_epochs_sl)
        self.epoch_checkpoint = kwargs.get('epoch_checkpoint', params.checkpoint_interval_sl)

        self.stats['train_acc'] = 0
        self.stats['test_acc'] = 0
        self.stats['acc_list'] = []

    def train_epoch(self, train_db, test_db):
        db_idx = np.random.permutation(len(train_db))  # Shuffle sequences
        num_actions = 0
        num_correct_actions = 0
        total_epoch_loss = 0 
        for i, seq_idx in enumerate(tqdm(db_idx)):
            seq = train_db[int(seq_idx)]
            #for pos_data, neg_data in SequenceDataset(seq, self.params):
            for pos_data, neg_data in DataLoader(SequenceDataset(seq, self.params), shuffle=True, batch_size=self.params.batch_frames, pin_memory=True, collate_fn=SequenceDataset._collate, num_workers=2):
                pos_patches, pos_actions, pos_scores = pos_data
                neg_patches, neg_scores = neg_data
                
                pos_patches = pos_patches.to(self.device) 
                pos_actions = pos_actions.to(self.device)
                pos_scores = pos_scores.to(self.device)

                neg_patches = neg_patches.to(self.device)
                neg_scores = neg_scores.to(self.device)

                # Action history is set to zero in SL training!
                action_history_oh_zero = torch.tensor(0.0).expand(pos_patches.size(0), self.model.action_history_size).to(self.device)

                # TODO why not combine positive and negatives?
                
                # Optimize for positive samples
                self.optimizer.zero_grad()
                try:
                    out_actions, out_scores = self.model(pos_patches, action_history_oh_zero, log_softmax=True)
                except RuntimeError:
                    print("RuntimeError")
                    continue
                action_loss = self.criterion(out_actions, pos_actions)
                score_loss = self.criterion(out_scores, pos_scores)
                loss1 = action_loss + score_loss  # Loss sum from paper
                loss1.backward()
                self.optimizer.step()

                # Log acc to wandb 
                #num_correct_actions += (out_actions.argmax(dim=1) == pos_actions).sum().item()
                total_epoch_loss += loss1.item()
                num_actions += len(pos_actions)

                action_history_oh_zero = torch.tensor(0.0).expand(neg_patches.size(0), self.model.action_history_size).to(self.device)

                # Optimize for negative samples
                # In this case we don't have any action labels, so don't optimize for action_loss.
                try:
                    self.optimizer.zero_grad()
                    _, out_scores = self.model(neg_patches, action_history_oh_zero, log_softmax=True)
                    loss2 = self.criterion(out_scores, neg_scores)
                    loss2.backward()
                    self.optimizer.step()
                except RuntimeError:
                    print("RuntimeError")
                    wandb.log({"loss" : loss1.item()})
                    self.stats['total_epoch_loss'] += loss1.item()
                    continue

                self.stats['total_epoch_loss'] += loss1.item() + loss2.item()
        
        epoch_loss = total_epoch_loss / num_actions
        #accuracy = num_correct_actions / num_actions
        #wandb.log({"loss" : epoch_loss, "accuracy": accuracy})
        wandb.log({"loss" : epoch_loss})

    def evaluate_performance(self, train_db, test_db=None):
        train_db = train_db[:50]
        st_time = time.time()
        train_acc = classification_accuracy(self.model, train_db, self.device, self.params)
        end_time = time.time()
        print(f"Time to evaluate train accuracy: {end_time - st_time}")
        self.stats['train_acc'] = train_acc
        self.stats['acc_list'].append(train_acc)
        if test_db is not None:
            st_time = time.time()
            test_acc = classification_accuracy(self.model, test_db, self.device, self.params)
            end_time = time.time()
            print(f"Time to evaluate eval accuracy: {end_time - st_time}")
            self.stats['test_acc'] = test_acc

        wandb.log({"train_accuracy" : train_acc[0], "validation_accuracy" : test_acc[0]})

    def print_stats(self):
        # super().print_stats()
        print(
            colorama.Fore.GREEN
            + f"\nEpoch {self.stats['epoch']}/{self.n_epochs}, Loss={self.stats['avg_epoch_loss'][-1]:.4f}, Train-Acc={self.stats['train_acc']}, Valid-Acc={self.stats['test_acc']}",
            colorama.Fore.RESET
        )
