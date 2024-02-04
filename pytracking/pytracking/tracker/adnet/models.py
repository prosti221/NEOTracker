import pathlib
from collections import OrderedDict
##############################################
import sys
sys.path.append("/Users/admin/Desktop/masters_thesis/code/prototype/adnet-rl-vot/pytracking")
sys.path.append("/Users/admin/Desktop/masters_thesis/code/prototype/adnet-rl-vot/pytracking/pytracking")
sys.path.append("/Users/admin/Desktop/masters_thesis/code/prototype/adnet-rl-vot/pytracking/ltr")
from pathlib import Path
##############################################

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# For Loading efficientnet:
from efficientnet_pytorch import EfficientNet

from pytracking.evaluation.environment import env_settings

from time import time
############################################## ADNET ##############################################
class ADNet(nn.Module):
    # NOTE: must use train() and eval() because of Dropout ...
    
    # First time load the pretrained VGG-M backbone.
    # Otherwise, load adnet_init.pth

    def __init__(self, load_backbone=False, n_actions=11, n_action_history=10):
        super().__init__()

        self.action_history_size = n_actions * n_action_history

        # conv1-3 from VGG-m
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5))),
                ('fc5',   nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)))]))
        self.backbone = self.layers[:3]
        self.fc4_5 = self.layers[3:]

        # Action probability
        self.fc6 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, n_actions)
            # nn.Softmax(dim=1)
        )

        # Binary confidence
        self.fc7 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, 2)
            # nn.Softmax(dim=1)
        )

        branches = nn.ModuleList([self.fc6, self.fc7])

        # Initialize weights
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if load_backbone:
            self.load_backbone()

    def forward(self, feats, actions=None, skip_backbone=False, log_softmax=False):
        if skip_backbone:
            out = feats
        else:
            out = self.backbone(feats)
            out = out.reshape(out.size(0), -1)  # Batch size x Flat features
            #out = out.view(out.size(0), -1)  # Batch size x Flat features
            # Use reshape instead of view
        
        if actions is  None:
            actions = torch.zeros(out.size(0), self.action_history_size, device=out.device)

        out = self.fc4_5(out)
        out = torch.cat((out, actions), dim=1)  # Concatenate actions
        out1 = self.fc6(out)
        out2 = self.fc7(out)

        if log_softmax:
            out1 = F.log_softmax(out1, dim=1)
            out2 = F.log_softmax(out2, dim=1)
        else:
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
        return out1, out2

    def extract_features(self, imgs):
        out = self.backbone(imgs)
        #out = out.view(out.size(0), -1)  # Batch size x Flat features
        out = out.reshape(out.size(0), -1)  # Batch size x Flat features
        return out

    def load_backbone(self, path=None):
        if path is None:
            env = env_settings()
            path = pathlib.Path(env.network_path) / "imagenet-vgg-m.mat"
            # path = pathlib.Path(env.network_path) / "imagenet-vgg-m-conv1-3.mat"

        print(f"Loading {path.name} ...")

        mat = scipy.io.loadmat(path)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.backbone[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.backbone[i][0].bias.data = torch.from_numpy(bias[:, 0])

    def load_network(self, path, freeze_backbone=True):
        if path.suffix == '.mat':
            print(path)
            self.load_backbone(path)
        else:
            print(f"Loading {path.name} ...")
            state = torch.load(path, map_location=torch.device("cpu"))
            if 'model' in state:  # If loading a training checkpoint
                state = state['model']
            if state["fc6.0.weight"].shape[0] != 11:
                self.fc6 = nn.Sequential(
                    nn.Linear(512 + self.action_history_size, state["fc6.0.weight"].shape[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(state["fc6.2.weight"].shape[1], state["fc6.2.weight"].shape[0]),
                    # nn.Softmax(dim=1)
                )
            self.load_state_dict(state)

        self.backbone.requires_grad_(not freeze_backbone)  # Freeze backbone

############################################## ADNET ##############################################


############################################## ADNET ENSEMBLE ##############################################

class ADNetEnsemble(nn.Module):
    """
    This modified version of ADNet allows for the use of several classification heads. (i.e. several fc6 layers)
    """
    def __init__(self, load_backbone=False, n_actions=11, n_action_history=10, n_heads=1):
        super().__init__()

        self.action_history_size = n_actions * n_action_history
        self.n_actions = n_actions
        self.n_heads = n_heads

        # conv1-3 from VGG-m
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5))),
                ('fc5',   nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)))]))
        self.backbone = self.layers[:3]
        self.fc4_5 = self.layers[3:]

        # Action probability
        self.fc6 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, n_actions),
            nn.Softmax(dim=1)
        )

        self.classifiers = nn.ModuleList([])
        self.classifier_rankings = [0 for i in range(n_heads)]

        # Binary confidence
        self.fc7 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, 2)
            # nn.Softmax(dim=1)
        )

        branches = nn.ModuleList([self.fc6, self.fc7])

        # Initialize weights
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if load_backbone:
            self.load_backbone()

    def forward(self, feats, actions=None, skip_backbone=False, log_softmax=False, majority_vote_type="soft"):
        if skip_backbone:
            out = feats
        else:
            out = self.backbone(feats)
            out = out.reshape(out.size(0), -1)  # Batch size x Flat features
            #out = out.view(out.size(0), -1)  # Batch size x Flat features
            # Use reshape instead of view
        
        if actions is  None:
            actions = torch.zeros(out.size(0), self.action_history_size, device=out.device)

        out = self.fc4_5(out)
        out = torch.cat((out, actions), dim=1)  # Concatenate actions

        # Run through all classifiers
        output_list = []
        for i, classifier in enumerate(self.classifiers):
            cl_out = classifier(out)
            cl_out *= self.classifier_rankings[i]
            output_list.append(cl_out)

        out1 = torch.stack(output_list, dim=1)
        # Soft voting (taking the mean of all softmax outputs)
        if majority_vote_type == "soft":
            out1 = out1.mean(dim=1)
        else: 
            """
            sf_vecs = []
            for i, sf in enumerate(output_list):
                sf_vecs.append(F.one_hot(torch.argmax(sf, dim=1), num_classes=self.n_actions).float() * self.classifier_rankings[i])
            
            out1 = torch.argmax(torch.sum(torch.stack(sf_vecs, dim=1), dim=1), dim=1)
            out1 = F.one_hot(out1, num_classes=self.n_actions).float()
            """
            out1 = torch.argmax(out1, dim=2)
            out1 = out1.mode(dim=1).values
            out1 = F.one_hot(out1, num_classes=self.n_actions).float()

        out2 = self.fc7(out)

        return out1, out2

    def extract_features(self, imgs):
        out = self.backbone(imgs)
        #out = out.view(out.size(0), -1)  # Batch size x Flat features
        out = out.reshape(out.size(0), -1)  # Batch size x Flat features
        return out

    def load_backbone(self, path=None):
        if path is None:
            env = env_settings()
            path = pathlib.Path(env.network_path) / "imagenet-vgg-m.mat"
            # path = pathlib.Path(env.network_path) / "imagenet-vgg-m-conv1-3.mat"

        print(f"Loading {path.name} ...")

        mat = scipy.io.loadmat(path)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.backbone[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.backbone[i][0].bias.data = torch.from_numpy(bias[:, 0])

    def load_network(self, path, freeze_backbone=True, fc6_checkpoints_path=None):
        if path.suffix == '.mat':
            self.load_backbone(path)
        else:
            print(f"Loading {path.name} ...")
            state = torch.load(path, map_location=torch.device("cpu"))
            if 'model' in state:  # If loading a training checkpoint
                state = state['model']
            if state["fc6.0.weight"].shape[0] != self.n_actions:
                self.fc6 = nn.Sequential(
                    nn.Linear(512 + self.action_history_size, state["fc6.0.weight"].shape[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(state["fc6.2.weight"].shape[1], state["fc6.2.weight"].shape[0]),
                    # nn.Softmax(dim=1)
                )
            self.load_state_dict(state, strict=False)

        self.backbone.requires_grad_(not freeze_backbone)  # Freeze backbone

        reward_rescaler = lambda x: ((x + 1) / 2) 
        if fc6_checkpoints_path is not None:
            for path in self.get_paths(fc6_checkpoints_path):
                fc6, ranking, fitness = self.get_fc6(path)
                self.classifier_rankings[ranking] = reward_rescaler(fitness)
                self.classifiers.append(fc6)

    
    def get_fc6(self, path):
        state = torch.load(path, map_location=torch.device("cpu"))
        if "ranking" in state:
            ranking = state["ranking"] 
            fitness = state["fitness"]
            state = state["state_dict"]
        if len(state.keys()) > 2:
            new_state = state.copy()
            for key in state.keys():
                if "fc6" not in key:
                    del new_state[key]
                else:
                    new_state[key.replace("fc6.", "")] = state[key]
                    del new_state[key]
            state = new_state
        if "_submodules.0.weight" in state:
            state["0.weight"] = state["_submodules.0.weight"]
            del state["_submodules.0.weight"]
            state["0.bias"] = state["_submodules.0.bias"]
            del state["_submodules.0.bias"]
        fc6 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, 11),
            nn.Softmax(dim=1)
        )
        fc6.load_state_dict(state)

        return fc6, ranking, fitness

    def get_paths(self, checkpoints_path):
        paths = []
        for path in checkpoints_path.iterdir():
            if path.suffix == ".pth":
                paths.append(path)
        
        return sorted(paths, reverse=False)[:self.n_heads]

############################################## ADNET ENSEMBLE ##############################################


############################################## ADNET Enhanced ##############################################

class ADNetEnhanced(nn.Module):
    """
    A version of ADNet that uses a more modern feature extractor (EfficientNet) instead of VGG-M.
    """
    def __init__(self, load_backbone=False, n_actions=11, n_action_history=10):
        super().__init__()
        self.action_history_size = n_actions * n_action_history
        print("Loading EfficientNet ...")

        # Loading efficientnet
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        self.fc4_5 = nn.Sequential(OrderedDict([
                ('fc4',   nn.Sequential(nn.Linear(1280, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5))),
                ('fc5',   nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)))]))


        # Action probability
        self.fc6 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, n_actions)
            # nn.Softmax(dim=1)
        )

        # Binary confidence
        self.fc7 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, 2)
            # nn.Softmax(dim=1)
        )

        branches = nn.ModuleList([self.fc6, self.fc7])

        # Initialize weights
        for m in self.fc4_5.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats, actions=None, skip_backbone=False, log_softmax=False):
        if skip_backbone:
            out = feats
        else:
            out = self.backbone.extract_features(feats)
            out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)  # Batch size x Flat features

        if actions is  None:
            actions = torch.zeros(out.size(0), self.action_history_size, device=out.device)

        out = self.fc4_5(out)
        out = torch.cat((out, actions), dim=1)  # Concatenate actions
        out1 = self.fc6(out)
        out2 = self.fc7(out)

        if log_softmax:
            out1 = F.log_softmax(out1, dim=1)
            out2 = F.log_softmax(out2, dim=1)
        else:
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
        return out1, out2

    def extract_features(self, imgs):
        out = self.backbone.extract_features(imgs)
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)

        return out

    def load_backbone(self, path=None):
        pass

    def load_network(self, path, freeze_backbone=True):
        print(f"Loading {path.name} ...")
        state = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in state:  # If loading a training checkpoint
            state = state['model']
        if state["fc6.0.weight"].shape[0] != 11:
            self.fc6 = nn.Sequential(
                nn.Linear(512 + self.action_history_size, state["fc6.0.weight"].shape[0]),
                nn.ReLU(inplace=True),
                nn.Linear(state["fc6.2.weight"].shape[1], state["fc6.2.weight"].shape[0]),
                # nn.Softmax(dim=1)
            )
        self.load_state_dict(state)

        self.backbone.requires_grad_(not freeze_backbone)  # Freeze backbone
    

############################################## ADNET ENHANCED ##############################################

if __name__ == "__main__":
    model_path = Path("/Users/admin/Desktop/masters_thesis/code/prototype/adnet-rl-vot/networks/adnet_checkpoints/RL-XL/checkpoint_100.pth")
    checkpoints_path = Path("/Users/admin/Desktop/masters_thesis/code/prototype/adnet-rl-vot/networks/adnet_checkpoints/ensemble")
    device = torch.device('cpu')
    #adnet = ADNetEnsemble(n_actions=11, n_action_history=10, n_heads=3)

    adnet_enanced = ADNetEnhanced(n_actions=11, n_action_history=10)
    adnet = ADNet(n_actions=11, n_action_history=10)
    dummy_input = torch.randn(1, 3, 107, 107, device=device)
    adnet_time = 0
    adnet_enhanced_time = 0

    for i in range(100):
        st_time = time()
        adnet(dummy_input)
        adnet_time += time() - st_time

        st_time = time()
        adnet_enanced(dummy_input)
        adnet_enhanced_time += time() - st_time
     
    print(f"ADNet time: {adnet_time/100}")
    print(f"ADNet Enhanced time: {adnet_enhanced_time/100}")

    """
    adnet.load_network(model_path, freeze_backbone=True, fc6_checkpoints_path=checkpoints_path)  # Also freezes backbone
    adnet.to(device)

    # Set seed
    torch.manual_seed(412)
    dummy_input = torch.randn(1, 3, 107, 107, device=device)

    actions, conf = adnet(dummy_input)
    print(actions.shape)
    print(actions.argmax(dim=1).item())
    """

