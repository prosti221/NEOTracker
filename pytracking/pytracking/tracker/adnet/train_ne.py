from .train import TrackerTrainer, clamp_bbox, perform_action, one_hot, identity_func, image_loader
from .utils import overlap_ratio, extract_region
#from .ne_utils import *
from ltr.data.sampler import RandomSequenceSampler
import numpy as np
from .models import ADNet
# Evotorch imports 
from evotorch import Problem
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import SNES, PGPE, GeneticAlgorithm, Cosyne
from evotorch.logging import StdOutLogger, PandasLogger
from evotorch.operators import GaussianMutation, OnePointCrossOver
# PyTorch imports
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import ray
import math
from PIL import Image

import os
import rerun as rr
from evaluation import Tracker
import torch.utils.tensorboard as tb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Tensorboard for plots
writer = tb.SummaryWriter("./runs/")

# Rerun for tracking evaluation runs
rr.init("Training NE")
rr.save("rerun.rrd")
#rr.spawn()

class ModelWrapper(nn.Module):
    def __init__(self, backbone, fc4_5, action_history_size=10, maxpool_size=1):
        super(ModelWrapper, self).__init__()
        self.backbone = backbone 
        self.fc4_5 = fc4_5 # The output shape of this is 512
        # Add a maxpool layer to fc4_5 to reduce the output size to 256
        self.maxpool_size = maxpool_size
        if maxpool_size > 1:
            self.maxpool = nn.MaxPool1d(maxpool_size)
        self.action_history_size = action_history_size
    
    
    def forward(self, x, actions=None, skip_backbone=False, log_softmax=False):
        if actions is None:
            actions = torch.zeros(x.size(0), self.action_history_size).to(x.device)
        out = self.backbone(x)
        out = out.reshape(out.size(0), -1)  # Batch size x Flat features
        #out = out.view(out.size(0), -1)  # Batch size x Flat features
        # Use reshape instead of view
        out = self.fc4_5(out)
        if self.maxpool_size > 1:
            out = self.maxpool(out)
        out = torch.cat((out, actions), dim=1)  # Concatenate actions

        return out

@ray.remote
class DatasetIncrementer:
    def __init__(self, dataset_size, batch_size, repeat_for=0):
        self.step = 0
        self.repeat_for = repeat_for
        self.repeate_counter = 0
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.idx = [i for i in range(self.batch_size)]

    def increment(self):
        if self.repeat_for > 0 and self.repeate_counter < self.repeat_for:
                self.repeate_counter += 1
                return
        else:
            self.repeate_counter = 0
            self.step += 1
            if (self.batch_size * self.step) + self.batch_size > self.dataset_size:
                self.step = 0
                self.idx = [i for i in range(self.batch_size)]
            else:
                self.idx = [i for i in range(self.batch_size * self.step, (self.batch_size * self.step) + self.batch_size)]
    
    def get_idx(self):
        return self.idx

    def get_step(self):
        return self.step

class TrainTracker_NE(TrackerTrainer):

    def __init__(self, params, reward_version="RETURNS", algorithm="NEAT", population_size=30, n_generations=100, maxpool_size=1, checkpoint_paths=[], resume_from_checkpoint=False, batch_size=1, **kwargs):
        super().__init__(params, **kwargs)
        # We are now only interested in the backbone of the model as well as the fc4_5 layer
        # x -> out = model.backbone(x) -> out = out.reshape(out.size(0), -1) -> out = model.fc4_5(out)
        self.stats['reward_version'] = reward_version
        self.stats['rewards'] = []
        self.stats['actions'] = []
        self.stats['total_epoch_loss'] = 0
        
        self.stats['max_epoch_reward'] = []
        self.stats['avg_epoch_reward'] = []


        self.population_size = population_size
        self.n_generations = n_generations
        self.action_history_size = self.model.action_history_size
        self.maxpool_size = maxpool_size

        self.resume_from_checkpoint = resume_from_checkpoint

        self.algorithm = algorithm

        self.fc6 = self.model.fc6 # Includes the ReLU layer
        # Print the shape of the weights of the fc6 layer
        self.softmax = nn.Softmax(dim=1)
        self.original_model = self.model
        self.model = ModelWrapper(self.model.backbone, self.model.fc4_5, action_history_size=self.action_history_size, maxpool_size=self.maxpool_size).to(self.device)

        # These parameters do not need to be trained
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        self.fc6.eval()
        self.model.eval()

        # Convert model to default dtype
        self.model = self.model.half()

        # Load checkpoints from the list of paths if we want to do transfer learning.
        self.checkpoints = [] 
        if len(checkpoint_paths) > 0:
            for path in checkpoint_paths:
                chk = ADNet(n_actions=params.num_actions, n_action_history=params.num_action_history)
                chk.load_network(path, freeze_backbone=True)
                chk.eval()
                chk.to(self.device)
                self.checkpoints.append(chk)

        
        self.train_db = None
        self.eval_db = None
        self.population = None

        self.batch_size = batch_size
        self.dataset_incrementer = None

        # These are stats that are computed each time the entire dataset has been gone through
        self.avgs = []; self.maxes = []
        self.end_of_dataset_stats = {"avg" : [], "max" : []}

        
    def extract_region(self, image, bbox):
        return extract_region(image, bbox, crop_size=self.params.img_size, 
                                           padding=self.params.padding, 
                                           means=self.params.means)


    @torch.no_grad()
    def track_frame(self, genome, image, bbox, action_history_oh):
        # Perform actions until convergence in current frame (for RL).
        params = self.params 
        opts = params.opts

        img_size = np.array(image.shape[1::-1])

        bboxes = [bbox]
        
        # For oscillation checking
        round_bboxes = set()
        round_bboxes.add(tuple(bbox.round()))

        move_counter = 0
        prev_action = -1
        curr_bbox = bbox
        while move_counter < opts['num_action_step_max'] and prev_action != opts['stop_action']:
            curr_patch = self.extract_region(image, curr_bbox).to(self.device).half()

            feats = self.model(curr_patch, action_history_oh).float()
            actions = genome(feats)

            action = torch.argmax(actions)
            action_idx = action.item()
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
            action_history_oh[0, n:] = action_history_oh[0, :-n].clone()
            action_history_oh[0, :n] = one_hot(torch.tensor(action_idx), num_classes=n)

            curr_bbox = next_bbox
            bboxes.append(curr_bbox)
            round_bboxes.add(next_bbox_round)
            prev_action = action_idx
            move_counter += 1

        return np.stack(bboxes)

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

    @torch.no_grad()
    def fitness_function(self, genome):
        train_db = self.train_db
        total_epoch_weights = 0
        total_mean_num_actions = 0

        db_idx = ray.get(self.dataset_incrementer.get_idx.remote())
        for i, seq_idx in enumerate(db_idx):
            seq = train_db[int(seq_idx)]
            # Collect experience by acting in the environment with current policy
            # Batching decreases variance
            batch_weights = []
            mean_num_actions = 0
            mean_num_actions_list = []

            # All simulations in the batch are sampled from the same sequence (correct?)
            sampler = RandomSequenceSampler([seq], self.params.rl_sequence_length, samples_per_epoch=self.params.n_batches_rl)
            for sequence, in DataLoader(sampler, batch_size=1, num_workers=0, collate_fn=identity_func, pin_memory=True):
                sim_overlaps = []
                sim_total_reward = 0  # Not actually the 'reward' but IoU overlap!
                
                # Simulate sequence and gather results
                curr_bbox = sequence.ground_truth_rect[0]
                image = image_loader(sequence.frames[0])
                action_history_oh = torch.zeros(1, self.action_history_size).to(self.device).half()  # Updated in-place
                for frame_num, frame_path in enumerate(sequence.frames[1:], start=1):
                    image = image_loader(frame_path)
                    bboxes = self.track_frame(genome, image, curr_bbox, action_history_oh)

                    # Compute "rewards" for each action used in simulation.
                    next_bbox = bboxes[-1]
                    gt = sequence.ground_truth_rect[frame_num]
                    frame_overlaps = overlap_ratio(gt, bboxes[1:])

                    sim_total_reward += frame_overlaps[-1]  # Final state overlap
                    sim_overlaps.append(frame_overlaps)

                    curr_bbox = next_bbox

                sim_weights = self.calc_weights(sim_overlaps, version=self.stats['reward_version'])
                batch_weights.append(sim_weights)

            batch_weights = torch.cat(batch_weights).to(self.device)
            total_epoch_weights += torch.mean(batch_weights)
        
        total_epoch_weights /= len(db_idx)

        return total_epoch_weights
    

    def train(self, train_db, eval_db):
        self.train_db = train_db
        self.eval_db = eval_db

        # These are the samples I use for evaluating all the genomes after an entire run of the dataset
        # This is so that we take the genome that generalizes the best as the model that we save for the current checkpoint
        # Necessary when we use batch_size = 1 because of the high instability.
        #self.eval_db = [eval_db[1], train_db[0], train_db[21], train_db[38], train_db[42], train_db[33]] + train_db[-9:]
        self.eval_db = [eval_db[1]] + train_db[:8]
        #self.eval_db = [eval_db[1]]+[train_db[0]]
        #self.eval_db = [train_db[0]]

        # Keep track of the average fitness of the population for each generation per sample
        # sample_progress[(sample_idx, sample_name)] = [avg_fitness_gen0, avg_fitness_gen1, ...]
        sample_progress = {}


        # Declare the problem

        ### Single layer network
        #policy = (
        #    """
        #    Linear(622, 11)
        #    >> Softmax(dim=1)
        #    """
        #)
        #
        ## Two layer network
        policy = (
            """
            Linear(622, 512)
            >> ReLU()
            >> Linear(512, 11)
            >> Softmax(dim=1)
            """
        )
        problem = NEProblem(
            #objective_sense=["max", "max"],
            objective_sense="max",
            network=policy,
            network_eval_func=self.fitness_function,
            num_actors=6,
            num_gpus_per_actor=1/6,
            #device=self.device,
        )

        operators=[
            #OnePointCrossOver(problem, tournament_size=16, cross_over_rate=0.6), #cross_over_rate=0.6
            OnePointCrossOver(problem, tournament_size=4, cross_over_rate=0.6),
            GaussianMutation(problem, stdev=0.1, mutation_probability=0.4)  #0.03 stdev, 0.4 prob
        ]

        """
        searcher = GeneticAlgorithm(
            problem,
            popsize=self.population_size,
            operators=operators,
            #elitist=False
        )

        """
        searcher = Cosyne(
            problem,
            popsize=self.population_size,
            tournament_size=4,
            mutation_stdev=0.2,
            mutation_probability=0.4,
            elitism_ratio=0.75, 
            #permute_all=True
            #eta=1.8 # 0.8
        )

        self.dataset_incrementer = DatasetIncrementer.remote(len(self.train_db), self.batch_size, repeat_for=1)

        # Transfer FC6 weights from pre-trained model
        #searcher = self.initialize_population(searcher, problem, self.fc6, stdev=0.001)
        #self.visualize_population(searcher)

        # Initialize a standard output logger, and a pandas logger
        _ = StdOutLogger(searcher, interval=1)
        pandas_logger = PandasLogger(searcher)


        reward_rescaler = lambda x: ((x + 1) / 2) * 100
        #reward_rescaler = lambda x: x

        # Run evolution for the specified amount of generations
        c = 0
        previous_best_eval = 0 
        previous_best = None
        for i in range(self.n_generations):
            # Perform an evolution step
            """
            if i == 0:
                # Save the best population member
                best_model = self.get_best_genome(searcher._population, self.eval_db, problem, frames_per_sequennce=30)
                self.save_model(best_model, i)

                self.evaluate(self.train_db[0], i, mode="train")
                if self.eval_db is not None:
                    self.evaluate(self.eval_db[0], i, mode="eval")
            """

            searcher.step()
            # Code for "profiling" and generating a report
            #sample_progress = self.sample_profiling(searcher, sample_progress, self.checkpoint_path / "pytorch_checkpoints" / "sample_progress_new.txt")

            # Handle the next batch of the dataset for the next generation
            self.dataset_incrementer.increment.remote()
                
            # Get the average and max fitness of the population            
            self.avgs.append(reward_rescaler(searcher.status["mean_eval"]))
            self.maxes.append(reward_rescaler(searcher.status["pop_best_eval"]))

            # Log the average and max fitness of the population on rerun
            self.log_time_series("Population avg", reward_rescaler(searcher.status["mean_eval"]), i)
            self.log_time_series("Population max", reward_rescaler(searcher.status["pop_best_eval"]), i)
            self.visualize_population(searcher)

            # Log the average fitness of the population on tensorboard
            writer.add_scalars("Population statistics",
                {"avg" : reward_rescaler(searcher.status["mean_eval"]), 
                 "max" : reward_rescaler(searcher.status["pop_best_eval"])
                }, i)

            if c * self.batch_size >= (len(self.train_db)) * 2: 
                # Save the best population member
                rankings = self.get_best_genome(searcher._population, self.eval_db, problem, frames_per_sequennce=80, return_rankings=True)
                best_model = rankings[0][0]
                self.save_model(best_model, i)

                # Save the best n FC6 layers of the population as pytorch models
                self.save_top_n_population(rankings, top_n=200, step=i)

                self.end_of_dataset_stats["avg"].append(np.mean(self.avgs))
                self.end_of_dataset_stats["max"].append(np.mean(self.maxes))
                self.avgs = []; self.maxes = []

                # Log the dataset iteration average and max fitness of the population on rerun
                self.log_time_series("Dataset iteration avg", self.end_of_dataset_stats["avg"][-1], i)
                self.log_time_series("Dataset iteration max_avg", self.end_of_dataset_stats["max"][-1], i)

                # Log the dataset iteration average and max fitness of the population on tensorboard
                writer.add_scalars("Dataset iteration statistics", 
                    {"avg" : self.end_of_dataset_stats["avg"][-1], 
                     "max" : self.end_of_dataset_stats["max"][-1]
                    }, i)

                print("End of dataset stats:")
                print(f"Average: {self.end_of_dataset_stats['avg'][-1]}")
                print(f"Max: {self.end_of_dataset_stats['max'][-1]}\n")

                self.evaluate(self.train_db[0], i, mode="train")
                if self.eval_db is not None:
                    self.evaluate(self.eval_db[0], i, mode="eval")
                c = 0
            c += 1

        writer.close()


    def save_model(self, fc6, num):
        path = self.checkpoint_path / "pytorch_checkpoints" / f"best_genome_{num}.pth"
        model = self.original_model
        #Assign the fc6 layer to the model wihtout the softmax layer
        model.fc6 = fc6[:-1]
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(model.state_dict(), path)

    # Used to initialize population based on pre-trained fc6 layer
    def initialize_population(self, searcher : GeneticAlgorithm, problem : NEProblem, fc6 : nn.ModuleList, stdev=0.1):
        weights = fc6[0].weight
        bias = fc6[0].bias
        weights = weights.view(-1)
        weights = torch.cat((weights, bias)).detach()
        # turn off the gradients for the weights
        weights.requires_grad = False

        searcher._population = problem.generate_batch(popsize=searcher._popsize, center=weights, stdev=stdev)
        # Assign the first genome weights to the pre-trained fc6 layer
        values = searcher._population.access_values()
        values[0] = weights
        searcher._population.set_values(values)
        """
        # Sanity check

        problem.evaluate(searcher._population)
        print(searcher._population.access_evals())
        print(f"Mean eval: {searcher._population.access_evals().mean()}")
        exit()
        """

        return searcher

    def visualize_population(self, searcher : GeneticAlgorithm):
        reward_rescaler = lambda x: ((x + 1) / 2) * 100
        step = searcher._steps_count
        pop = [pop.values.clone() for pop in searcher._population]
        #evals = [pop.access_values(keep_evals=True)[1].item() for pop in searcher._population]
        evals = [reward_rescaler(pop.evals.numpy()[0]) if not math.isnan(pop.evals.numpy()[0]) else 1 for pop in searcher._population]

        pop = torch.stack(pop)

        U, S, V = torch.pca_lowrank(pop, q=3)
        points3D = torch.matmul(pop, V)

        # Project evals to a color scale
        evals = (torch.tensor(evals) / 100) * 255 if step > 0 else torch.tensor(evals) * 0 
        # Make it a blue gradient
        colors = torch.stack((torch.zeros_like(evals), torch.zeros_like(evals), evals), dim=1)

        rr.log("Population", rr.Points3D(points3D, colors=colors, radii=0.3))
    
    def log_time_series(self, name, data, step=0):
        rr.log(name, rr.TimeSeriesScalar(data, label=name))

    def evaluate(self, sequence, step, mode="train"):
        target =  self.checkpoint_path / "pytorch_checkpoints" / f"best_genome_{step}.pth"
        step = 0
        namespace = f"{mode}_runs/{mode}_images_{step}"

        N_RUNS = 1
        force = True
        trackers = []
        debug = 0

        tracker = Tracker('adnet', 'only_track', run_id=step, display_name=f"Evaluation {step}", params_callback=lambda p: setattr(p, 'model_path', target), experiment_name=f"NE-evotorch_{step}")

        out = tracker.run_sequence(sequence, debug=debug, visdom_info=None)
        bboxes = out["target_bbox"] # X_max, Y_max, X_min, Y_min
        res = zip(sequence.frames, bboxes)

        for i, (img, bbox) in enumerate(res):
            # Load image
            img = Image.open(img).convert("RGB")
            rr.log(namespace + "/rgb", rr.Image(img))
            rr.log(namespace + "/tracked", rr.Boxes2D(
                array=bbox,
                array_format=rr.Box2DFormat.XYWH,
                class_ids=[0])
            )

    def get_best_genome(self, population, dataset, problem, frames_per_sequennce=20, return_rankings=False):
        """
        This will run through all the genomes in the population and evaluate them on a dataset.
        The best genome will be returned.
        """
        db_idx = np.arange(len(dataset))
        genome_rankings = []
        for i, genome in enumerate(population):
            genome = problem.make_net(genome).to(self.device)
            total_epoch_weights = 0
            for j, seq_idx in enumerate(db_idx):
                seq = dataset[int(seq_idx)]
                # All simulations in the batch are sampled from the same sequence (correct?)
                batch_weights = []                                                     
                sampler = RandomSequenceSampler([seq], frames_per_sequennce, 1)
                for sequence, in DataLoader(sampler, batch_size=1, num_workers=0, collate_fn=identity_func, pin_memory=True):
                    # Not actually the 'reward' but IoU overlap!
                    sim_overlaps = []                                                  
                    sim_total_reward = 0  # Not actually the 'reward' but IoU overlap!
                                                                                    
                    # Simulate sequence and gather results                             
                    curr_bbox = sequence.ground_truth_rect[0]                          
                    image = image_loader(sequence.frames[0])                           
                    action_history_oh = torch.zeros(1, self.action_history_size).to(self.device)
                    for frame_num, frame_path in enumerate(sequence.frames[1:], start=1):
                        image = image_loader(frame_path)                               
                        bboxes = self.track_frame(genome, image, curr_bbox, action_history_oh)
                                                                                    
                        # Compute "rewards" for each action used in simulation.        
                        next_bbox = bboxes[-1]                                         
                        gt = sequence.ground_truth_rect[frame_num]                     
                        frame_overlaps = overlap_ratio(gt, bboxes[1:])                 
                                                                                    
                        sim_total_reward += frame_overlaps[-1]  # Final state overlap
                        sim_overlaps.append(frame_overlaps)                            
                                                                                    
                        curr_bbox = next_bbox                                          
                                                                                    
                    sim_weights = self.calc_weights(sim_overlaps, version=self.stats['reward_version'])
                    batch_weights.append(sim_weights)                                  
                                                                                    
                batch_weights = torch.cat(batch_weights).to(self.device)               
                #print(f"Performance for genome {i} on sample {j}: {torch.mean(batch_weights).item()}")
                total_epoch_weights += torch.mean(batch_weights)                       
                                                                                    
            total_epoch_weights = (total_epoch_weights / len(db_idx)).item()
            #print(f"Evaluated genome {i} with fitness: {total_epoch_weights}")
            genome_rankings.append((genome, total_epoch_weights))
        
        genome_rankings = sorted(genome_rankings, key=lambda x: x[1], reverse=True)
        #print([rankings[1] for rankings in genome_rankings[:10]])
        
        if return_rankings:
            return genome_rankings

        return genome_rankings[0][0]


    def save_top_n_population(self, rankings, top_n, step):

        for i, genome in enumerate(rankings[:top_n]):
            genome, fitness = genome
            population_repository = self.checkpoint_path / "pytorch_checkpoints" / f"saved_populations" / f"population_{step}"
            if not population_repository.exists():
                population_repository.mkdir(parents=True)

            path = population_repository / f"genome_{i}.pth"

            # Save the genome ranking in state_dict
            torch.save({"state_dict" : genome.state_dict(), "ranking" : i, "fitness" : fitness}, path)

    def sample_profiling(self, searcher, sample_progress, path):
            # Code for "profiling" and generating a report
            current_sample_idx = ray.get(self.dataset_incrementer.get_idx.remote())[0]
            key = (current_sample_idx, self.train_db[current_sample_idx].name)
            sample_progress[key] = ([(searcher.status["mean_eval"])], 0) if key not in sample_progress else (sample_progress[key][0] + [searcher.status["mean_eval"]], 0)
            # Compute the average change in fitness for the current sample
            if len(sample_progress[key][0]) > 1:
                change = np.mean(np.diff(sample_progress[key][0]))
                sample_progress[key] = (sample_progress[key][0], change)

            sample_progress = {k: v for k, v in sorted(sample_progress.items(), key=lambda item: item[1], reverse=True)}
            # Save this sample progress to file as json
            with open(path, "w") as f:
                for k, v in sorted(sample_progress.items(), key=lambda item: item[1][1], reverse=True):
                    f.write(f"{k}: {v}\n")
                f.close()
