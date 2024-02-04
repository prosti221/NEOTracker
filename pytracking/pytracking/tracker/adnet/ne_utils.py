import tensorflow as tf
import nevopy as ne 
import torch
import torch.nn as nn
import functools
import numpy as np
import os

RAY_CONFIG = {
    "num_cpus": 6,
    "num_gpus": 1,
}

VALID_ALGORITHMS = ["NEAT", "RANDOM", "FIXED-LARGE", "FIXED", "NEAT-LARGE"]
VALID_PROCESSING_SCHEDULERS = ["SERIAL", "POOL", "RAY"]


SCHEDULER_MAP = {
     "SERIAL"   : ne.processing.SerialProcessingScheduler,
     "POOL"     : ne.processing.PoolProcessingScheduler,
     "RAY"      : ne.processing.RayProcessingScheduler,
    }

NEAT_CONFIG = ne.neat.NeatConfig(
    weak_genomes_removal_pc=0.7,
    weight_mutation_chance=(0.7, 0.9),
    new_node_mutation_chance=(0.1, 0.5),
    new_connection_mutation_chance=(0.08, 0.5),
    enable_connection_mutation_chance=(0.08, 0.5),
    disable_inherited_connection_chance=0.75,
    mating_chance=0.75,
    interspecies_mating_chance=0.05,
    rank_prob_dist_coefficient=1.75,
    # weight mutation specifics
    weight_perturbation_pc=(0.05, 0.1),
    weight_reset_chance=(0.05, 0.4),
    new_weight_interval=(-2, 2),
    # mass extinction
    mass_extinction_threshold=25,
    maex_improvement_threshold_pc=0.03,
    # infanticide
    infanticide_output_nodes=False,
    infanticide_input_nodes=False,
    # speciation
    species_distance_threshold=1.75,
)

FIXED_CONFIG = ne.genetic_algorithm.GeneticAlgorithmConfig(
        # weight mutation
        mutation_chance=(0.6, 0.9),
        weight_mutation_chance=(0.5, 1),
        weight_perturbation_pc=(0.05, 0.5),
        weight_reset_chance=(0.05, 0.5),
        new_weight_interval=(-2, 2),
        # reproduction
        weak_genomes_removal_pc=0.5,
        mating_chance=0.7,
        interspecies_mating_chance=0.05,
        mating_mode="weights_mating",
        rank_prob_dist_coefficient=1.75,
        predatism_chance=0.1,
        # speciation
        species_distance_threshold=1.75,
        species_elitism_threshold=5,
        elitism_pc=0.03,
        species_no_improvement_limit=15,
        # mass extinction
        mass_extinction_threshold=15,
        maex_improvement_threshold_pc=0.03,
)

# Adding a local response layer to the nevopy pipeline when running with FIXED-LARGE
local_response_norm = tf.keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75))
# This is an updated process function for the genomes that evolve the enitre ADNet model
def process(self, x):
    """
    This is an updated process function for the genomes that evolve the enitre ADNet model
        Inputs:
            x: A tuple of (image, action_history_oh)
        Returns:
            out: The output of the model
    """
    inp, action_history_oh = x
    backbone = self.layers[:-1]
    classifier = self.layers[-1]
    for i, layer in enumerate(backbone):
        if i == 0 or i == 2:
            inp = layer(inp)
            inp = local_response_norm(inp)
        else:
            inp = layer(inp)

    inp = tf.concat((inp, action_history_oh), axis=1)
    inp = classifier(inp)

    return inp

"""
 A wrapper class for the part of the ADNet model that we won't be evolving, will act as a backbone.
 Interfaces throgh pytorch
"""
class AugmentedModel(nn.Module):
    def __init__(self, backbone, fc4_5, action_history_size=10, maxpool_size=1):
        super(AugmentedModel, self).__init__()
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


# NevoPy callback class for saving population per generation.
class PopulationCheckpointCallback(ne.callbacks.Callback):
    def __init__(self, path, save_every=1):
        super().__init__()
        self.path = path
        self.save_every = save_every

    def on_generation_end(self, current_generation : int, max_generations, **kwargs):
        if current_generation % self.save_every == 0:
            self.population.save(self.path / f"population_checkpoint_gen_{current_generation}.pkl")

def load_from_checkpoint(path, processing_scheduler, size, algorithm="FIXED"):
    # Iterate through all checkpoints in path and get the latest one
    if processing_scheduler == "RAY":
        scheduler = SCHEDULER_MAP[processing_scheduler](**RAY_CONFIG)
    elif processing_scheduler == "POOL":
        scheduler = SCHEDULER_MAP[processing_scheduler](num_processes=6)
    else:
        scheduler = SCHEDULER_MAP[processing_scheduler]()

    checkpoints = []
    for checkpoint in path.iterdir():
        if checkpoint.is_file() and checkpoint.suffix == ".pkl":
            checkpoints.append(checkpoint)
    
    # Sort by the number at the end of the checkpoint name
    checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("_")[-1].split(".")[0]), reverse=True)
    basename = checkpoints[0].name
    print(f"Loading from checkpoint: {basename}...")
    
    checkpoint = checkpoints[0]
    """
    population = ne.genetic_algorithm.GeneticPopulation(size=size, base_genome=get_base_genome(algorithm), processing_scheduler=scheduler)
    """
    population = ne.genetic_algorithm.GeneticPopulation.load(checkpoint, scheduler)
    print(f"Done!")

    return population

def get_base_genome(algorithm, maxpool_size=1, action_history_size=11*10, num_actions=11):
    if algorithm == "FIXED-LARGE":
        BASE_GENOME_EXTENDED = ne.fixed_topology.FixedTopologyGenome(
          layers=[
              ne.fixed_topology.layers.TFConv2DLayer(96, (7, 7), strides=(2, 2), activation="relu"),
              ne.fixed_topology.layers.TFMaxPool2DLayer((3, 3), strides=(2, 2)),
              ne.fixed_topology.layers.TFConv2DLayer(256, (5, 5), strides=(2, 2), activation="relu"),
              ne.fixed_topology.layers.TFMaxPool2DLayer((3, 3), strides=(2, 2)),
              ne.fixed_topology.layers.TFConv2DLayer(512, (3, 3), strides=(1, 1), activation="relu"),
              ne.fixed_topology.layers.TFFlattenLayer(),
              ne.fixed_topology.layers.TFDenseLayer(512, activation="relu"),
              ne.fixed_topology.layers.TFDenseLayer(512, activation="relu"),
              ne.fixed_topology.layers.TFDenseLayer(num_actions, activation="softmax")
          ],
          input_shape=(1, 107, 107, 3),
        ) 

        BASE_GENOME_EXTENDED.layers[-1] = ne.fixed_topology.layers.TFDenseLayer(num_actions, activation="softmax", input_shape=(1, 512 + action_history_size))

        return BASE_GENOME_EXTENDED
    else:
        BASE_GENOME = ne.fixed_topology.FixedTopologyGenome(
            layers=[#ne.fixed_topology.layers.TFFlattenLayer(),
                    #ne.fixed_topology.layers.TFDenseLayer((512 // self.maxpool_size) + self.action_history_size, activation="relu"),
                    ne.fixed_topology.layers.TFDenseLayer(11, activation="softmax")],
            input_shape=(1, (512 // maxpool_size) + action_history_size),
        )

        return BASE_GENOME

def get_population(algorithm="FIXED", population_size=65, maxpool_size=1, num_actions=11, action_history_size=11*10, processing_scheduler="POOL"):
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Algorithm {algorithm} not valid, must be one of {VALID_ALGORITHMS}")
    if processing_scheduler not in VALID_PROCESSING_SCHEDULERS:
        raise ValueError(f"Processing scheduler {processing_scheduler} not valid, must be one of {VALID_PROCESSING_SCHEDULERS}")
    
    if processing_scheduler == "RAY":
        scheduler = SCHEDULER_MAP[processing_scheduler](**RAY_CONFIG)
    elif processing_scheduler == "POOL":
        scheduler = SCHEDULER_MAP[processing_scheduler](num_processes=8)
    else:
        scheduler = SCHEDULER_MAP[processing_scheduler]()

    if algorithm == "NEAT":
        population = ne.neat.population.NeatPopulation(
            size=population_size,
            num_inputs=(512 // maxpool_size) + action_history_size,
            num_outputs=num_actions, 
            config=NEAT_CONFIG, 
            processing_scheduler=scheduler
        )

    elif algorithm == "FIXED-LARGE":
        population = ne.genetic_algorithm.GeneticPopulation(
            size=population_size,
            base_genome=get_base_genome(algorithm, maxpool_size=maxpool_size),
            processing_scheduler=scheduler
            #processing_scheduler=ne.processing.SerialProcessingScheduler() # use when debugging
        )
    else:
        population = ne.genetic_algorithm.GeneticPopulation(
            size=population_size,
            base_genome=get_base_genome(algorithm, maxpool_size=maxpool_size),
            #processing_scheduler=ne.processing.SerialProcessingScheduler() # use when debugging
            processing_scheduler=scheduler
        )

    return population


def load_weights_to_torch(model, genome, algorithm="FIXED"):
    """
    Takes a pytorch model and loads the weights in with a chosen genome.
    Will be used for checkpointing the evolved genome into a pytorch model.
    This will make it easier for the finished model to interact with the rest of the code.

    TOOD:
        Implement this function for the FIXED-LARGE algorithm
        For now we will just use the FIXED algorithm
    """
    device = model.fc6[0].weight.device
    if algorithm == "NEAT" or algorithm == "RANDOM":
        return
    elif algorithm == "FIXED-LARGE":
        return
    else:
        model.fc6[0].weight.data = torch.from_numpy(genome.layers[0].tf_layer.get_weights()[0].T).float().to(device)
        model.fc6[0].bias.data = torch.from_numpy(genome.layers[0].tf_layer.get_weights()[1]).float().to(device)

    return model


# Loads Pytroch weights into genomes with Tensorflow layers
# TODO: We also need a function for the other way around. 
# Loading Tensorflow weights into Pytorch layers
def load_weights_to_tf(model, fc6, genome, algorithm="FIXED"): 
    """
    Takes a genome and loads in the weights from the torch model.
    This can allow for pretrained weights to be used as a starting point for the NEAT algorithm, and do a form of transfer learning.

    TOOD: Something is not quite right here, the reward is not reflecting the pretrained weights being loaded.
          Maybe something is wrong with the way the weights are being loaded into the genome?
    """
    if algorithm == "NEAT" or algorithm == "RANDOM":
        return
    elif algorithm == "FIXED-LARGE":
        weights = [model.backbone[0][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.backbone[1][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.backbone[2][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.fc4_5[0][0].weight.detach().cpu().permute(1, 0).numpy(),
                   model.fc4_5[1][0].weight.detach().cpu().permute(1, 0).numpy(),
                   fc6[0].weight.detach().cpu().permute(1, 0).numpy()]

        bias = [model.backbone[0][0].bias.detach().cpu().numpy(), 
                model.backbone[1][0].bias.detach().cpu().numpy(),
                model.backbone[2][0].bias.detach().cpu().numpy(),
                model.fc4_5[0][0].bias.detach().cpu().numpy(),
                model.fc4_5[1][0].bias.detach().cpu().numpy(),
                fc6[0].bias.detach().cpu().numpy()]
        i = 0
        for layer in genome.layers:
            # check if tensorflow layer is a dense layer or a conv2d layer
            if isinstance(layer.tf_layer, tf.keras.layers.Dense) or isinstance(layer.tf_layer, tf.keras.layers.Conv2D):
                layer.tf_layer.set_weights([weights[i], bias[i]])
                i += 1
    else:
        genome.layers[0].tf_layer.set_weights([fc6[0].weight.detach().cpu().permute(1, 0).numpy(), fc6[0].bias.detach().cpu().numpy()])

def check_weights(model, fc6, genome, algorithm="FIXED"):
    """
    A function for checking that the weights of the genome are the same as the weights of the fc6 layer of the torch model.
    """
    if algorithm == "NEAT" or algorithm == "RANDOM":
        return
    elif algorithm == "FIXED-LARGE":
        weights = [model.backbone[0][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.backbone[1][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.backbone[2][0].weight.detach().cpu().permute(2, 3, 1, 0).numpy(),
                   model.fc4_5[0][0].weight.detach().cpu().permute(1, 0).numpy(),
                   model.fc4_5[1][0].weight.detach().cpu().permute(1, 0).numpy(),
                   fc6[0].weight.detach().cpu().permute(1, 0).numpy()]

        bias = [model.backbone[0][0].bias.detach().cpu().numpy(), 
                model.backbone[1][0].bias.detach().cpu().numpy(),
                model.backbone[2][0].bias.detach().cpu().numpy(),
                model.fc4_5[0][0].bias.detach().cpu().numpy(),
                model.fc4_5[1][0].bias.detach().cpu().numpy(),
                fc6[0].bias.detach().cpu().numpy()]
        i = 0
        for layer in genome.layers:
            # check if tensorflow layer is a dense layer or a conv2d layer
            if isinstance(layer.tf_layer, tf.keras.layers.Dense) or isinstance(layer.tf_layer, tf.keras.layers.Conv2D):
                genome_weights = layer.tf_layer.get_weights()
                assert np.allclose(genome_weights[0], weights[i])
                assert np.allclose(genome_weights[1], bias[i])
                i += 1
    else:
        genome_weights = genome.layers[0].tf_layer.get_weights()
        fc6_weights = [fc6[0].weight.detach().cpu().permute(1, 0).numpy().T, fc6[0].bias.detach().cpu().permute(1, 0).numpy()]
        assert np.allclose(genome_weights[0], fc6_weights[0])
    

def check_outputs(model, fc6, genome, algorithm="FIXED"):
    """
    Iterates throguh all the layers individually and checks that the output is the same.
    """
    # Lambda function that takes a nn.Sequential object and returns the first n layers
    get_layers = lambda x, n: nn.Sequential(*list(x.children())[:n])
    if algorithm == "NEAT" or algorithm == "RANDOM":
        return
    elif algorithm == "FIXED-LARGE":
        genome.process = functools.partial(process, genome) # Uncomment this line if using the FIXED-LARGE algorithm
        torch_layers = [get_layers(model.backbone[0], 2),
                        get_layers(model.backbone[1], 2),
                        get_layers(model.backbone[2], 2),
                        get_layers(model.fc4_5[0], 2),
                        get_layers(model.fc4_5[1], 2),
                        get_layers(fc6, 2)
                    ]

        tf_layers = [layer.tf_layer for layer in genome.layers 
                     if isinstance(layer.tf_layer, tf.keras.layers.Dense) or isinstance(layer.tf_layer, tf.keras.layers.Conv2D)]
        
        torch_test_inputs = [torch.randn((1, 3, 107, 107)),
                            #torch.randn((1, 96, 51, 51)),
                            torch.randn((1, 96, 25, 25)),
                            #torch.randn((1, 256, 11, 11)),
                            torch.randn((1, 256, 5, 5)),
                            torch.randn((1, 512*3*3)),
                            torch.randn((1, 512)),
                            torch.randn((1, 622))
                    ]
        
        tf_test_inputs = [torch_test_inputs[0].permute(0, 2, 3, 1).cpu().numpy(),
                          torch_test_inputs[1].permute(0, 2, 3, 1).cpu().numpy(),
                          torch_test_inputs[2].permute(0, 2, 3, 1).cpu().numpy(),
                          torch_test_inputs[3].cpu().numpy(),
                          torch_test_inputs[4].cpu().numpy(),
                          torch_test_inputs[5].cpu().numpy(),
        ]
        
        for i in range(len(torch_layers)):
            torch_layer = torch_layers[i]
            device = torch_layer[0].weight.device
            tf_layer = tf_layers[i]
            tf_test_input = tf_test_inputs[i]
            torch_test_input = torch_test_inputs[i].to(device)

            #torch_test_input = torch_test_input.view(torch_test_input.size(0), -1)
            torch_output = torch_layer(torch_test_input)
            tf_output = tf_layer(tf_test_input)
            if i <= 2:
                torch_output = torch_output.permute(0, 2, 3, 1)

            if i == 5:
                softmax = nn.Softmax(dim=1)
                torch_output = softmax(torch_output)

            # Check that the outputs are the same
            assert np.allclose(torch_output.detach().cpu().numpy(), tf_output, atol=1e-2)
        print("Checked weights for all layers: Outputs match!")

    elif algorithm == "FIXED":
        torch_layer = get_layers(fc6, 2)
        device = torch_layer[0].weight.device
        tf_layer = genome.layers[0].tf_layer
        inp = torch.randn((1, 622)).to(device)
        softmax = nn.Softmax(dim=1)

        torch_output = torch_layer(inp)
        torch_output = softmax(torch_output)

        tf_output = tf_layer(inp.cpu().numpy())

        assert np.allclose(torch_output.detach().cpu().numpy(), tf_output, atol=1e-5)
        print("Checked FC6 weights: Outputs match!")

def save_checkpoint(genome_path, checkpoint_path, original_model, algorithm, device):
    p = checkpoint_path / "pytorch_checkpoints"
    # Check if pytorch_checkpoints directory exists if not create it
    if not os.path.exists(p):
        os.makedirs(p)
    # Load the best genome
    base_genome = get_base_genome(algorithm)
    #genome_path = get_latest_checkpoint(genome_path)
    base_name = genome_path.name.split(".")[0]
    base_genome.load(genome_path)

    # Load the best genome into the pytorch model
    model = load_weights_to_torch(original_model, base_genome, algorithm=algorithm).to(device)

    # Save the model
    torch.save(model.state_dict(), p / f"{base_name}_torch.pth")

def get_latest_checkpoint(root, checkpoint_path):
    # Base name of the checkpoint files are:
    # genome_checkpoint_{n}.pkl 
    
    # Get all the files in the checkpoint directory
    files = os.listdir(checkpoint_path)
    files = [f for f in files if f.endswith(".pkl") and f.startswith("genome_checkpoint_")]
    # Sort the file names alphabetically (this should work because of the naming convention)
    files = sorted(files)

    return checkpoint_path / files[-1]

def get_all_checkpoints(root):
    checkpoints = []
    for path in root.iterdir():
        if path.is_dir():
            checkpoints += get_all_checkpoints(path)
        elif path.is_file() and path.suffix == ".pkl":
            path = root / path
            checkpoints.append(path)

    return checkpoints
