from mnist import download_and_parse_mnist_file
import numpy as np
from collections import namedtuple

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Callback
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential


from ml_genn.compilers.event_prop_compiler import default_params
from ml_genn.utils.data import preprocess_tonic_spikes, linear_latency_encode_data
from argparse import ArgumentParser
from callbacks import CSVLog

import os
import re
PreprocessedSpikes = namedtuple("PreprocessedSpikes", ["end_spikes", "spike_times"])

parser = ArgumentParser()
parser.add_argument("--num_hidden", type=int, default=64, help="Number of hidden neurons for MNIST")
parser.add_argument("--sparsity", type=float, default=0.01, help="Sparsity of connections")
parser.add_argument("--delay_min", type=float, default=0.0, help="Initialise delays with this minimum value")
parser.add_argument("--delay_max", type=float, default=0.0, help="Initialise delays with this minimum value")
parser.add_argument("--delays_within", type=int, default=1, help="Have delays within the module")
parser.add_argument("--delay_within_min", type=float, default=0.0, help="Initialise within delays with this minimum value")
parser.add_argument("--delay_within_max", type=float, default=0.0, help="Initialise within delays with this maximum value")
parser.add_argument("--k_reg", type=float, default=5e-11, help="Firing regularisation strength for mnist")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()
np.random.seed(args.seed)

unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

BATCH_SIZE = 256 
NUM_INPUT = 784
NUM_HIDDEN = args.num_hidden
NUM_OUTPUT = 10
#NUM_OUTPUT = 20


class EaseInSchedule(Callback):
    def __init__(self):
        pass
    def set_params(self, compiled_network, **kwargs):
        self._optimiser = compiled_network.optimisers[0][0]
    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        if self._optimiser.alpha < 0.001 :
            self._optimiser.alpha = (self._optimiser.alpha) * (1.05 ** batch)
        else:
            self._optimiser.alpha = 0.001

labels_mnist_test = download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz", target_dir="../data")
mnist_test_images = download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz", target_dir="../data")
labels_mnist_train = download_and_parse_mnist_file("train-labels-idx1-ubyte.gz", target_dir="../data")
mnist_train_images = download_and_parse_mnist_file("train-images-idx3-ubyte.gz", target_dir="../data")

def linear_latency_encode(data: np.ndarray, max_time: float,
                               min_time: float = 0.0,
                               thresh: int = 1):
    time_range = max_time - min_time
    # Get boolean mask of spiking neurons
    spike_vector = data > thresh

    # Take cumulative sum to get end spikes
    end_spikes = np.cumsum(spike_vector)

    # Extract values of spiking pixels
    spike_pixels = data[spike_vector]

    # Calculate spike times
    spike_times = (((255.0 - spike_pixels) / 255.0) * time_range) + min_time

    # Add to list
    return PreprocessedSpikes(end_spikes, spike_times)


EXAMPLE_TIME = 20.0
DT = 1.0
spikes_mnist_train, spikes_mnist_test =  mnist_train_images, mnist_test_images



def merge_paired_spikes(spikes, labels_orig):
    # Determine the number of pairs based on the smaller dataset
    num_pairs = len(spikes)
    
    # Randomly sample indices for both datasets
    indices_1 = np.random.choice(len(spikes), num_pairs, replace=False)
    indices_2 = np.random.choice(len(spikes), num_pairs, replace=False)
    sequence = np.random.randint(2, size=num_pairs)
    sequenced_spikes_1, sequenced_spikes_2, labels_1, labels_2 = [], [], [], []
    for ind_1, ind_2, seq in zip(indices_1, indices_2, sequence):
        if seq == 1:
            sequenced_spikes_1.append(linear_latency_encode(
                spikes[ind_1],
                EXAMPLE_TIME - (2.0 * DT), 2.0 * DT))
            sequenced_spikes_2.append(linear_latency_encode(
                spikes[ind_2],
                EXAMPLE_TIME + 100.0 - (2.0 * DT), (2.0 + 100.0) * DT))
        else:
            sequenced_spikes_2.append(linear_latency_encode(
                spikes[ind_2],
                EXAMPLE_TIME - (2.0 * DT), 2.0 * DT))
            sequenced_spikes_1.append(linear_latency_encode(
                spikes[ind_1],
                EXAMPLE_TIME + 100.0 - (2.0 * DT), (2.0 + 100.0) * DT))
        labels_1.append(labels_orig[ind_1])
        labels_2.append(labels_orig[ind_2])
    return sequenced_spikes_1, sequenced_spikes_2, labels_1, labels_2

network = Network(default_params)
network_weights = "checkpoints_mnist_sequence_" + unique_suffix
pattern = re.compile(r"^(\d+)-")

last_epoch = 0

for filename in os.listdir(network_weights):
    match = pattern.match(filename)
    if match:
        num = int(match.group(1))  # Convert to integer
        if num > last_epoch:
            last_epoch = num

best_epoch = last_epoch - 16

with network:
    # Populations
    input_1 = Population(SpikeInput(max_spikes= BATCH_SIZE * 784),
                       NUM_INPUT)
    input_2 = Population(SpikeInput(max_spikes= BATCH_SIZE * 784),
                       NUM_INPUT)
    hidden_1 = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, record_spikes=True)
    hidden_2 = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, record_spikes=True)
    output_1 = Population(LeakyIntegrate(tau_mem=20.0, readout="max_var"),
                        NUM_OUTPUT)
    output_2 = Population(LeakyIntegrate(tau_mem=20.0, readout="max_var"),
                        NUM_OUTPUT)

    # Connections
    input_hidden_1 = Connection(input_1, hidden_1, Dense(Normal(mean=0.078, sd=0.045)),
               Exponential(5.0))
    
    input_hidden_2 = Connection(input_2, hidden_2, Dense(Normal(mean=0.078, sd=0.045)),
               Exponential(5.0))

    
    hidden_1_hidden_1 = Connection(hidden_1, hidden_1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_within_min,args.delay_within_max)),
               Exponential(5.0), max_delay_steps=150)
    hidden_2_hidden_2 = Connection(hidden_2, hidden_2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_within_min,args.delay_within_max)),
               Exponential(5.0), max_delay_steps=150)
    if args.sparsity == 1.0:
        hidden_1_hidden_2 = Connection(hidden_1, hidden_2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_min,args.delay_max)),
                Exponential(5.0), max_delay_steps=150)
        hidden_2_hidden_1 = Connection(hidden_2, hidden_1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_min,args.delay_max)),
                Exponential(5.0), max_delay_steps=150)
    elif args.sparsity > 0.0:
        connectivity_hidden_1_hidden_2 = FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_min,args.delay_max))
        connectivity_hidden_1_hidden_2.pre_ind = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop3-pre_ind.npy")
        connectivity_hidden_1_hidden_2.post_ind = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop3-post_ind.npy")
        hidden_1_hidden_2 = Connection(hidden_1, hidden_2, connectivity_hidden_1_hidden_2,
                Exponential(5.0), max_delay_steps=150, name="Conn_Pop2_Pop3")
        connectivity_hidden_2_hidden_1 = FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(args.delay_min,args.delay_max))
        connectivity_hidden_2_hidden_1.pre_ind = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop2-pre_ind.npy")
        connectivity_hidden_2_hidden_1.post_ind = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop2-post_ind.npy")
        hidden_2_hidden_1 = Connection(hidden_2, hidden_1, connectivity_hidden_2_hidden_1,
                Exponential(5.0), max_delay_steps=150)

    hidden1_output1 = Connection(hidden_1, output_1, Dense(weight=Normal(mean=0.2, sd=0.37)),
               Exponential(5.0))
    hidden1_output2 = Connection(hidden_1, output_2, Dense(weight=Normal(mean=0.2, sd=0.37)),
               Exponential(5.0))
    
k_reg = {}

k_reg[hidden_1] = args.k_reg
k_reg[hidden_2] = args.k_reg

max_example_timesteps = int(np.ceil(EXAMPLE_TIME / DT)) +  100
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=k_reg, reg_lambda_lower=0, 
                                reg_nu_upper=1, max_spikes=1500,
                                delay_learn_conns=[],
                                optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(0.0), weight_fix_conns=[input_hidden_1, input_hidden_2, hidden_1_hidden_1, hidden_2_hidden_2, hidden_1_hidden_2, hidden_2_hidden_1],
                                batch_size=BATCH_SIZE, rng_seed=args.seed)

model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

# Apply augmentation to events and preprocess

spikes_train_1, spikes_train_2, labels_train_1, labels_train_2 = merge_paired_spikes(spikes_mnist_train, labels_mnist_train)
spikes_test_1, spikes_test_2, labels_test_1, labels_test_2  = merge_paired_spikes(spikes_mnist_test, labels_mnist_test)

with compiled_net:
    # Loop through epochs
    callbacks = [CSVLog(f"results/train_output_mnist_sequence_probe_correct_{unique_suffix}.csv", output_1), CSVLog(f"results/train_output_mnist_sequence_probe_incorrect_{unique_suffix}.csv", output_2), EaseInSchedule()]
    validation_callbacks = [CSVLog(f"results/valid_output_mnist_sequence_probe_correct_{unique_suffix}.csv", output_1), CSVLog(f"results/valid_output_mnist_sequence_probe_incorrect_{unique_suffix}.csv", output_2)]
    best_e, best_acc_1, best_acc_2 = 0, 0, 0
    early_stop = 15
    _input_hidden_1 = compiled_net.connection_populations[input_hidden_1]
    _input_hidden_1.vars["g"].pull_from_device()
    _input_hidden_1.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop0_Pop2-g.npy")
    _input_hidden_1.vars["g"].push_to_device()
    _input_hidden_2 = compiled_net.connection_populations[input_hidden_2]
    _input_hidden_2.vars["g"].pull_from_device()
    _input_hidden_2.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop1_Pop3-g.npy")
    _input_hidden_2.vars["g"].push_to_device()

    _hidden1_hidden_1 = compiled_net.connection_populations[hidden_1_hidden_1]
    _hidden1_hidden_1.vars["g"].pull_from_device()
    _hidden1_hidden_1.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop2-g.npy")
    _hidden1_hidden_1.vars["g"].push_to_device()
    _hidden2_hidden_2 = compiled_net.connection_populations[hidden_2_hidden_2]
    _hidden2_hidden_2.vars["g"].pull_from_device()
    _hidden2_hidden_2.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop3-g.npy")
    _hidden2_hidden_2.vars["g"].push_to_device()

    if args.delays_within:
        _hidden1_hidden_1.vars["d"].pull_from_device()
        _hidden1_hidden_1.vars["d"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop2-d.npy")
        _hidden1_hidden_1.vars["d"].push_to_device()
        _hidden2_hidden_2.vars["d"].pull_from_device()
        _hidden2_hidden_2.vars["d"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop3-d.npy")
        _hidden2_hidden_2.vars["d"].push_to_device()
    
    _hidden1_hidden_2 = compiled_net.connection_populations[hidden_1_hidden_2]
    _hidden1_hidden_2.vars["g"].pull_from_device()
    _hidden1_hidden_2.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop3-g.npy")
    _hidden1_hidden_2.vars["g"].push_to_device()
    _hidden1_hidden_2.vars["d"].pull_from_device()
    _hidden1_hidden_2.vars["d"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop2_Pop3-d.npy")
    _hidden1_hidden_2.vars["d"].push_to_device()
    _hidden2_hidden_1 = compiled_net.connection_populations[hidden_2_hidden_1]
    _hidden2_hidden_1.vars["g"].pull_from_device()
    _hidden2_hidden_1.vars["g"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop2-g.npy")
    _hidden2_hidden_1.vars["g"].push_to_device()
    _hidden2_hidden_1.vars["d"].pull_from_device()
    _hidden2_hidden_1.vars["d"].values = np.load(network_weights+"/"+str(best_epoch)+"-Conn_Pop3_Pop2-d.npy")
    _hidden2_hidden_1.vars["d"].push_to_device()
        
    for e in range(500):
        
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input_1: spikes_train_1, input_2: spikes_train_2},
                                            {output_1: labels_train_1, output_2: labels_train_2},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input_1: spikes_test_1, input_2: spikes_test_2}, validation_y={output_1: labels_test_1, output_2: labels_test_2})

        

        
        not_improved = True
        if train_metrics[output_1].result > best_acc_1:
            best_acc_1 = train_metrics[output_1].result
            best_e = e
            early_stop = 15
            not_improved = False
        if train_metrics[output_2].result > best_acc_2:
            best_acc_2 = train_metrics[output_2].result
            best_e = e
            early_stop = 15
            not_improved = False
        if not_improved:
            early_stop -= 1
            if early_stop < 0:
                break



    
