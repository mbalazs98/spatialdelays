from tonic.datasets import SSC
from mnist import download_and_parse_mnist_file
import numpy as np


from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, Checkpoint, VarRecorder
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal, Uniform
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential


from ml_genn.compilers.event_prop_compiler import default_params
from ml_genn.utils.data import preprocess_tonic_spikes
from argparse import ArgumentParser
from callbacks import CSVLog

import os

parser = ArgumentParser()
parser.add_argument("--num_hidden_ssc", type=int, default=256, help="Number of hidden neurons for SSC")
parser.add_argument("--num_hidden_mnist", type=int, default=64, help="Number of hidden neurons for MNIST")
parser.add_argument("--sparsity", type=float, default=0.01, help="Sparsity of connections")
parser.add_argument("--delay_init", type=float, default=0.0, help="Initialise delays with this maximum value")
parser.add_argument("--delays_lr", type=float, default=0.1, help="Delay learning rate")
parser.add_argument("--delay_within", type=int, default=1, help="Initialise delays with this maximum value")
parser.add_argument("--distance_cost", type=float, default=0.0, help="Distance regularisation strength")
parser.add_argument("--reg_nu_upper", type=int, default=14, help="Firing rate")
parser.add_argument("--k_reg_ssc", type=float, default=5e-11, help="Firing regularisation strength for ssc")
parser.add_argument("--k_reg_mnist", type=float, default=5e-11, help="Firing regularisation strength for mnist")
args = parser.parse_args()


unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

BATCH_SIZE = 256 
NUM_INPUT = 700 + 784
NUM_HIDDEN_SSC = args.num_hidden_ssc
NUM_HIDDEN_MNIST = args.num_hidden_mnist
#NUM_OUTPUT = 10
NUM_OUTPUT = 20


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


def linear_latency_encode_data(data: np.ndarray, max_time: float,
                               min_time: float = 0.0,
                               thresh: int = 1):
    """Generate PreprocessedSpikes format stimuli by linearly
    latency encoding static data

    Args:
        data:       Data in uint8 format
        max_time:   Spike time for inputs with a value of 0
        min_time:   Spike time for inputs with a value of 255
        thresh:     Threshold values must reach for any spike to be emitted
    """
    # **TODO** handle floating point data
    # Loop through examples
    time_range = max_time - min_time
    dataset = []
    for i in range(len(data)):
        # Get boolean mask of spiking neurons
        spike_vector = data[i] > thresh
        
        # Extract values of spiking pixels
        spike_pixels = data[i, spike_vector]
        # Calculate spike times
        spike_times = (((255.0 - spike_pixels) / 255.0) * time_range) + min_time
        
        #dataset.append({"x": np.arange(700, 700+784), "t": spike_times})
        dtype = [('t', int), ('x', int), ('p', int)]
        structured_array = np.zeros(len(spike_times), dtype=dtype)
        structured_array["t"] = spike_times
        structured_array["x"] = np.arange(700, 700+784)[spike_vector.flatten()]
        structured_array["p"] = np.ones(len(spike_times))
        dataset.append(structured_array)

    return dataset


labels_mnist_test = download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz", target_dir="../data")
labels_mnist_test += 10
mnist_test_images = download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz", target_dir="../data")
labels_mnist_train = download_and_parse_mnist_file("train-labels-idx1-ubyte.gz", target_dir="../data")
labels_mnist_train += 10
mnist_train_images = download_and_parse_mnist_file("train-images-idx3-ubyte.gz", target_dir="../data")


EXAMPLE_TIME = 20.0
DT = 1.0
spikes_mnist_train = linear_latency_encode_data(
    mnist_train_images,
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)
spikes_mnist_test = linear_latency_encode_data(
    mnist_test_images,
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)


dataset = SSC(save_to="../data", split="train")
max_spikes = 0
latest_spike_time = 0
spikes_ssc_train = []
labels_ssc_train = []
for i, data in enumerate(dataset):
    events, label = data
    if label < 10:
        # Add raw events and label to list

        spikes_ssc_train.append(events)
        labels_ssc_train.append(label)
        
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

dataset = SSC(save_to="../data", split="test")
spikes_ssc_test = []
labels_ssc_test = []
for i, data in enumerate(dataset):
    events, label = data
    if label < 10:
        # Add raw events and label to list

        spikes_ssc_test.append(events)
        labels_ssc_test.append(label)
        
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

def merge_paired_spikes(spikes1, spikes2, labels1, labels2):
    # Determine the number of pairs based on the smaller dataset
    num_pairs = min(len(spikes1), len(spikes2))
    
    # Randomly sample indices for both datasets
    indices1 = np.random.choice(len(spikes1), num_pairs, replace=False)
    indices2 = np.random.choice(len(spikes2), num_pairs, replace=False)
    # Extract the paired subsets
    paired_spikes1 = [spikes1[idx] for idx in indices1]
    paired_spikes2 = [spikes2[idx] for idx in indices2]
    paired_labels1 = [labels1[idx] for idx in indices1]
    paired_labels2 = [labels2[idx] for idx in indices2]
    
    # Merge the paired dictionaries
    merged_spikes = []
    merged_labels = []
    for s1, s2, l1, l2 in zip(paired_spikes1, paired_spikes2, paired_labels1, paired_labels2):
        merged_spike = np.zeros(len(s1) + len(s2), dtype=s1.dtype)
        
        # Copy data from both arrays
        merged_spike[:len(s1)] = s1
        merged_spike[len(s1):] = s2
        
        # Sort by time field (assuming 't' is the time field)
        merged_spike.sort(order='t')
        merged_spikes.append(merged_spike)
        sigma = (l1 + l2) % 2
        merged_labels.append(sigma * l1 + (1-sigma) * l2)
    
    # Sort merged spikes by time 't'
    #merged_spikes.sort(key=lambda x: x['t'])
    
    return merged_spikes, merged_labels



network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes= BATCH_SIZE * (784 + max_spikes)),
                       NUM_INPUT)
    hidden1 = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN_SSC, record_spikes=True)
    hidden2 = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN_MNIST, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        NUM_OUTPUT)

    # Connections
    input_hidden1 = Connection(input, hidden1, FixedProbability(p=700/1484, weight=Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    pre_ind, post_ind = np.meshgrid(np.arange(700), np.arange(NUM_HIDDEN_SSC))

    # Flatten the arrays to get 1D arrays of all pairs
    pre_ind = pre_ind.flatten()  # Length will be 700 * 256
    post_ind = post_ind.flatten()  # Length will be 700 * 256
    input_hidden1.connectivity.pre_ind = pre_ind
    input_hidden1.connectivity.post_ind = post_ind
    #need to zero out connections from other module
    
    input_hidden2 = Connection(input, hidden2, FixedProbability(p=784/1484, weight=Normal(mean=0.078, sd=0.045)),
               Exponential(5.0))

    pre_ind, post_ind = np.meshgrid(np.arange(700,1484), np.arange(NUM_HIDDEN_MNIST))
    pre_ind = pre_ind.flatten()  # Length will be 784 * 64
    post_ind = post_ind.flatten()  # Length will be 784 * 64
    input_hidden2.connectivity.pre_ind = pre_ind
    input_hidden2.connectivity.post_ind = post_ind

    
    hidden1_hidden1 = Connection(hidden1, hidden1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    hidden2_hidden2 = Connection(hidden2, hidden2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    if args.sparsity == 1.0:
        hidden1_hidden2 = Connection(hidden1, hidden2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden2_hidden1 = Connection(hidden2, hidden1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
    elif args.sparsity > 0.0:
        hidden1_hidden2 = Connection(hidden1, hidden2, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden2_hidden1 = Connection(hidden2, hidden1, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
    
    '''hidden1_output = Connection(hidden1, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    hidden2_output = Connection(hidden2, output, Dense(Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))'''
    hidden1_output = Connection(hidden1, output, FixedProbability(p=0.5, weight=Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    pre_ind_ssc_out, post_ind_ssc_out = np.meshgrid(np.arange(NUM_HIDDEN_SSC), np.arange(10))

    # Flatten the arrays to get 1D arrays of all pairs
    pre_ind_ssc_out = pre_ind_ssc_out.flatten() 
    post_ind_ssc_out = post_ind_ssc_out.flatten()  
    hidden1_output.connectivity.pre_ind = pre_ind_ssc_out
    hidden1_output.connectivity.post_ind = post_ind_ssc_out
    #need to zero out connections from other module
    
    hidden2_output = Connection(hidden2, output, FixedProbability(p=0.5, weight=Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))

    pre_ind_mnist_out, post_ind_mnist_out = np.meshgrid(np.arange(NUM_HIDDEN_MNIST), np.arange(10,20))
    pre_ind_mnist_out = pre_ind_mnist_out.flatten()
    post_ind_mnist_out = post_ind_mnist_out.flatten()
    hidden2_output.connectivity.pre_ind = pre_ind_mnist_out
    hidden2_output.connectivity.post_ind = post_ind_mnist_out
    

k_reg = {}

k_reg[hidden1] = args.k_reg_ssc
k_reg[hidden2] = args.k_reg_mnist
if args.sparsity > 0.0:
    delay_learn_conns = [hidden1_hidden2,hidden2_hidden1]
else:
    delay_learn_conns = []
if bool(args.delay_within):
    delay_learn_conns.append(hidden1_hidden1)
    delay_learn_conns.append(hidden2_hidden2)

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
serialiser = Numpy("checkpoints_" + unique_suffix)
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=k_reg, reg_lambda_lower=k_reg, 
                                reg_nu_upper=args.reg_nu_upper, max_spikes=1500,
                                delay_learn_conns=delay_learn_conns,
                                optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(0.1),
                                batch_size=BATCH_SIZE, rng_seed=0)

model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)

# Apply augmentation to events and preprocess
merged_spikes, merged_labels = merge_paired_spikes(spikes_ssc_train, spikes_mnist_train, labels_ssc_train, labels_mnist_train)
spikes_train = []
labels_train = []
for events, label in zip(merged_spikes, merged_labels):
    spikes_train.append(preprocess_tonic_spikes(events, ('t', 'x', 'p'),
                                            (1484, 1, 1)))
    labels_train.append(label)

merged_spikes, merged_labels = merge_paired_spikes(spikes_ssc_test, spikes_mnist_test, labels_ssc_test, labels_mnist_test)
spikes_test = []
labels_test = []
for events, label in zip(merged_spikes, merged_labels):
    spikes_test.append(preprocess_tonic_spikes(events, ('t', 'x', 'p'),
                                            (1484, 1, 1)))
    labels_test.append(label)

with compiled_net:
    # Loop through epochs
    callbacks = [CSVLog(f"results/train_output_{unique_suffix}.csv", output),  SpikeRecorder(hidden2, key="hidden2_spikes", record_counts=True), SpikeRecorder(hidden1, key="hidden1_spikes", record_counts=True), EaseInSchedule(), Checkpoint(serialiser)]
    validation_callbacks = [CSVLog(f"results/valid_output_{unique_suffix}.csv", output)]
    best_e, best_acc = 0, 0
    early_stop = 15

    for e in range(500):
        
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input: spikes_train},
                                            {output: labels_train},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_test}, validation_y={output: labels_test})

        
        
        hidden1_spikes = np.zeros(NUM_HIDDEN_SSC)
        for cb_d in train_cb['hidden1_spikes']:
            hidden1_spikes += cb_d
        
        _input_hidden1 = compiled_net.connection_populations[input_hidden1]
        _input_hidden1.vars["g"].pull_from_device()
        g_values = _input_hidden1.vars["g"].values
        g_values.reshape(700, NUM_HIDDEN_SSC, order='F')[:, hidden1_spikes == 0] += 0.002
        _input_hidden1.vars["g"].push_to_device()
        
        hidden2_spikes = np.zeros(NUM_HIDDEN_MNIST)
        for cb_d in train_cb['hidden2_spikes']:
            hidden2_spikes += cb_d

        _input_hidden2 = compiled_net.connection_populations[input_hidden2]
        _input_hidden2.vars["g"].pull_from_device()
        g_values = _input_hidden2.vars["g"].values
        g_values.reshape(784, NUM_HIDDEN_MNIST, order='F')[:, hidden2_spikes == 0] += 0.002
        _input_hidden2.vars["g"].push_to_device()


        
        
        if train_metrics[output].result > best_acc:
            best_acc = train_metrics[output].result
            best_e = e
            early_stop = 15
        else:
            early_stop -= 1
            if early_stop < 0:
                break
        
    compiled_net.save_connectivity((best_e,), serialiser)
