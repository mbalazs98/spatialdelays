from tonic.datasets import SSC
import mnist
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
#from callbacks import CSVLog

parser = ArgumentParser()
parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden neurons")
parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity of connections")
parser.add_argument("--delay_init", type=float, default=0, help="Initialise delays with this maximum value")
parser.add_argument("--delay_within", type=int, default=1, help="Initialise delays with this maximum value")
args = parser.parse_args()


unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

BATCH_SIZE = 32 
NUM_INPUT = 700 + 784
NUM_HIDDEN = args.num_hidden
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

EXAMPLE_TIME = 20.0
DT = 1.0
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels_mnist_train = mnist.train_labels() + 10
spikes_mnist_train = linear_latency_encode_data(
    mnist.train_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

labels_mnist_test = mnist.test_labels() + 10
spikes_mnist_test = linear_latency_encode_data(
    mnist.test_images(),
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
                        NUM_HIDDEN, record_spikes=True)
    hidden2 = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN//4, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        NUM_OUTPUT)

    # Connections
    input_hidden1 = Connection(input, hidden1, FixedProbability(p=700/1484, weight=Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    pre_ind_ssc, post_ind_ssc = np.meshgrid(np.arange(700), np.arange(NUM_HIDDEN))

    # Flatten the arrays to get 1D arrays of all pairs
    pre_ind_ssc = pre_ind_ssc.flatten()  # Length will be 700 * 512
    post_ind_ssc = post_ind_ssc.flatten()  # Length will be 700 * 512
    
    input_hidden1.connectivity.pre_ind = pre_ind_ssc
    input_hidden1.connectivity.post_ind = post_ind_ssc
    #need to zero out connections from other module
    
    input_hidden2 = Connection(input, hidden2, FixedProbability(p=784/1484, weight=Normal(mean=0.078, sd=0.045)),
               Exponential(5.0))

    pre_ind_mnist, post_ind_mnist = np.meshgrid(np.arange(700,1484), np.arange(NUM_HIDDEN//4))
    pre_ind_mnist = pre_ind_mnist.flatten()  # Length will be 700 * 256
    post_ind_mnist = post_ind_mnist.flatten()  # Length will be 700 * 256
    input_hidden2.connectivity.pre_ind = pre_ind_mnist
    input_hidden2.connectivity.post_ind = post_ind_mnist

    
    hidden1_hidden1 = Connection(hidden1, hidden1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    
    hidden2_hidden2 = Connection(hidden2, hidden2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    
    '''if args.sparsity != 1.0:
        hidden1_hidden2 = Connection(hidden1, hidden2, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden2_hidden1 = Connection(hidden2, hidden1, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
    else:
        hidden1_hidden2 = Connection(hidden1, hidden2, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden2_hidden1 = Connection(hidden2, hidden1, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
    hidden1_output = Connection(hidden1, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    
    hidden2_output = Connection(hidden2, output, Dense(Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))'''
    

    hidden1_output = Connection(hidden1, output, FixedProbability(p=0.5, weight=Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    pre_ind_ssc_out, post_ind_ssc_out = np.meshgrid(np.arange(NUM_HIDDEN), np.arange(10))

    # Flatten the arrays to get 1D arrays of all pairs
    pre_ind_ssc_out = pre_ind_ssc_out.flatten() 
    post_ind_ssc_out = post_ind_ssc_out.flatten()  
    hidden1_output.connectivity.pre_ind = pre_ind_ssc_out
    hidden1_output.connectivity.post_ind = post_ind_ssc_out
    #need to zero out connections from other module
    
    hidden2_output = Connection(hidden2, output, FixedProbability(p=0.5, weight=Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))

    pre_ind_mnist_out, post_ind_mnist_out = np.meshgrid(np.arange(NUM_HIDDEN//4), np.arange(10,20))
    pre_ind_mnist_out = pre_ind_mnist_out.flatten()
    post_ind_mnist_out = post_ind_mnist_out.flatten()
    input_hidden2.connectivity.pre_ind = pre_ind_mnist_out
    input_hidden2.connectivity.post_ind = post_ind_mnist_out
    
k_reg = {}

k_reg[hidden1] = 5e-11
k_reg[hidden2] = 1e-20

'''delay_learn_conns = [hidden1_hidden2,hidden2_hidden1]
if bool(args.delay_within):
    delay_learn_conns.append(hidden1_hidden1, hidden2_hidden2)'''

max_example_timesteps = int(np.ceil(latest_spike_time / DT))
serialiser = Numpy("checkpoints_" + unique_suffix)
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=k_reg, reg_lambda_lower=k_reg, 
                                reg_nu_upper=10, max_spikes=1500,
                                delay_learn_conns=[],#delay_learn_conns,
                                optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(0.1),
                                batch_size=BATCH_SIZE, rng_seed=0, distance_cost=0.0)

compiled_net = compiler.compile(network)

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
    callbacks = ["batch_progress_bar", SpikeRecorder(hidden1, key="hidden1_spikes", record_counts=True), SpikeRecorder(hidden2, key="hidden2_spikes", record_counts=True), EaseInSchedule(), Checkpoint(serialiser)]
    validation_callbacks = ["batch_progress_bar"]
    best_e, best_acc = 0, 0
    early_stop = 15
    for e in range(500):
        
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input: spikes_train},
                                            {output: labels_train},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_test}, validation_y={output: labels_test})

        
        
        hidden1_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in train_cb['hidden1_spikes']:
            hidden1_spikes += cb_d
        
        _input_hidden1 = compiled_net.connection_populations[input_hidden1]
        _input_hidden1.vars["g"].pull_from_device()
        g_values = _input_hidden1.vars["g"].values.reshape(700, NUM_HIDDEN)
        g_values[:, hidden1_spikes == 0] += 0.002
        g_values.ravel()
        _input_hidden1.vars["g"].push_to_device()
        
        hidden2_spikes = np.zeros(NUM_HIDDEN//4)
        for cb_d in train_cb['hidden2_spikes']:
            hidden2_spikes += cb_d

        _input_hidden2 = compiled_net.connection_populations[input_hidden2]
        _input_hidden2.vars["g"].pull_from_device()
        g_values = _input_hidden2.vars["g"].values.reshape(784, NUM_HIDDEN//4)
        g_values[:, hidden2_spikes == 0] += 0.002
        g_values.ravel()
        _input_hidden2.vars["g"].push_to_device()
        print("silent in first: ", 256-np.count_nonzero(hidden1_spikes), "silent in second: ", NUM_HIDDEN//4-np.count_nonzero(hidden2_spikes))

        

        if train_metrics[output].result > best_acc:
            best_acc = train_metrics[output].result
            best_e = e
            early_stop = 15
        else:
            early_stop -= 1
            if early_stop < 0:
                break
