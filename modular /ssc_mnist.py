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
from ml_genn.utils.data import preprocess_tonic_spikes, linear_latency_encode_data
from argparse import ArgumentParser
#from callbacks import CSVLog

parser = ArgumentParser()
parser.add_argument("--num_hidden_ssc", type=int, default=256, help="Number of hidden neurons for SSC")
parser.add_argument("--num_hidden_mnist", type=int, default=64, help="Number of hidden neurons for MNIST")
parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity of connections")
parser.add_argument("--delay_init", type=float, default=0.0, help="Initialise delays with this maximum value")
parser.add_argument("--delays_lr", type=float, default=0.1, help="Delay learning rate")
parser.add_argument("--delay_within", type=int, default=1, help="Initialise delays with this maximum value")
parser.add_argument("--distance_cost", type=float, default=0.0, help="Distance regularisation strength")
parser.add_argument("--reg_nu_upper", type=int, default=14, help="Firing rate")
parser.add_argument("--k_reg_ssc", type=float, default=5e-12, help="Firing regularisation strength for ssc")
parser.add_argument("--k_reg_mnist", type=float, default=5e-20, help="Firing regularisation strength for mnist")
args = parser.parse_args()


unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items() if not arg.startswith("__"))

BATCH_SIZE = 256 
NUM_INPUT_SSC = 700
NUM_INPUT_MNIST = 784
NUM_HIDDEN_SSC = args.num_hidden_ssc
NUM_HIDDEN_MNIST = args.num_hidden_mnist
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

EXAMPLE_TIME = 20.0
DT = 1.0
mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
labels_mnist_train = mnist.train_labels()
labels_mnist_train += 10
spikes_mnist_train = linear_latency_encode_data(
    mnist.train_images(),
    EXAMPLE_TIME - (2.0 * DT), 2.0 * DT)

labels_mnist_test = mnist.test_labels()
labels_mnist_test += 10
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

        spikes_ssc_train.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          dataset.sensor_size))
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

        spikes_ssc_test.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          dataset.sensor_size))
        labels_ssc_test.append(label)
        
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

def merge_paired_spikes(spikes_ssc, spikes_mnist, labels_ssc, labels_mnist):
    # Determine the number of pairs based on the smaller dataset
    num_pairs = min(len(spikes_ssc), len(spikes_mnist))
    
    # Randomly sample indices for both datasets
    indices_ssc = np.random.choice(len(spikes_ssc), num_pairs, replace=False)
    indices_mnist = np.random.choice(len(spikes_mnist), num_pairs, replace=False)
    labels = [((l1 + l2) % 2) * l1 + (1-((l1 + l2) % 2)) * l2 for l1, l2 in zip([labels_ssc[idx] for idx in indices_ssc],[labels_mnist[idx] for idx in indices_mnist])]
    
    for l1, l2, l_true in zip([labels_ssc[idx] for idx in indices_ssc][:10], [labels_mnist[idx] for idx in indices_mnist][:10], labels[:10]):
        print(l1, l2, l_true)
    
    return [spikes_ssc[idx] for idx in indices_ssc], [spikes_mnist[idx] for idx in indices_mnist], labels



network = Network(default_params)
with network:
    # Populations
    input_SSC = Population(SpikeInput(max_spikes= BATCH_SIZE * max_spikes),
                       NUM_INPUT_SSC)
    input_MNIST = Population(SpikeInput(max_spikes= BATCH_SIZE * 784),
                       NUM_INPUT_MNIST)
    hidden_SSC = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN_SSC, record_spikes=True)
    hidden_MNIST = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN_MNIST, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        NUM_OUTPUT)

    # Connections
    input_hidden_SSC = Connection(input_SSC, hidden_SSC, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    
    input_hidden_MNIST = Connection(input_MNIST, hidden_MNIST, Dense(Normal(mean=0.078, sd=0.045)),
               Exponential(5.0))

    
    hidden_SSC_hidden_SSC = Connection(hidden_SSC, hidden_SSC, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    hidden_MNIST_hidden_MNIST = Connection(hidden_MNIST, hidden_MNIST, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,0)),
               Exponential(5.0), max_delay_steps=1000)
    if args.sparsity == 1.0:
        hidden_SSC_hidden_MNIST = Connection(hidden_SSC, hidden_MNIST, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden_MNIST_hidden_SSC = Connection(hidden_MNIST, hidden_SSC, Dense(Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
    elif args.sparsity > 0.0:
        hidden_SSC_hidden_MNIST = Connection(hidden_SSC, hidden_MNIST, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)
        hidden_MNIST_hidden_SSC = Connection(hidden_MNIST, hidden_SSC, FixedProbability(p=args.sparsity, weight=Normal(mean=0.0, sd=0.02), delay=Uniform(0,args.delay_init)),
                Exponential(5.0), max_delay_steps=1000)

    '''hidden_output_SSC = Connection(hidden_SSC, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    
    hidden_output_MNIST = Connection(hidden_MNIST, output, Dense(Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))'''
    hidden1_output = Connection(hidden_SSC, output, FixedProbability(p=0.5, weight=Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    pre_ind_ssc_out, post_ind_ssc_out = np.meshgrid(np.arange(NUM_HIDDEN_SSC), np.arange(10))

    # Flatten the arrays to get 1D arrays of all pairs
    pre_ind_ssc_out = pre_ind_ssc_out.flatten() 
    post_ind_ssc_out = post_ind_ssc_out.flatten()  
    hidden1_output.connectivity.pre_ind = pre_ind_ssc_out
    hidden1_output.connectivity.post_ind = post_ind_ssc_out
    #need to zero out connections from other module
    
    hidden2_output = Connection(hidden_MNIST, output, FixedProbability(p=0.5, weight=Normal(mean=0.007, sd=0.73)),
               Exponential(5.0))

    pre_ind_mnist_out, post_ind_mnist_out = np.meshgrid(np.arange(NUM_HIDDEN_MNIST), np.arange(10,20))
    pre_ind_mnist_out = pre_ind_mnist_out.flatten()
    post_ind_mnist_out = post_ind_mnist_out.flatten()
    hidden2_output.connectivity.pre_ind = pre_ind_mnist_out
    hidden2_output.connectivity.post_ind = post_ind_mnist_out
    
k_reg = {}

k_reg[hidden_SSC] = args.k_reg_ssc
k_reg[hidden_MNIST] = args.k_reg_mnist
if args.sparsity > 0.0:
    delay_learn_conns = [hidden_MNIST_hidden_SSC,hidden_SSC_hidden_MNIST]
else:
    delay_learn_conns = []
if bool(args.delay_within):
    delay_learn_conns.append(hidden_MNIST_hidden_MNIST)
    delay_learn_conns.append(hidden_SSC_hidden_SSC)


max_example_timesteps = int(np.ceil(latest_spike_time / DT))
serialiser = Numpy("checkpoints_" + unique_suffix)
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=k_reg, reg_lambda_lower=k_reg, 
                                reg_nu_upper=10, max_spikes=1500,
                                delay_learn_conns=delay_learn_conns,
                                optimiser=Adam(0.001 * 0.01), delay_optimiser=Adam(0.1),
                                batch_size=BATCH_SIZE, rng_seed=0, distance_cost=0.0)

compiled_net = compiler.compile(network)

spikes_train_ssc, spikes_train_mnist, labels_train = merge_paired_spikes(spikes_ssc_train, spikes_mnist_train, labels_ssc_train, labels_mnist_train)
spikes_test_ssc, spikes_test_mnist, labels_test = merge_paired_spikes(spikes_ssc_test, spikes_mnist_test, labels_ssc_test, labels_mnist_test)

with compiled_net:
    # Loop through epochs
    callbacks = ["batch_progress_bar", SpikeRecorder(hidden_SSC, key="hidden_SSC_spikes", record_counts=True), SpikeRecorder(hidden_MNIST, key="hidden_MNIST_spikes", record_counts=True), EaseInSchedule(), Checkpoint(serialiser)]
    validation_callbacks = ["batch_progress_bar"]
    best_e, best_acc = 0, 0
    early_stop = 15
    for e in range(500):
        
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input_SSC: spikes_train_ssc, input_MNIST: spikes_train_mnist},
                                            {output: labels_train},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input_SSC: spikes_test_ssc, input_MNIST: spikes_test_mnist}, validation_y={output: labels_test})

        
        
        hidden_SSC_spikes = np.zeros(NUM_HIDDEN_SSC)
        for cb_d in train_cb['hidden_SSC_spikes']:
            hidden_SSC_spikes += cb_d
        

        _input_hidden_SSC = compiled_net.connection_populations[input_hidden_SSC]
        _input_hidden_SSC.vars["g"].pull_from_device()
        g_view = _input_hidden_SSC.vars["g"].view.reshape((700, NUM_HIDDEN_SSC))
        g_view[:,hidden_SSC_spikes==0] += 0.002
        _input_hidden_SSC.vars["g"].push_to_device()
        
        hidden_MNIST_spikes = np.zeros(NUM_HIDDEN_MNIST)
        for cb_d in train_cb['hidden_MNIST_spikes']:
            hidden_MNIST_spikes += cb_d

        _input_hidden_MNIST = compiled_net.connection_populations[input_hidden_MNIST]
        _input_hidden_MNIST.vars["g"].pull_from_device()
        g_view = _input_hidden_MNIST.vars["g"].view.reshape((784, NUM_HIDDEN_MNIST))
        g_view[:,hidden_MNIST_spikes==0] += 0.002
        _input_hidden_MNIST.vars["g"].push_to_device()

        print("SSC: ", np.count_nonzero(hidden_SSC_spikes==0), "MNIST: ", np.count_nonzero(hidden_MNIST_spikes==0))

        

        if train_metrics[output].result > best_acc:
            best_acc = train_metrics[output].result
            best_e = e
            early_stop = 15
        else:
            early_stop -= 1
            if early_stop < 0:
                break

    
