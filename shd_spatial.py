import numpy as np

from hashlib import md5

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, Checkpoint
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params

from argparse import ArgumentParser

import os

from scipy.spatial.distance import cdist

import json


parser = ArgumentParser()
parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden neurons")
parser.add_argument("--sparsity", type=float, default=0.01, help="Sparsity of connections")
parser.add_argument("--k_reg", type=float, default=5e-11, help="Spike regularisation strength")
parser.add_argument("--pos_lr", type=float, default=0.1, help="Learning rate for the positions")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

NUM_HIDDEN = args.num_hidden
BATCH_SIZE = 256
NUM_EPOCHS = 300
DT = 1.0


# Class implementing simple augmentation where all events
# in example are shifted in space by normally-distributed value
np.random.seed(args.seed)

# Figure out unique suffix for model data
unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items())

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




dataset = SHD(save_to="../data", train=True)
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

# Loop through dataset
max_spikes = 0
latest_spike_time = 0
raw_dataset = []
classes = [[] for _ in range(20)]
for i, data in enumerate(dataset):
    events, label = data
    events = np.delete(events, np.where(events["t"] >= 1000000))
    # Add raw events and label to list
    classes[label].append(len(raw_dataset))
    raw_dataset.append((events, label))
    
    # Calculate max spikes and max times
    max_spikes = max(max_spikes, len(events))
    latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)

dataset = SHD(save_to="../data", train=False)

spikes_test = []
labels_test = []
for i in range(len(dataset)):
    events, label = dataset[i]
    events = np.delete(events, np.where(events["t"] >= 1000000))
    spikes_test.append(preprocess_tonic_spikes(events, dataset.ordering,
                                            dataset.sensor_size))
    labels_test.append(label)

# Determine max spikes and latest spike time
max_spikes = max(max_spikes, calc_max_spikes(spikes_test))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_test))



serialiser = Numpy("checkpoints_space" + unique_suffix)
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0,
                                           tau_refrac=None),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)
    
    x_pos = np.random.normal(loc=0.0, scale=1.0, size=NUM_HIDDEN)#np.zeros(NUM_HIDDEN)
    y_pos = np.random.normal(loc=0.0, scale=1.0, size=NUM_HIDDEN)#np.zeros(NUM_HIDDEN)
    z_pos = np.random.normal(loc=0.0, scale=1.0, size=NUM_HIDDEN)#np.zeros(NUM_HIDDEN)

    pos_dic = {}
    pos_dic[hidden] = (x_pos, y_pos, z_pos)

    points = np.column_stack((x_pos, y_pos, z_pos))
    dist = cdist(points, points)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    if args.sparsity == 1.0:
        Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02), dist),
                Exponential(5.0), max_delay_steps=int(1000))
    else:
        Conn_Pop1_Pop1 = Connection(hidden, hidden, FixedProbability(p=0.5, weight=Normal(mean=0.0, sd=0.02)),
                Exponential(5.0), max_delay_steps=int(1000))
        # All possible (pre, post) pairs
        all_pairs = [(i, j) for i in range(NUM_HIDDEN) for j in range(NUM_HIDDEN)]
        sampled_pairs = [all_pairs[idx] for idx in np.random.choice(len(all_pairs), size=max(1,int(NUM_HIDDEN * NUM_HIDDEN * args.sparsity)), replace=False)]
        Conn_Pop1_Pop1.connectivity.pre_ind = np.array([pair[0] for pair in sampled_pairs], dtype=int)
        Conn_Pop1_Pop1.connectivity.post_ind = np.array([pair[1] for pair in sampled_pairs], dtype=int)
        Conn_Pop1_Pop1.connectivity.weight = np.random.normal(loc=0.0, scale=0.02, size=max(1,int(NUM_HIDDEN * NUM_HIDDEN * args.sparsity)))
        Conn_Pop1_Pop1.connectivity.delay = dist[Conn_Pop1_Pop1.connectivity.pre_ind, Conn_Pop1_Pop1.connectivity.post_ind].flatten()

    Conn_Pop1_Pop2 = Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
max_example_timesteps = int(np.ceil(latest_spike_time / DT))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda_upper=args.k_reg, reg_lambda_lower=args.k_reg, 
                                reg_nu_upper=1, max_spikes=1500, delay_learn_conns=[Conn_Pop1_Pop1],
                                pops_and_conns=([hidden],[[Conn_Pop1_Pop1]],[[Conn_Pop1_Pop1]]),
                                conns_and_pops=([Conn_Pop1_Pop1],[hidden],[hidden]),
                                optimiser=Adam(0.001), x_optimiser=Adam(args.pos_lr), y_optimiser=Adam(args.pos_lr), z_optimiser=Adam(args.pos_lr),
                                pos_init = pos_dic,
                                batch_size=BATCH_SIZE, rng_seed=args.seed)

model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)
results_dic = {}

with compiled_net:
    # Loop through epochs
    start_time = perf_counter()
    callbacks = [SpikeRecorder(hidden, key="hidden_spikes", record_counts=True)]
    validation_callbacks = []
    best_e, best_acc = 0, 0
    early_stop = 15
    for e in range(NUM_EPOCHS):
        # Apply augmentation to events and preprocess
        spikes_train = []
        labels_train = []
        for events, label in raw_dataset:
            spikes_train.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                    dataset.sensor_size))
            labels_train.append(label)


        #print("sparsities: ", np.count_nonzero(np.abs(g_view_0)<np.finfo(np.float32).eps)/int(g_view_0.size), np.count_nonzero(np.abs(g_view_1)<np.finfo(np.float32).eps)/int(g_view_1.size), np.count_nonzero(np.abs(g_view_2)<np.finfo(np.float32).eps)/int(g_view_2.size))
        # Train epoch
        train_metrics, valid_metrics, train_cb, valid_cb  = compiled_net.train({input: spikes_train},
                                            {output: labels_train},
                                            start_epoch=e, num_epochs=1, 
                                            shuffle=True, callbacks=callbacks, validation_callbacks=validation_callbacks, validation_x={input: spikes_test}, validation_y={output: labels_test})


        
        hidden_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in train_cb['hidden_spikes']:
            hidden_spikes += cb_d

        
        _Conn_Pop0_Pop1 = compiled_net.connection_populations[Conn_Pop0_Pop1]
        _Conn_Pop0_Pop1.vars["g"].pull_from_device()
        g_view = _Conn_Pop0_Pop1.vars["g"].view.reshape((num_input, NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        _Conn_Pop0_Pop1.vars["g"].push_to_device()



        if train_metrics[output].result > best_acc:
            best_acc = train_metrics[output].result
            results_dic["train_acc"] = str(best_acc)
            results_dic["val_acc"] = str(valid_metrics[output].result)
            _Conn_Pop1_Pop1 = compiled_net.connection_populations[Conn_Pop1_Pop1]
            results_dic["min_delay"] = str(_Conn_Pop1_Pop1.vars["d"].values.min())
            results_dic["max_delay"] = str(_Conn_Pop1_Pop1.vars["d"].values.max())
            best_e = e
            early_stop = 15
            compiled_net.save(("best",), serialiser)
            if args.sparsity < 1.0:
                compiled_net.save_connectivity(("best",), serialiser)
            Pop1 = compiled_net.neuron_populations[hidden]
            np.save("checkpoints_space" + unique_suffix + "/XPos.npy", Pop1.vars["XPos"].values)
            np.save("checkpoints_space" + unique_suffix + "/YPos.npy", Pop1.vars["YPos"].values)
            np.save("checkpoints_space" + unique_suffix + "/ZPos.npy", Pop1.vars["ZPos"].values)
        else:
            early_stop -= 1
            if early_stop < 0:
                break


with open(f"results_pos/acc_pos_{unique_suffix}.json", 'w') as f:
    json.dump(results_dic, f, indent=4)
