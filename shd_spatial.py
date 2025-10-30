import numpy as np

from hashlib import md5

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, Checkpoint
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
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
import copy


parser = ArgumentParser()
parser.add_argument("--num_hidden", type=int, default=128, help="Number of hidden neurons")
parser.add_argument("--num_dim", type=int, default=2, help="Number of dimensions")
parser.add_argument("--k_reg", type=float, default=5e-11, help="Spike regularisation strength")
parser.add_argument("--delay_lr", type=float, default=0.1, help="Learning rate for the R")
parser.add_argument("--max_delay", type=int, default=62, help="Maximum possible delay")
parser.add_argument("--l1", type=float, default=0.0, help="L1 regularisation strength")
parser.add_argument("--dist_lambda", type=int, default=1, help="Distance cost")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

NUM_HIDDEN = args.num_hidden
BATCH_SIZE = 256
NUM_EPOCHS = 300
DT = 1.0



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
        if self._optimiser.alpha < 0.001 :
            self._optimiser.alpha = (self._optimiser.alpha) * (1.05 ** batch)
        else:
            self._optimiser.alpha = 0.001


class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)
        
        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy

class Blend:
    def __init__(self, p_blend, sensor_size):
        self.p_blend = p_blend
        self.n_blend = 7644
        self.sensor_size = sensor_size
    def __call__(self, dataset: list, classes: list) -> list:
        
        for i in range(self.n_blend):
            idx = np.random.randint(0,len(dataset))
            idx2 = np.random.randint(0,len(classes[dataset[idx][1]]))
            assert dataset[idx][1] == dataset[classes[dataset[idx][1]][idx2]][1]
            dataset.append((self.blend(dataset[idx][0], dataset[classes[dataset[idx][1]][idx2]][0]), dataset[idx][1]))
            
        return dataset

    def blend(self, X1, X2):
        X1 = copy.deepcopy(X1)
        X2 = copy.deepcopy(X2)
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)
        X1 = np.delete(
            X1,
            np.where(
                (X1["x"] < 0) | (X1["x"] >= self.sensor_size[0]) | (X1["t"] < 0) | (X1["t"] >= 1000000)))
        X2 = np.delete(
            X2,
            np.where(
                (X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) | (X2["t"] < 0) | (X2["t"] >= 1000000)))
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < self.p_blend
        X1_X2 = np.concatenate([X1[mask1], X2[mask2]])
        idx= np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2




dataset = SHD(save_to="../data", train=True)
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)
shift = Shift(40, dataset.sensor_size)
blend = Blend(0.5, dataset.sensor_size)


max_spikes = 0
latest_spike_time = 0
raw_dataset = []
classes = [[] for _ in range(20)]
for i, data in enumerate(dataset):
    events, label = data
    events = np.delete(events, np.where(events["t"] >= 1000000))
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



serialiser = Numpy("checkpoints_space_cartesian_limit_" + unique_suffix)
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

    if args.num_dim > 0:
        positons = tuple(np.random.uniform(low=-1.0, high=1.0, size=NUM_HIDDEN) for i in range(args.num_dim))

        pos_dic = {}
        pos_dic[hidden] = positons

        pos_dic["max_delay"] = args.max_delay


        points = np.column_stack(positons)
        dist = cdist(points, points)
    elif args.delay_lr>0:
        dist = np.zeros((NUM_HIDDEN, NUM_HIDDEN))
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    if args.delay_lr > 0:
        Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02), dist),
                Exponential(5.0), max_delay_steps=args.max_delay)
    else:
        Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
                Exponential(5.0))
        

    Conn_Pop1_Pop2 = Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
max_example_timesteps = int(np.ceil(latest_spike_time / DT))
if args.num_dim > 0:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                    losses="sparse_categorical_crossentropy",
                                    reg_lambda_upper=args.k_reg, reg_lambda_lower=args.k_reg, 
                                    reg_nu_upper=1, max_spikes=1500, delay_learn_conns=[Conn_Pop1_Pop1],
                                    pops_and_conns_and_pops=([hidden],[[(Conn_Pop1_Pop1, hidden)]],[[(Conn_Pop1_Pop1,hidden)]]),
                                    conns_and_pops=([Conn_Pop1_Pop1],[hidden],[hidden]),
                                    optimiser=Adam(0.001), pos_optimiser=Adam(args.delay_lr),
                                    pos_init = pos_dic,
                                    reg_conns=[Conn_Pop1_Pop1], l1_reg_lambda=args.l1, dist_lambda=args.dist_lambda,
                                    batch_size=BATCH_SIZE, rng_seed=args.seed)
else:
    compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                    losses="sparse_categorical_crossentropy",
                                    reg_lambda_upper=args.k_reg, reg_lambda_lower=args.k_reg, 
                                    reg_nu_upper=1, max_spikes=1500, delay_learn_conns=[Conn_Pop1_Pop1],
                                    optimiser=Adam(0.001), delay_optimiser=Adam(args.delay_lr),
                                    reg_conns=[Conn_Pop1_Pop1], l1_reg_lambda=args.l1, dist_lambda=args.dist_lambda,
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
        blended_dataset = blend(copy.deepcopy(raw_dataset), classes)
        for events, label in blended_dataset:
            spikes_train.append(preprocess_tonic_spikes(shift(events), dataset.ordering,
                                                    dataset.sensor_size))
            labels_train.append(label)


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


        compiled_net.save((e,), serialiser)
        Pop1 = compiled_net.neuron_populations[hidden]
        for i in range(args.num_dim):
            np.save("checkpoints_space_cartesian_limit_" + unique_suffix + "/" + str(e) + "-Pos" + str(i) + ".npy", Pop1.vars["Pos"+str(i)].values)
        if train_metrics[output].result > best_acc:
            best_acc = train_metrics[output].result
            results_dic["train_acc"] = str(best_acc)
            results_dic["val_acc"] = str(valid_metrics[output].result)
            if args.delay_lr > 0:
                _Conn_Pop1_Pop1 = compiled_net.connection_populations[Conn_Pop1_Pop1]
                results_dic["min_delay"] = str(_Conn_Pop1_Pop1.vars["d"].values.min())
                results_dic["max_delay"] = str(_Conn_Pop1_Pop1.vars["d"].values.max())
            best_e = e
            early_stop = 15
        else:
            early_stop -= 1
            if early_stop < 0:
                break


with open(f"results_pos/acc_pos_cartesian_limit_{unique_suffix}.json", 'w') as f:
    json.dump(results_dic, f, indent=4)

#network.load(("best",), serialiser)
network.load((best_e,), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:
    metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test})
    conn = compiled_net.connection_populations[Conn_Pop1_Pop1]
    conn.vars["g"].pull_from_device()
    thresh = np.quantile(np.abs(conn.vars["g"].values), q=0.9)
    weight = conn.vars["g"].values
    weight[np.abs(weight) < thresh] = 0
    conn.vars["g"].values = weight
    conn.vars["g"].push_to_device()
    metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test})
    results_dic["magnitude_pruned_acc"] = str(metrics[output].result)

    with open(f"results_pos/acc_pos_cartesian_limit_{unique_suffix}.json", 'w') as f:
        json.dump(results_dic, f, indent=4)
    
#network.load(("best",), serialiser)
network.load((best_e,), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)

with compiled_net:

    conn = compiled_net.connection_populations[Conn_Pop1_Pop1]
    conn.vars["g"].pull_from_device()
    conn.vars["d"].pull_from_device()
    thresh = np.quantile(conn.vars["d"].values, q=0.1)
    weight = conn.vars["g"].values
    weight[conn.vars["d"].values > thresh] = 0
    conn.vars["g"].values = weight
    conn.vars["g"].push_to_device()
    metrics, _  = compiled_net.evaluate({input: spikes_test},
                                        {output: labels_test})
    results_dic["length_pruned_acc"] = str(metrics[output].result)


    with open(f"results_pos/acc_pos_cartesian_limit_{unique_suffix}.json", 'w') as f:
        json.dump(results_dic, f, indent=4)
