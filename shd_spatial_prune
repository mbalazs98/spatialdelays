import numpy as np

from hashlib import md5

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, Checkpoint
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense, FixedProbability
from ml_genn.initializers import Normal, Uniform
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
parser.add_argument("--num_hidden", type=int, default=256, help="Number of hidden neurons")
parser.add_argument("--num_dim", type=int, default=3, help="Number of dimensions")
parser.add_argument("--sparsity", type=float, default=1.0, help="Sparsity of connections")
parser.add_argument("--k_reg", type=float, default=5e-11, help="Spike regularisation strength")
parser.add_argument("--delay_lr", type=float, default=0.1, help="Learning rate for the R")
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



dataset = SHD(save_to="../data", train=False)
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

spikes_test = []
labels_test = []
for i in range(len(dataset)):
    events, label = dataset[i]
    events = np.delete(events, np.where(events["t"] >= 1000000))
    spikes_test.append(preprocess_tonic_spikes(events, dataset.ordering,
                                            dataset.sensor_size))
    labels_test.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes_test)
latest_spike_time = calc_latest_spike_time(spikes_test)



serialiser = Numpy("checkpoints_space_cartesian" + unique_suffix)
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


    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02), Uniform(0,0)),
            Exponential(5.0), max_delay_steps=int(1000))

    Conn_Pop1_Pop2 = Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
max_example_timesteps = int(np.ceil(latest_spike_time / DT))
network.load(("best",), serialiser)
compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE, rng_seed=args.seed)
model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                  if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network, name=model_name)
results_dic = {}


percentages = [95, 90, 85, 80, 75, 70]
with compiled_net:
    _Conn_Pop1_Pop1 = compiled_net.connection_populations[Conn_Pop1_Pop1]
    _Conn_Pop1_Pop1.vars["g"].pull_from_device()
    g_orig = copy.deepcopy(_Conn_Pop1_Pop1.vars["g"].view.reshape((NUM_HIDDEN, NUM_HIDDEN)))
    _Conn_Pop1_Pop1.vars["d"].pull_from_device()
    d_view = _Conn_Pop1_Pop1.vars["d"].view.reshape((NUM_HIDDEN, NUM_HIDDEN))
    for percentage in percentages:
        _Conn_Pop1_Pop1 = compiled_net.connection_populations[Conn_Pop1_Pop1]
        _Conn_Pop1_Pop1.vars["g"].pull_from_device()
        g_view = _Conn_Pop1_Pop1.vars["g"].view.reshape((NUM_HIDDEN, NUM_HIDDEN))
        g_view[:] = g_orig
        g_view[d_view > np.percentile(d_view, percentage)] = 0
        _Conn_Pop1_Pop1.vars["g"].push_to_device()
        metrics, _  = compiled_net.evaluate({input: spikes_test},
                                            {output: labels_test}, callbacks=[])

        results_dic["acc"+str(percentage)] = metrics[output].result

with open(f"results_pos/prune_pos_cartesian{unique_suffix}.json", 'w') as f:
    json.dump(results_dic, f, indent=4)
