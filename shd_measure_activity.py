import numpy as np

from ml_genn import Connection, Network, Population
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD
from ml_genn.callbacks import SpikeRecorder

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from ml_genn.compilers.event_prop_compiler import default_params

import scipy.spatial.distance

import bct

import copy


NUM_HIDDEN = 128

DT = 1.0

# Get SHD dataset
dataset = SHD(save_to='../data', train=False)
# Preprocess
spikes = []
labels = []

times, ids = [], []
for i in range(len(dataset)):
    events, label = dataset[i]
    times.append([event[0] for event in events])
    ids.append([event[1] for event in events])
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                        dataset.sensor_size))
    labels.append(label)
np.savez_compressed("checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14/input_spikes_times.npz", *[np.array(l) for l in times])
np.savez_compressed("checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14/input_spikes_ids.npz", *[np.array(l) for l in ids])
BATCH_SIZE = 283

# Determine max spikes and latest spike time
max_spikes = 14917#calc_max_spikes(spikes)
latest_spike_time = 1369.14#calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

serialiser = Numpy("checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    
    # Connections
    Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    rec_conn = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02), np.zeros((NUM_HIDDEN, NUM_HIDDEN))),
               Exponential(5.0), max_delay_steps=1000)
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))
    
    
max_example_timesteps = int(np.ceil(latest_spike_time / DT))



# Load network state from final checkpoint
network.load(("299",), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                                reset_in_syn_between_batches=True,
                                batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network)


with compiled_net:
    callbacks = [SpikeRecorder(hidden, key="hidden_spikes"), "batch_progress_bar"]
    conn = compiled_net.connection_populations[rec_conn]
    metrics, test_cb  = compiled_net.evaluate({input: spikes},
                                        {output: labels}, callbacks = callbacks)
    
    np.savez_compressed("checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14/hidden_spikes_times.npz", *[np.array(l) for l in test_cb['hidden_spikes'][0]])
    np.savez_compressed("checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_14/hidden_spikes_ids.npz", *[np.array(l) for l in test_cb['hidden_spikes'][1]])

    print(f"Accuracy = {100 * metrics[output].result}%")
