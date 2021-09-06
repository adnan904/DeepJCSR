# Joint Scheduling and Routing of Coflows Using Deep Reinforcement Learning

Coordinated Coflow Scheduling and Routing using reinforcement learning.


## Setup

You need to have [Python 3.6](https://www.python.org/downloads/release/)+ and [venv](https://docs.python.org/3/library/venv.html) module installed.

### Create the venv

On your local machine:

```bash
# create venv once
python3 -m venv ./venv
# activate the venv (always)
source venv/bin/activate
```

### Install dependencies
Tested on Ubuntu 20.04 and Python 3.7
_Recommended for development_: 

```
sudo apt update
sudo apt install libopenmpi-dev libsm6 libxext6 python-dev python3-dev
```

Check your Python 3 version and ensure you have Python 3.6+; otherwise install them:

```
# check version
python --version

# if not 3.6+, install python 3.6+
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.6 python3.6-dev

```

To install the RL dependencies(after activating the venv):

```bash
pip install -r requirements.txt
```

## Use the RL agent

From the main directory of the repo and with the venv activated:
```bash
python -m src.algos.main
```

By default it requires 3 main config files:

1) `res/config/train-traces.txt` : containing the paths to the train traces
2) `res/config/test-traces.txt`: containing the paths to the train traces
3) `res/config/PPO_config.yaml`: containing the hyperparameters for the PPO algo.

_Important_: 

1) The defaults for these have been already set. You can use different files using additional parameters as shown below.
2) Also make sure you use the train and test traces having the same number of flows. This limitation is because the agent's observation space is dependent on the number of flows within a trace file.
3) Also for the network file you use, please only use the traces provided. This is required because each network has a fixed number of nodes. And the trace files have been created such that they only have that many input sources and destinations.

Additional Parameters:

```bash

Options:
  --seed     INTEGER     Specify the random seed for the   environment and the learning agent.

  --algo     INTEGER     Specify the algo you wanna run on the specified train and test traces. 0 - all algos, 1 - RL, 2 - Random, 3 - FCFS

  -ac, --agent-config   TEXT   the location for the RL agent config file in yaml format

  --train_traces  TEXT   Location to the train config file containing the paths to the train traces, one line for one trace

  --test_traces  TEXT   Location to the test config file containing the paths to the test traces, one line for one trace

  --net_topo_file TEXT Location to the network file in graphml format

  --results_folder TEXT Location for saving results, logs, model

  --eps     INTEGER    Number of episodes for each trace file
```

### Training and testing

Example for training over 10 episodes and then testing:

```bash
python -m src.algos.main --algo 1 --eps 10
```

Results are stored under `results/` according to the input arguments and the current time stamp.
There, you'll find copies of the used inputs, the trained weights, logs, and all result files of any test runs that you performed.


## Generating Traces:
- First thing to note is how many nodes there are in the network topology file. E.g. Abilene network has 11 nodes.
- Use [workload-generator](https://github.com/sincronia-coflow/workload-generator) to produce the coflow files. Change the [num_ports](https://github.com/sincronia-coflow/workload-generator/blob/master/distribution_producer.py#L5) to the number of nodes in the network. Use either Uniform or zipf distribution parameters. Since with FB-UP the num of input ports has to be 150.
```bash
python trace_producer.py 1000 20 0.9 0.5 Z Z
```
This will produce a trace file with 1000 coflows, network load of 0.9 and zipf distrinution.

_Note_: Some of these files are already provided in `res\traces\coflows`
- Use the `flow-generator.py` from `res\traces` to have the final flow level trace files required for this project. It requires as input the file produced in step 2.
```bash 
python -m res.traces.flow-generator --tf 'res/traces/1000-0.1-20-50.0-Z-Z-100.txt' --nf 50
```
This will produce a trace file with 50 flows.

_Note_: Some of these files are provided within `res\traces\flows`


## Visualizing/Analyzing Results

To get a better understanding of what the agent is doing, there is a Juypter notebook that is provided in the repository. To view the sample results, open and run the `evaluation.ipynb` Jupyter Notebook:

```bash
# first time installation
pip install jupyter
# run jupyter server
jupyter notebook
```

It requires 3 directories containing the result files from the 3 algos: RL, random, and fcfs.