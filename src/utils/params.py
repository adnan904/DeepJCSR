import time
import argparse
import random
from .constants import Constants

parser = argparse.ArgumentParser(description='Jcsr')
# Experiment
parser.add_argument("--seed", dest="seed", type=int, default=random.randint(1000, 9999),
                    help="Seed for reproducing results")
parser.add_argument("--algo", dest="algo", type=int, default=0,
                    help="0 - all algos, 1 - RL, 2 - random, 3 - fcfs")


# Training algorithm
parser.add_argument("-ac", "--agent-config", dest="agent_config_file", type=str, default=Constants.DEFAULT_AGENT_CONFIG,
                    help="Specifiy the location for the RL agent config file in yaml format")
parser.add_argument("--eps", type=int, default=100, help="Number of episodes")
parser.add_argument("--tts", type=int, default=1e10, help="Total timesteps before terminating the training of agent")
parser.add_argument("--num_ros", type=int, default=2, help="Number of rollouts")

# trained Model
parser.add_argument("--model_file", type=str, default="",
                    help="Provide filename for saving model")
parser.add_argument("--results_folder", type=str, default="./results/{}".format(
                    time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())),
                    help="Provide file location for saving results")


# Simulator
parser.add_argument("-t", "--net_topo_file", default=Constants.DEFAULT_TOPOLOGY_FILE,
                    help="Topology file, default is abilene.graphml")
parser.add_argument("-g", "--debug", type=bool, default=False,
                    help='show debug messages, by default False')

# trace files
parser.add_argument("--train_traces", type=str, default=Constants.DEFAULT_TRAIN_TRACES, help='Path to config file '
                                                                                             'containing paths to all '
                                                                                             'train traces')
parser.add_argument("--test_traces", type=str, default=Constants.DEFAULT_TEST_TRACES, help='Path to config file '
                                                                                           'containing paths to all '
                                                                                           'test traces')


args = parser.parse_args()
