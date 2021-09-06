import os
import csv
import random
import numpy as np
from src.envs.jcsr.simulators.simulator import Simulator
from src.utils.utils import get_avg_cct


def random_baseline(trace_file, paths, paths_map, results_dir):
    """
    Baseline Algo. that randomly selects the overall placement and routes for each flow in the trace file
    """
    basename = os.path.basename(trace_file.trace_file).split('.txt')[0]
    metrics_filename = f"{results_dir}/metrics/{basename}.csv"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
    metrics_stream = open(metrics_filename, 'a+', newline='')
    metrics_writer = csv.writer(metrics_stream)
    metrics_output_header = ['episode', 'mean_reward', 'avg_cct']
    metrics_writer.writerow(metrics_output_header)
    simulator = Simulator(trace_file, paths, paths_map)
    done = False
    schedules = random.sample(range(trace_file.num_flows), trace_file.num_flows)
    routes = [random.randint(0, 2) for x in range(trace_file.num_flows)]
    rewards = []
    i = 0
    while not done:
        action = [schedules[i], routes[i]]
        done, reward, repeated_action, processed_flow_ids = simulator.step(action)
        rewards.append(reward)
        i += 1
    ep_ccts = get_avg_cct(simulator.trace_file.coflows)
    ep_avg_cct = np.mean(ep_ccts)

    metrics_writer.writerow([1, np.mean(rewards), ep_avg_cct])
    metrics_stream.close()


def fcfs_baseline(trace_file, paths, paths_map, results_dir):
    """
    The Baseline algo that schedules the flows based on the arrival times. Its selects the route based on equal cost
    """
    basename = os.path.basename(trace_file.trace_file).split('.txt')[0]
    metrics_filename = f"{results_dir}/metrics/{basename}.csv"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
    metrics_stream = open(metrics_filename, 'a+', newline='')
    metrics_writer = csv.writer(metrics_stream)
    metrics_output_header = ['episode', 'mean_reward', 'avg_cct']
    metrics_writer.writerow(metrics_output_header)
    simulator = Simulator(trace_file, paths, paths_map)
    done = False
    schedules = [x for x in range(trace_file.num_flows)]
    routes = [2 for x in range(trace_file.num_flows)]
    rewards = []
    i = 0
    while not done:
        action = [schedules[i], routes[i]]
        done, reward, repeated_action, processed_flow_ids = simulator.step(action)
        rewards.append(reward)
        i += 1
    ep_ccts = get_avg_cct(simulator.trace_file.coflows)
    ep_avg_cct = np.mean(ep_ccts)

    metrics_writer.writerow([1, np.mean(rewards), ep_avg_cct])
    metrics_stream.close()