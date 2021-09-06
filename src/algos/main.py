import copy
import os
import csv
import gym
from src.utils.params import *
from src.utils.constants import Constants
from src.utils.network_reader import read_network
from src.utils.utils import read_agent_config_file, read_traces, TensorboardCallback, ProgressBarManager, \
    StopTrainingOnMaxEpisodes
import torch as th
import numpy as np
from stable_baselines3 import PPO, A2C

from src.algos.baselines import random_baseline, fcfs_baseline


from stable_baselines3.common.monitor import Monitor


def make_env(name, trace_file, results_folder, link_objects, links_map):
    """
    Function to create the gym environment. Also creates a writer for writing the result file
    """
    # Create the metrics directory if it does not exist
    basename = os.path.basename(trace_file.trace_file).split('.txt')[0]
    metrics_filename = f"{results_folder}/metrics/{basename}.csv"
    os.makedirs(os.path.dirname(metrics_filename), exist_ok=True)
    metrics_stream = open(metrics_filename, 'a+', newline='')
    metrics_writer = csv.writer(metrics_stream)
    metrics_output_header = ['episode', 'mean_reward', 'avg_cct']
    metrics_writer.writerow(metrics_output_header)
    env = Monitor(env=gym.make(name), filename=results_folder)
    env.reset(trace_file=trace_file, results_folder=results_folder, link_objects=link_objects, links_map=links_map,
              metrics_writer=metrics_writer)
    return env, metrics_stream


def create_or_load_agent(env, agent_config, results_folder, load_agent=False, model_name=""):
    """
    Create an agent based on  RL algorithm PPO
    """

    if load_agent and os.path.isfile(model_name):
        agent = PPO.load(path=model_name, env=env)
    else:
        # policy for agent
        policy_kwargs = dict(net_arch=[dict(pi=agent_config['pi_nn'], vf=agent_config['vf_nn'])],
                             activation_fn=th.nn.Tanh)
        agent = PPO(policy="MlpPolicy", env=env, learning_rate=agent_config['learning_rate'],
                    batch_size=agent_config['batch_size'], gamma=agent_config['gamma'],
                    gae_lambda=agent_config['gae_lambda'], ent_coef=agent_config['ent_coef'],
                    vf_coef=agent_config['vf_coef'], max_grad_norm=agent_config['max_grad_norm'],
                    verbose=agent_config['verbose'], seed=args.seed, policy_kwargs=policy_kwargs,
                    tensorboard_log=results_folder)

    return agent


def main():
    # Setting the results directory based on the seed
    args.results_folder = "{}-{}".format(args.results_folder, args.seed)
    # Reading the network file and creating the paths/links between all pairs of nodes in the network
    networkx, link_objects, links_map = read_network(args.net_topo_file)
    # creating a trace object per input train and test files
    train_traces, test_traces = read_traces(args.train_traces, args.test_traces)
    # hyper-parameters for the agent provided in the agent config file
    agent_config = read_agent_config_file(args.agent_config_file)
    # All the provided train and test traces should have the same number of flows in them
    # This is required for the observation space, as it has to be constant throughout the run
    Constants.NUM_FLOWS_PER_TRACE = train_traces[0].num_flows

    # Setting the results directories
    train_results_dir_path = "{}/train/".format(args.results_folder)
    test_results_dir_path = "{}/test/".format(args.results_folder)
    random_baseline_dir_path = "{}/baselines/random".format(args.results_folder)
    fcfs_baseline_dir_path = "{}/baselines/fcfs".format(args.results_folder)
    env_name = "Jcsr-v0"

    # model name
    if args.model_file != "":
        model_name = args.model_file
    else:
        model_name = Constants.DEFAULT_MODEL_NAME + "_" + "PPO"
    model_name += ".zip"

    # callback to plot average episode reward on tensorboard
    mean_ep_reward_callback = TensorboardCallback(train_results_dir_path, model_name)

    # Running the RL algo with both training and testing
    if args.algo == 0 or args.algo == 1:
        # Training
        with ProgressBarManager(args.eps * len(train_traces)) as pb_callback:

            # callback to stop training when the model reaches max episodes
            max_eps_callback = StopTrainingOnMaxEpisodes(max_episodes=args.eps, verbose=1,
                                                         callback_on_new_best=pb_callback)

            # Train on all train traces one after the other
            for i in range(0, len(train_traces)):
                print(f"Training on trace: {train_traces[i].trace_file} for {args.eps} episodes ..")
                link_objects_copy = copy.deepcopy(link_objects)

                env, metrics_stream = make_env(env_name, train_traces[i], train_results_dir_path,
                                               link_objects_copy, links_map)

                # create or load model to resume training on new traces
                agent = create_or_load_agent(env, agent_config=agent_config, results_folder=args.results_folder,
                                             load_agent=True, model_name=os.path.join(train_results_dir_path,
                                                                                      model_name))

                # reset num of eps to 0 for new traces
                max_eps_callback.reset()

                # update progress bar
                pb_callback.update_nc_eps(i * args.eps)

                # total_timesteps is set large to let the agent terminate training on completing episodes
                agent.learn(total_timesteps=args.tts, callback=[max_eps_callback, mean_ep_reward_callback])

                # close the metrics stream
                metrics_stream.close()

                # release env resources
                env.close()

                # Save the model.
                agent.save(os.path.join(train_results_dir_path, model_name))

        # Testing

        for j in range(len(test_traces)):
            print(f"Testing on trace: {test_traces[j].trace_file} for 1 episodes ..")
            link_objects_copy = copy.deepcopy(link_objects)
            test_trace_copy = copy.deepcopy(test_traces[j])

            env, metrics_stream = make_env(env_name, test_trace_copy, test_results_dir_path,
                                           link_objects_copy, links_map)

            # create or load model to resume training on new traces
            agent = create_or_load_agent(env, agent_config=agent_config, results_folder=args.results_folder,
                                         load_agent=True, model_name=os.path.join(train_results_dir_path, model_name))

            obs = env.reset()
            done = False
            while not done:
                action, _states = agent.predict(obs)
                obs, reward, done, info = env.step(action)

            metrics_stream.close()
            env.close()

    # Testing Baseline Random
    if args.algo == 0 or args.algo == 2:
        for j in range(len(test_traces)):
            print(f"Running Baseline:random on trace: {test_traces[j].trace_file} for 1 episodes ..")
            link_objects_copy = copy.deepcopy(link_objects)
            test_trace_copy = copy.deepcopy(test_traces[j])
            random_baseline(test_trace_copy, link_objects_copy, links_map, random_baseline_dir_path)

    # Testing Baseline FCFS
    if args.algo == 0 or args.algo == 3:
        for j in range(len(test_traces)):
            print(f"Running Baseline:FCFS on trace: {test_traces[j].trace_file} for 1 episodes ..")
            link_objects_copy = copy.deepcopy(link_objects)
            test_trace_copy = copy.deepcopy(test_traces[j])
            fcfs_baseline(test_trace_copy, link_objects_copy, links_map, fcfs_baseline_dir_path)


if __name__ == "__main__":
    main()
