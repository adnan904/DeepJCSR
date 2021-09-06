import yaml
import os
import numpy as np
from tqdm import tqdm
from typing import Optional
from src.envs.jcsr.parsers.trace_parser import Trace
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EventCallback


def get_avg_cct(coflows):
    """
    Function to calculate the average coflow completion time based on the provided list of finished coflows
    Return: List of completion times of each coflow in the provided list
    """
    ccts = []
    for coflow in coflows:
        max_finish_time = max([f.actual_finish_time for f in coflow.flows])
        ccts.append(max_finish_time - coflow.arrival_time)
    return ccts


def read_agent_config_file(file_path):
    """
    Function to read the provided agent config. yaml file.
    Return: a dict with the config parameters
    """
    with open(file_path, 'r') as stream:
        config_yaml = yaml.load(stream, Loader=yaml.Loader)
    return config_yaml


def read_traces(train_traces_config, test_traces_config):
    train_traces = []
    test_traces = []
    with open(train_traces_config, 'r+') as f1:
        for trace_path in f1.readlines():
            trace_object = Trace(trace_path.rstrip())
            train_traces.append(trace_object)
    with open(test_traces_config, 'r+') as f2:
        for trace_path in f2.readlines():
            trace_object = Trace(trace_path.rstrip())
            test_traces.append(trace_object)

    return train_traces, test_traces


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values(reward) in tensorboard.
    """

    def __init__(self, log_dir, model_name, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, model_name)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward
            mean_reward = np.mean(y)
            self.logger.record('train/mean_ep_reward', mean_reward)
        return True


# Inspired from
# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/4_callbacks_hyperparameter_tuning.ipynb
class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        self.is_tb_set = False
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar
        self.nc_eps = 0

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.locals.get('n_episodes') + self.nc_eps
        self._pbar.update(0)

    def update_nc_eps(self, eps):
        self.nc_eps = eps


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, max_episodes):
        # init object with maximum episodes
        self.pbar = None
        self.pb_cb = None
        self.max_episodes = max_episodes

    def __enter__(self):
        # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.max_episodes)
        self.pb_cb = ProgressBarCallback(self.pbar)

        return self.pb_cb

    def __exit__(self, exc_type, exc_val, exc_tb):
        # close the callback
        self.pbar.n = self.max_episodes
        self.pbar.update(0)
        self.pbar.close()

    def update_nc_eps(self, eps):
        """
        Update number of complete episodes for progressbar
        """
        self.pb_cb.update_nc_eps(eps)


class StopTrainingOnMaxEpisodes(EventCallback):
    """
    Stop the training once a maximum number of episodes are played.

    For multiple environments presumes that, the desired behavior is that the agent trains on each env for ``max_episodes``
    and in total for ``max_episodes * n_envs`` episodes.

    :param max_episodes: Maximum number of episodes to stop training.
    :param verbose: Select whether to print information about when training ended by reaching ``max_episodes``
    """

    def __init__(self, max_episodes: int, callback_on_new_best: Optional[BaseCallback] = None, \
        verbose: int = 0):
        super(StopTrainingOnMaxEpisodes, self).__init__(callback_on_new_best, verbose=verbose)
        self.max_episodes = max_episodes
        self._total_max_episodes = max_episodes
        self.n_episodes = 0

    def _init_callback(self) -> None:
        # At start set total max according to number of envirnments
        self._total_max_episodes = self.max_episodes * self.training_env.num_envs

    def _on_step(self) -> bool:
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        self.n_episodes += np.sum(done_array).item()

        continue_training = self.n_episodes < self._total_max_episodes

        if self.verbose > 0 and not continue_training:
            mean_episodes_per_env = self.n_episodes / self.training_env.num_envs
            mean_ep_str = (
                f"with an average of {mean_episodes_per_env:.2f} episodes per env" if self.training_env.num_envs > 1 else ""
            )

            print(
                f"Stopping training with a total of {self.num_timesteps} steps because the "
                f"{self.locals.get('tb_log_name')} model reached max_episodes={self.max_episodes}, "
                f"by playing for {self.n_episodes} episodes "
                f"{mean_ep_str}"
            )

        # Trigger callback (to update progress bar)
        if self.callback is not None:
            self.locals['n_episodes'] = self.n_episodes
            self._on_event()

        return continue_training

    def reset(self) -> None:
        self.n_episodes = 0
