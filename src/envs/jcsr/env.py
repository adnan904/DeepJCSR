from gym import spaces
from gym.spaces import flatten_space, flatten
from src.envs.jcsr.simulators.simulator import Simulator
from src.utils.constants import Constants
from src.utils.utils import get_avg_cct
import warnings
import copy
import numpy as np
import gym
gym.logger.set_level(40)
warnings.simplefilter(action='ignore', category=FutureWarning)


class JcsrEnv(gym.Env):
    """
    Gym environment for the RL Agent to train and test on
    """
    def __init__(self, **kwargs):
        self.debug = kwargs.get('debug')

        self.trace_file = ""
        self.link_objects = ""
        self.links_map = ""
        self.metrics_writer = ""
        self.simulator = None
        self.current_observation = None
        self.initial_observation = None
        self.results_folder = kwargs.get('results_folder')
        self.rewards = []
        self.num_episode = 0


        self.num_flows = Constants.NUM_FLOWS_PER_TRACE

        # total data sent by an active coflow in last
        self.obs_shape = (self.num_flows, Constants.NUM_FEATURES)
        self.obs_low = np.full(shape=self.obs_shape, fill_value=0, dtype=np.float32)
        self.obs_high = np.tile([self.num_flows, np.inf, np.inf, 1.0, np.inf, 1.0, 1.0, 1.0],
                                self.num_flows).reshape(self.obs_shape)
        observation_space = spaces.Box(low=self.obs_low, high=self.obs_high,
                                            shape=self.obs_shape, dtype=np.float32)

        # Conatins 8 features for each flow in the network
        # The Features are: flow_id, arrival_time, coflow_id, is_processed, duration, path1 score, path2 score, path3
        self.observation_space = flatten_space(observation_space)

        # action space[choose one of flow to schedule, choose one of the path(src to dest) for that flow in the network]
        self.action_space = spaces.MultiDiscrete([self.num_flows, 3])

    def calc_weighted_link_score(self, flow, link_id):
        """
        Calculates a weighted score between [0,1], preferring links with low delay
        """
        link_pos = self.links_map[flow.src][flow.dest][link_id]
        path = self.link_objects[link_pos]
        if path.is_available:
            if path.delay:
                score = 1 / path.delay
            else:
                score = 1
        else:
            # When the chosen path is already busy we divided by the delay of that path + the time it would be busy for
            score = 1 / (path.delay + path.total_time_busy_for)
        return score

    def observe(self, processed_flows=None, initial_observation=False):
        """
        Observation is representation of state space e.g., link status, processed and unprocessed flows
        For the initial reset call to the environment we already create a initial observation for the whole trace file.
        For each of the consecutive time the step function is called and an action performed this initial observation is
        updated.
        """
        if initial_observation:
            ob = self.obs_low
            for flow in self.trace_file.flows:
                ob[flow.id][0] = flow.id
                ob[flow.id][1] = flow.arrival_time  # in ms
                ob[flow.id][2] = flow.cid
                ob[flow.id][3] = 0              # is_processed
                ob[flow.id][4] = flow.duration
                ob[flow.id][5] = self.calc_weighted_link_score(flow, 0)
                ob[flow.id][6] = self.calc_weighted_link_score(flow, 1)
                ob[flow.id][7] = self.calc_weighted_link_score(flow, 2)
            self.current_observation = copy.deepcopy(ob)
        else:
            ob = self.current_observation
            for flow in self.trace_file.flows:
                if flow.id in processed_flows:
                    ob[flow.id][3] = 1              # is_processed
                    ob[flow.id][4] = 0              # duration
                    ob[flow.id][5] = 0              # link1
                    ob[flow.id][6] = 0              # link2
                    ob[flow.id][7] = 0              # link3
                else:
                    ob[flow.id][5] = self.calc_weighted_link_score(flow, 0)
                    ob[flow.id][6] = self.calc_weighted_link_score(flow, 1)
                    ob[flow.id][7] = self.calc_weighted_link_score(flow, 2)

        return ob

    def step(self, action):
        """
        Perform action and move to next state
        Args:
            action: action from the scheduling agent
        Returns:
            new_state: new state
            reward: reward (or penalty) for performing given action
            done: True when new_state is terminal state
        """

        new_state = self.current_observation
        info = {}

        # perform action and move to next state
        # repeated_action tells us if the flow chosen has been already processed. In this case the reward is 0 and the
        # state space remains same.
        # processed_flows are used to update the observation space to represent the change in the links adn flows status
        done, reward, repeated_action, processed_flows = self.simulator.step(action)

        self.rewards.append(reward)
        info['reward'] = reward

        # only when its a new action is the observation space updated
        if not repeated_action:
            new_state = self.observe(processed_flows=processed_flows)
            self.current_observation = new_state

        new_state = flatten(self.observation_space, new_state)

        # To prevent the episode to get into an infinite loop because of repeated actions
        if len(self.rewards) > 50*self.num_flows:
            for flow in self.trace_file.flows:
                if not flow.is_finished and not flow.is_active:
                    flow.expected_finish_time = flow.arrival_time
                    flow.start_time = flow.arrival_time
                    flow.actual_finish_time = flow.arrival_time
                    flow.is_finished = True
            done = True

        # Once all the flows have been processed, some of them might still be in the active_flows queue because the
        # current time of the simulator has not crossed their finishing time. So doing that here and releasing the
        # links used. Also writing the episode avg. cct to the results file
        if done:
            for flow in self.simulator.active_flows:
                flow.is_active = False
                flow.is_finished = True
                flow.link.reset()
                flow.link = None
            self.simulator.active_flows = []
            self.num_episode += 1
            ep_ccts = get_avg_cct(self.trace_file.coflows)
            ep_avg_cct = np.mean(ep_ccts)

            self.metrics_writer.writerow([self.num_episode, np.mean(self.rewards), ep_avg_cct])

        return new_state, reward, done, info

    def reset(self, trace_file="", results_folder="", link_objects="", links_map="", metrics_writer=""):
        """
        Reset environment
        """

        if results_folder != "":
            self.results_folder = results_folder
        if trace_file != "":
            self.trace_file = trace_file
        if link_objects != "":
            self.link_objects = link_objects
        if links_map != "":
            self.links_map = links_map
        if metrics_writer != "":
            self.metrics_writer = metrics_writer
        if self.initial_observation is None:
            self.initial_observation = self.observe(initial_observation=True)

        # reset all the path objects
        for link in self.link_objects:
            link.reset()
        # reinitialize everything
        if self.simulator is None:
            self.simulator = Simulator(trace_file, link_objects, links_map)
        self.simulator.reset()

        for flow in self.trace_file.flows:
            flow.reset()

        self.current_observation = copy.deepcopy(self.initial_observation)

        self.rewards = []
        return flatten(self.observation_space, self.initial_observation)
