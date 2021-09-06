
class Simulator:
    def __init__(self, trace_file, link_objects, links_map,  debug=False):
        self.trace_file = trace_file
        self.link_objects = link_objects
        self.links_map = links_map
        self.current_time = 0
        self.num_finished_flows = 0
        self.total_flows = trace_file.num_flows
        self.processed_flow_ids = []
        self.active_flows = []
        self.debug = debug
        self.done = False

    def process_finished_flows(self):
            if self.active_flows:
                for flow in self.active_flows:
                    if self.current_time > flow.actual_finish_time:
                        flow.link.num_active_flows -= 1
                        flow.link.total_time_busy_for -= (flow.duration + flow.link.delay)
                        if flow.link.num_active_flows <= 0:
                            flow.link.reset()
                        flow.link = None
                        flow.is_active = False
                        flow.is_finished = True
                        self.active_flows.remove(flow)

    def step(self, action):
        """
        Perform simulation step
        Args:
                action: action taken by scheduling agent

        Admit/reject action at the beginning of event and do necessary processing on flow/coflows.
        """
        self.process_finished_flows()

        repeated_action = False
        if action[0] in self.processed_flow_ids:
            reward = 0
            repeated_action = True
            return self.done, reward, repeated_action, self.processed_flow_ids

        flow = self.trace_file.flows[action[0]]
        flow.is_active = True
        if flow.arrival_time > self.current_time:
            self.current_time = flow.arrival_time
        link_pos = self.links_map[flow.src][flow.dest][action[1]]
        chosen_path = self.link_objects[link_pos]
        shortest_link = self.links_map[flow.src][flow.dest][0]
        shortest_path = self.link_objects[shortest_link]
        shortest_delay = shortest_path.delay
        flow.expected_finish_time = flow.arrival_time + flow.duration + shortest_delay
        if chosen_path.is_available:
            if flow.arrival_time < self.current_time:
                flow.start_time = flow.arrival_time
            else:
                flow.start_time = self.current_time
            flow.actual_finish_time = flow.start_time + flow.duration + chosen_path.delay
            chosen_path.is_available = False
        else:
            if flow.arrival_time < chosen_path.next_available_time:
                flow.start_time = chosen_path.next_available_time
            else:
                flow.start_time = flow.arrival_time
            flow.actual_finish_time = flow.start_time + flow.duration + chosen_path.delay

        chosen_path.next_available_time = flow.actual_finish_time + 1
        chosen_path.num_active_flows += 1
        chosen_path.total_time_busy_for += (flow.duration + chosen_path.delay)
        flow.link = chosen_path
        self.active_flows.append(flow)
        self.processed_flow_ids.append(flow.id)
        self.num_finished_flows += 1

        reward = flow.expected_finish_time / flow.actual_finish_time

        if self.num_finished_flows == self.total_flows:
            self.done = True

        self.process_finished_flows()

        return self.done, reward, repeated_action, self.processed_flow_ids

    def reset(self):
        """
        Reset
        """

        self.current_time = 0
        self.num_finished_flows = 0
        self.processed_flow_ids = []
        self.active_flows = []
        self.done = False