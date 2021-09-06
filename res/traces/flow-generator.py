import argparse
import os
import random
from operator import attrgetter

CURRENT_WORKING_DIR = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Flow:
    def __init__(self, id, cid, arrival_time, src, dst, size):
        self.id = id
        self.cid = cid
        self.arrival_time = arrival_time
        self.src = src
        self.dst = dst
        self.size = size   # in MBs


class ProcTraces:
    """
      1)  Reads coflow traces generated using https://git.cs.uni-paderborn.de/cn_upb/workload-generator.git
      2) Creates a flow level trace file with a specified number of flows
      
      The created file is of the following format:
            Line1 : <Num_Ports> - <Num_Coflows> - <Num_Flows>
            Num_Flows lines below: <Flow_id> - <Arrival-time> - <Coflow-id> - <Source> - <Destination> - <Size>
    """

    def __init__(self, source_path, num_flows, res_folder, seed):
        self.source_trace_path = source_path
        self.req_num_flows = num_flows
        self.results_dir = res_folder
        self.total_num_coflows = 0
        self.total_num_ports = 0
        self.total_num_flows_in_coflow_file = 0
        self.total_flows = []
        self.selected_flows = []
        self.seed = seed

    def read_coflow_trace(self):

        with open(self.source_trace_path, 'r+') as f:
            flow_id = 0
            for i, line in enumerate(f.readlines()):
                line = line.rstrip().split(' ')
                if i==0:
                    self.total_num_ports = line[0]
                    self.total_num_coflows = line[1]
                else:
                    cid = line[0]
                    arrival_time = int(line[1])
                    start_index = 5
                    self.total_num_flows_in_coflow_file += int(line[2])
                    for _ in range(int(line[2])):
                        src = int(line[start_index])
                        dst = int(line[start_index+1])
                        size = int(line[start_index+2])
                        start_index += 3
                        flow = Flow(flow_id, cid, arrival_time, src, dst, size)
                        self.total_flows.append(flow)
                        flow_id += 1

    def gen_flow_trace(self):
        flows = []
        num_coflows = 0
        if self.req_num_flows > self.total_num_flows_in_coflow_file:
            print(f'The trace file provided has only {self.total_num_flows_in_coflow_file} flows. So only generating '
                  f'{self.total_num_flows_in_coflow_file} flows. ')
            flows = self.total_flows
            num_coflows = self.total_num_coflows
        else:
            selected_flows_indexes = random.sample(range(self.total_num_flows_in_coflow_file), self.req_num_flows)
            for i in selected_flows_indexes:
                flows.append(self.total_flows[i])
            flows.sort(key=attrgetter('arrival_time'))
            flow_id = 0
            num_coflows_set = set()
            for flow in flows:
                flow.id = flow_id
                num_coflows_set.add(flow.cid)
                flow_id += 1
            num_coflows = len(num_coflows_set)

        # setting the result file path
        basename = os.path.basename(self.source_trace_path)
        name = basename.split('-')
        filename = f'{len(flows)}-{name[1]}-{name[2]}-{name[3]}-{name[4]}-{name[5]}-{self.seed}.txt'
        os.makedirs(f'{CURRENT_WORKING_DIR}/{self.results_dir}', exist_ok=True)
        res_path = f'{CURRENT_WORKING_DIR}/{self.results_dir}/{filename}'
        self.write_to_file(flows, num_coflows, res_path)

    def write_to_file(self, flows, num_coflows, results_path):
        with open(results_path, 'w+') as file:
            header = f'{self.total_num_ports} {num_coflows} {len(flows)}\n'
            file.write(header)
            for flow in flows:
                line = f'{flow.id} {flow.arrival_time} {flow.cid} {flow.src} {flow.dst} ' \
                       f'{flow.size}\n'
                file.write(line)
        print(f"File Saved at: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to generate flows from a given file containing coflows. '
                                                 'Usage: python3 -m res.traces.flow-generator'
                                                 '               --tf res/traces/1000-0.1-20-50.0-Z-Z-100.txt'
                                                 '               --nf 50')
    parser.add_argument("--tf", dest="trace_file", type=str,
                        required=True,
                        help="Provide path to the file containing coflows!")
    parser.add_argument("--nf", dest='num_flows', type=int,
                        default=f"{CURRENT_WORKING_DIR}", required=True,
                        help="Provide number of flows to generate")
    parser.add_argument("--seed", dest="seed", type=int,
                        default=100,
                        help="Provide path to the file containing coflows!")
    parser.add_argument("--res", dest='res_path', type=str,
                        default="traces",
                        help="Provide folder path for saving results file")
    args = parser.parse_args()
    proc = ProcTraces(args.trace_file, args.num_flows, args.res_path, args.seed)
    proc.read_coflow_trace()
    proc.gen_flow_trace()

