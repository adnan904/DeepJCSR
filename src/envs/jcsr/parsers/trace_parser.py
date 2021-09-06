import os
import pandas as pd
from src.envs.jcsr.ds.coflow import Coflow
from src.envs.jcsr.ds.flow import Flow


class Trace:
    """
    Parser for traces of the following format:
        Line1 : <Num_Ports> - <Num_Coflows> - <Num_Flows>
        Num_Flows lines below: <Flow_id> - <Arrival-time> - <Coflow-id> - <Source> - <Destination> - <Size>
    The traces are generated using the flow-generator.py script within the res/traces folder of this repo
    """

    def __init__(self, trace_file):
        os.path.exists(trace_file)
        self.trace_file = trace_file
        self.coflows = []
        self.flows = []
        self.num_flows = 0
        self.num_coflows = 0
        self.num_ports = 0

        self.parse_trace()

    def parse_trace(self):
        with open(self.trace_file, 'r+') as f:
            header = f.readline().rstrip()
            header_list = header.split(' ')
            self.num_ports = int(header_list[0])
            self.num_coflows = int(header_list[1])
            self.num_flows = int(header_list[2])

        df = pd.read_csv(self.trace_file, skiprows=[0], sep=' ', names=['flow-id', 'arrival-time', 'coflow-id',
                                                                          'src', 'dest', 'size'])
        coflows_df = df.groupby('coflow-id')
        for group, frame in coflows_df:
            flows_in_coflow = []
            coflow_size = 0  # MBs
            for row in frame.itertuples(index=False, name=None):
                flow = Flow(row[0], row[1], group, row[3], row[4], row[5])
                coflow_size += row[5]
                flows_in_coflow.append(flow)
                self.flows.append(flow)
            coflow = Coflow(group, flows_in_coflow[0].arrival_time, coflow_size, flows_in_coflow)
            coflow.length = max([flow.size for flow in flows_in_coflow])
            coflow.width = len(flows_in_coflow)
            self.coflows.append(coflow)
