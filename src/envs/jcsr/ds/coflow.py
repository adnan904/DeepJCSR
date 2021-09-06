import sys
import numpy as np


class Coflow:
    """Simple container class for Co-flows"""

    def __init__(self, id, arrival_time, size, flows=[], duration=np.inf):
        self.id = id
        self.arrival_time = arrival_time
        self.flows = flows  # all flows in the coflow

        self.cct = np.inf		# computed coflow completion time, in milliseconds
        self.duration = duration # CCT in empty network, in milliseconds
        self.deadline = np.inf

        self.size = size * 1024			# total data size of shuffle in bytes(input is in MBs)
        self.length = 0			# size (in bytes) of the largest flow
        self.width = 0			# number of parallel flows

        self.total_shuffled_data = 0    # total shuffled data, in bytes
