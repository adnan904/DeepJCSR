from src.utils.constants import Constants


class Link:
	"""
	Collection of edges from source to destination ports/nodes
	"""
	def __init__(self, id, src, dst, edges, delay):
		self.rate = Constants.RACK_BITS_PER_SEC  # current, available data rate, in bits/sec
		self.id = id
		self.node1 = src
		self.node2 = dst
		self.edges = edges
		self.delay = delay
		self.next_available_time = 0
		self.is_available = True
		self.num_active_flows = 0
		self.total_time_busy_for = 0  #ms

	def reset(self):
		self.rate = Constants.RACK_BITS_PER_SEC
		self.next_available_time = 0
		self.is_available = True
		self.num_active_flows = 0
		self.total_time_busy_for = 0  # ms


