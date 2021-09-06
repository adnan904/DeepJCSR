import networkx as nx
from geopy.distance import distance as dist
import numpy as np
import math
from src.utils.constants import Constants
import logging
from collections import defaultdict

from src.envs.jcsr.ds.link import Link

log = logging.getLogger(__name__)

"""
Network parsing module. Reads and parses network files into NetworkX object
 """


def weight(edge_cap, edge_delay):
    """
    edge weight = 1 / (cap + 1/delay) => prefer high cap, use smaller delay as additional influence/tie breaker
    if cap = None, set it to 0 use edge_delay as weight
    """
    assert edge_delay is not None
    if edge_cap is None:
        return edge_delay
    if edge_cap == 0:
        return math.inf
    elif edge_delay == 0:
        return 0
    return 1 / (edge_cap + 1 / edge_delay)


def read_network(file):
    """
    Read the GraphML file and return networX object.
    """
    SPEED_OF_LIGHT = 299792458  # meter per second
    PROPAGATION_FACTOR = 0.77  # https://en.wikipedia.org/wiki/Propagation_delay
    links_map = defaultdict(lambda: defaultdict(list))
    link_objects = []

    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))

    graphml_network = nx.read_graphml(file, node_type=int)
    networkx_network = nx.Graph()

    #  Setting the nodes of the NetworkX Graph
    for n in graphml_network.nodes(data=True):
        node_id = n[0]
        node_name = n[1].get("label", None)
        # Adding a Node in the NetworkX Graph
        # {"id": node_id}
        networkx_network.add_node(node_id, name=node_name)

    # set links
    # calculate link delay based on geo positions of nodes;

    for e in graphml_network.edges(data=True):
        # Check whether LinkDelay value is set, otherwise default to None
        source = e[0]
        target = e[1]
        link_delay = e[2].get("LinkDelay", None)
        # As edges are undirectional, only LinkFwdCap determines the available data rate
        link_fwd_cap = Constants.RACK_BITS_PER_SEC

        # Setting a default delay of 3 incase no delay specified in GraphML file
        # and we are unable to set it based on Geo location
        delay = 3
        if link_delay is None:
            n1 = graphml_network.nodes(data=True)[e[0]]
            n2 = graphml_network.nodes(data=True)[e[1]]
            n1_lat, n1_long = n1.get("Latitude", None), n1.get("Longitude", None)
            n2_lat, n2_long = n2.get("Latitude", None), n2.get("Longitude", None)
            if n1_lat is None or n1_long is None or n2_lat is None or n2_long is None:
                log.warning("Link Delay not set in the GraphML file and unable to calc based on Geo Location,"
                            "Now using default delay for edge: ({},{})".format(source, target))
            else:
                distance = dist((n1_lat, n1_long), (n2_lat, n2_long)).meters  # in meters
                # round delay to int using np.around for consistency with emulator
                delay = int(np.around((distance / SPEED_OF_LIGHT * 1000) * PROPAGATION_FACTOR))  # in milliseconds
        else:
            delay = link_delay

        # Adding the undirected edges for each link defined in the network.
        # delay = edge delay , cap = edge capacity
        networkx_network.add_edge(source, target, delay=delay, cap=link_fwd_cap, remaining_cap=link_fwd_cap)

    # setting the weight property for each edge in the NetworkX Graph
    # weight attribute is used to find the shortest paths
    for edge in networkx_network.edges.values():
        edge['weight'] = weight(edge['cap'], edge['delay'])

    # Finding the 3 shortest paths(based on delay) between all pairs of nodes in the network
    link_id = 0
    for s in range(len(networkx_network.nodes)):
        for d in range(len(networkx_network.nodes)):
            if s == d:
                link = Link(link_id, s, d, [], 0)
                link_objects.append(link)
                links_map[s][d] = [link_id, link_id, link_id]
                link_id += 1
            else:
                if len(links_map[s][d]) > 0:
                    continue
                paths = nx.all_simple_paths(networkx_network, source=s, target=d)
                paths = [list(p) for p in map(nx.utils.pairwise, paths)]
                assert len(paths) > 0, f'No path between nodes: {s} and {d}. Graph should be connected'
                assert len(paths) > 2, f'Less than 3 paths between source and destination'
                delays = []
                paths_delay_dict = defaultdict(list)
                for path in paths:
                    total_path_delay = 0
                    for tuple in path:
                        e = networkx_network.get_edge_data(tuple[0], tuple[1])
                        total_path_delay += e['delay']
                    delays.append(total_path_delay)
                    paths_delay_dict[total_path_delay] = path
                delays = sorted(delays)
                if len(delays) >= 3:
                    link1 = Link(link_id, s, d, paths_delay_dict[delays[0]], delays[0])
                    link_objects.append(link1)
                    links_map[s][d].append(link_id)
                    links_map[d][s].append(link_id)
                    link_id += 1
                    link2 = Link(link_id, s, d, paths_delay_dict[delays[1]], delays[1])
                    link_objects.append(link2)
                    links_map[s][d].append(link_id)
                    links_map[d][s].append(link_id)
                    link_id += 1
                    link3 = Link(link_id, s, d, paths_delay_dict[delays[2]], delays[2])
                    link_objects.append(link3)
                    links_map[s][d].append(link_id)
                    links_map[d][s].append(link_id)
                    link_id += 1

    return networkx_network, link_objects, links_map
