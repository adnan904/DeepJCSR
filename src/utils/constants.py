
class Constants:
    RACK_BITS_PER_SEC = 1.0 * 1024 * 1024 * 1024        # 1Gbps
    RACK_BYTES_PER_SEC = RACK_BITS_PER_SEC / 8.0        # Bps
    SIM_MILLIS_IN_SEC = 1024
    MILLIS_IN_SEC = 1000
    MAX_CONCURRENT_ACTIVE_COFLOWS = 40
    MAX_CONCURRENT_ACTIVE_FLOWS = 1000
    NUM_FEATURES = 8  # (flow_id, arrival_time, coflow_id, is_processed, duration, link1, link2, link3)
    NUM_LINKS_PER_FLOW = 3
    NUM_FLOWS_PER_TRACE = 50
    DEFAULT_TOPOLOGY_FILE = "res/network/abilene.graphml"
    DEFAULT_TRAIN_TRACES = "res/config/train-traces.txt"
    DEFAULT_TEST_TRACES = "res/config/test-traces.txt"
    DEFAULT_AGENT_CONFIG = "res/config/PPO_config.yaml"
    TRACE_DIR = "results"
    DEFAULT_MODEL_NAME = "jcsr"
