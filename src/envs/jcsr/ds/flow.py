from src.utils.constants import Constants


class Flow():
    """
        Simple class representing a Flow
    """
    def __init__(self, id, arrival_time, cid, src, dest, size, debug=False):
        self.id = id
        self.arrival_time = arrival_time    # in millisecs
        self.size = size * 1024  # flow size in Bytes(input is in MBs, hence the mult)
        self.cid = cid
        self.src = src
        self.dest = dest
        self.is_active = False
        self.is_finished = False
        self.link = None
        self.start_time = -1
        self.duration = (size * 1024 * 8 / Constants.RACK_BITS_PER_SEC) * Constants.MILLIS_IN_SEC # msec
        self.expected_finish_time = -1
        self.actual_finish_time = -1
        self.debug = debug

        self.data_rm = size * 1024  # remaining data, in Bytes(input is in MBs)
        self.fct = 0        # flow completion time, milliseconds

    def __repr__(self):
        return ("Flow id: {}, arrival_time: {}, data_rm: {}, rate: {}, \
            fct: {}\n\t".format(
            self.id,
            self.arrival_time,
            self.data_rm,
            str(Constants.RACK_BITS_PER_SEC),
            self.fct))

    def __eq__(self, other):
        assert isinstance(other, Flow)
        return self.id == other.id

    def fct(self):
        """
        Flow completion time based on the remaining volume and link data rate
        """

        return (self.data_rm * 8 / Constants.RACK_BITS_PER_SEC) * Constants.MILLIS_IN_SEC # msec

    def reset(self):
        self.is_active = False
        self.is_finished = False
        self.start_time = -1
        self.expected_finish_time = -1
        self.actual_finish_time = -1
        self.link = None
