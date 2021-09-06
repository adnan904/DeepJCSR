from gym.envs.registration import register
from src.utils.params import *


# Env for joint co-flow scheduling and routing
register(
    id='Jcsr-v0',
    entry_point='src.envs.jcsr.env:JcsrEnv',
    kwargs=vars(args),
)