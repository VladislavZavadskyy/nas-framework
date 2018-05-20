from .mlp import *
from .rnn import *
from .base import *


def get_space_type(name):
    """
    Gets search space type by ``name``.
    """
    name = name.strip().lower()

    type = {
        'rnn': RNNSpace,
        'mlp': MLPSpace
    }.get(name)

    if type is None:
        raise NotImplementedError(f'Search space of type {name} is not implemented.')

    return type