import torch
from torch import nn
from tqdm import tqdm, trange
from numpy.random import uniform

from unittest import TestCase

from nasframe import RNNSpace, Architect
from nasframe.coaches import ArchitectCoach
from nasframe.storage import Storage
from nasframe.utils import logger


class TestCoaches(TestCase):
    def test_architect_coach(self):
        space = RNNSpace(20, 7)
        storage = Storage()
        arch = Architect(space)

        bar, c = tqdm(desc='Sampling', total=10), 0
        while c < 4*8:
            description, logps, values, entropies = arch.sample()
            desc = space.preprocess(description, (-1, 128))
            if desc is not None:
                complexity = sum(space.parameter_count(desc)[:2]) / 1e6  # Million parameters
                storage.append(description, logps, values, entropies, uniform(.6, 1), complexity)
                c += 1; bar.update(1)

        logger.setLevel('INFO')
        ac = ArchitectCoach(arch, storage, logger=logger, log_every=1,
                            complexity_penalty=.2, curriculum=False)
        ac.train(3)