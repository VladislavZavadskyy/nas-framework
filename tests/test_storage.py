from nasframe.storage import Storage, CurriculumStorage
from nasframe import RNNSpace, Architect
from unittest import TestCase
from numpy.random import uniform
import numpy as np

import torch


class TestStorage(TestCase):
    def setUp(self):
        self.space = RNNSpace(20, 5)
        self.storage = Storage()
        self.arch = Architect(self.space)

    def test_append(self, na_points=0):
        c = 0
        starting_len = len(self.storage)
        while c < 10 + na_points:
            description, logps, values, entropies = self.arch.sample()
            desc = self.space.preprocess(description, (-1, 128))
            if desc is not None:
                param_count = sum(self.space.parameter_count(desc)[:2]) / 1e6
                self.storage.append(description, logps, values, entropies,
                                    uniform(.6, 1) if c < 10 else None,
                                    param_count)
                c += 1
        self.assertEqual(len(self.storage), starting_len+10+na_points)

    def test_reward(self):
        chosen_description = None
        while chosen_description is None:
            description, logps, values, entropies = self.arch.sample()
            desc = self.space.preprocess(description, (-1, 128))
            if desc is not None:
                if uniform(0, 1) < .5:
                    chosen_description = description
                param_count = sum(self.space.parameter_count(desc)[:2]) / 1e6  # Million parameters
                self.storage.append(description, logps, values, entropies, None, param_count)
        self.storage.reward(chosen_description, 1.)
        index = self.storage.find(chosen_description)
        self.assertIsNotNone(index)
        self.assertTrue(torch.eq(self.storage[index][-3], 1.).all())

    def test__update_advantages(self):
        chosen_description = None
        while chosen_description is None:
            description, logps, values, entropies = self.arch.sample()
            desc = self.space.preprocess(description, (-1, 128))
            if desc is not None:
                if uniform(0, 1) < .5:
                    chosen_description = description
                param_count = sum(self.space.parameter_count(desc)[:2]) / 1e6  # Million parameters
                self.storage.append(description, logps, values, entropies, None, param_count)
        for adv in self.storage.advantages:
            self.assertEqual(torch.sum(adv != adv), adv.numel())
        self.storage.reward(chosen_description, 1.)
        index = self.storage.find(chosen_description)
        self.assertIsNotNone(index)
        self.assertFalse(np.isnan(self.storage[index][-3].mean().item()))

    def test_update(self):
        desc = None
        while desc is None:
            description, logps, values, entropies = self.arch.sample()
            desc = self.space.preprocess(description, (-1, 128))
            if desc is not None:
                param_count = sum(self.space.parameter_count(desc)[:2]) / 1e6  # Million parameters
                self.storage.append(description, logps, values, entropies, None, param_count)
                self.arch.reset()
                _, logps_, values_, entropies_ = self.arch.evaluate_description(description)
                self.storage.update(description, logps_, values_, entropies_)
                index = self.storage.find(description)
                self.assertIsNotNone(index)
                _, logps, values, entropies, _, _, _ = self.storage[index]
                self.assertTrue(torch.eq(logps, logps_).all())
                self.assertTrue(torch.eq(values, values_).all())
                self.assertTrue(torch.eq(entropies, entropies_).all())

    def test___get_item__(self):
        self.test_append()
        self.storage[5]
        self.storage[1:4]
        self.storage[1:4:-1]
        self.storage[np.arange(5)]

    def test___del_item__(self):
        self.test_append()
        item = self.storage[5]
        del self.storage[5]
        self.assertNotEqual(item, self.storage[5])

    def test___len__(self):
        self.test_append()
        self.assertIs(len(self.storage), 10)

    def test_filterna(self):
        self.test_append(5)
        self.assertEqual(len(self.storage), 15)
        self.storage.filter_na()
        self.assertEqual(len(self.storage), 10)

class TestCurriculumStorage(TestStorage):
    def setUp(self):
        super().setUp()
        self.storage = CurriculumStorage(20)

    def test_append(self):
        super().test_append()
        self.assertIs(len(self.storage), 10)
        self.storage.set_complexity(2)
        super().test_append()
        super().test_append()
        self.assertIs(len(self.storage), 20)
        self.storage.set_complexity(1)
        self.assertIs(len(self.storage), 10)
