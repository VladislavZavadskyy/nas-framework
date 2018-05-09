from unittest import TestCase
from nasframe import RNNSpace, Architect

from os.path import exists
from os import remove
from shutil import rmtree

import torch


class TestArchitect(TestCase):
    def setUp(self):
        self.space = RNNSpace(5, 3)
        self.arch = Architect(self.space)

    def test_is_cuda(self):
        self.assertFalse(self.arch.is_cuda)
        self.arch.cuda()
        self.assertTrue(self.arch.is_cuda)
        self.arch.cpu()

    def test_device(self):
        self.assertEqual(self.arch.device, torch.device('cpu'))
        self.arch.cuda()
        self.assertEqual(self.arch.device, torch.device('cuda:0'))
        self.arch.cpu()

    def test_reset(self):
        self.arch.reset()

    def test_init_hidden(self):
        hidden = self.arch.init_hidden(1)
        self.assertIsInstance(hidden, (list, tuple))
        self.assertEqual(len(hidden), len(self.arch.cells))

    def test_save(self):
        existed = exists('architect')
        self.assertFalse(exists('architect/the_best_name_ever.pth'))
        self.arch.save(name='the_best_name_ever')
        self.assertTrue(exists('architect/the_best_name_ever.pth'))
        if existed:
            remove('architect/the_best_name_ever.pth')
        else:
            rmtree('architect')

    def test_sample(self):
        for i in range(100):
            self.arch.sample()

    def test_evaluate_description(self):
        for i in range(30):
            sample = self.arch.sample()
            resample = self.arch.evaluate_description(sample[0])

            self.assertTrue((sample[1] - resample[1]).abs().sum() < 1e-5)
            self.assertTrue((sample[2] - resample[2]).abs().sum() < 1e-5)
            self.assertTrue((sample[3] - resample[3]).abs().sum() < 1e-5)