import copy
from unittest import TestCase
from flatten_dict import flatten
from functools import partial

from nasframe import MLPSpace, Architect
from nasframe.utils.misc import nested_defaultdict

from os.path import exists
from os import remove
from shutil import rmtree
import torch


class TestMLPSpace(TestCase):
    def setUp(self, num_nodes=1, num_states=1, input_dim=128, bias=True,
              add_unnecessary_weights=False, add_weight_after_combine=False,
              shape_resolution_strategy='min'):
        self.space = MLPSpace(num_nodes, bias=bias,
                              add_unnecessary_weights=add_unnecessary_weights,
                              add_weight_after_combine=add_weight_after_combine,
                              shape_resolution_strategy=shape_resolution_strategy,
                              activations=['relu'])
        self.arch = Architect(self.space)
        while True:
            sample = self.arch.sample()
            self.description = sample[0]
            self.desc = self.space.preprocess(self.description, (-1, input_dim))
            if self.desc is not None:
                break

    def test_is_cuda(self):
        self.assertIsInstance(self.space.is_cuda, bool)

    def test_device(self):
        device = self.space.device
        self.assertIsInstance(device, torch.device)
        if device.type == 'cuda':
            self.assertLess(device.index, torch.cuda.device_count())

    def test_delete_layer(self):
        key = 0, 1, (10,10)
        input_id, node_id, shape = key

        self.space.get_layer(*key)
        index = self.space.get_index(*key)
        self.space.delete_layer(*key)

        self.assertRaises(IndexError, lambda: self.space.layers[index])
        self.assertRaises(KeyError, lambda: self.space.layer_index[input_id][node_id][shape])
        self.assertRaises(KeyError, lambda: self.space.layer_created[input_id][node_id][shape])
        self.assertEquals(self.space.usage_count[input_id][node_id][shape], 0)
        self.assertRaises(KeyError, lambda: self.space.last_used[input_id][node_id][shape])

    def test_get_layer(self):
        input_id, node_id, shape = 0, 1, (10, 10)
        layer = self.space.get_layer(input_id, node_id, shape)
        index = self.space.layer_index[input_id][node_id][shape]

        try:
            self.space.layers[index]
            self.space.layer_index[input_id][node_id][shape]
            self.space.layer_created[input_id][node_id][shape]
            self.space.usage_count[input_id][node_id][shape]
            self.space.last_used[input_id][node_id][shape]
        except KeyError:
            self.fail("get_layer doesn't work very well.")

        self.assertEquals(layer.weight.device, self.space.device)

    def test__init_layer(self):
        layer = torch.nn.Linear(10,10)
        self.space._init_layer(layer)
        self.assertEquals(layer.bias.mean().item(), 0)

    def test_reset(self):
        self.space.get_layer(0,1,(10,10))
        self.space.reset()

        self.assertEquals(len(self.space.layers), 0)
        self.assertEquals(len(self.space.usage_count), 0)
        self.assertEquals(len(self.space.last_used), 0)
        self.assertEquals(len(self.space.layer_index), 0)
        self.assertEquals(len(self.space.layer_created), 0)

    def test__count_related_params(self):
        # num_nodes, num_states, input_dim, bias,
        # add_unnecessary_weights,add_weight_after_combine, shape_resolution_strategy
        args = [
            [1, 1, 1, False, False, False, 'min'],
            [10, 10, 128, True, True, True, 'max']
        ]
        for argset in args:
            self.setUp(*argset)

            existing, new, total_existing, _ = self.space.parameter_count(self.desc)
            self.assertEquals(existing, 0)
            self.assertEquals(total_existing, 0)

            input = torch.randn(6, argset[2])
            model = self.space.prepare(input, self.desc)
            model(input, self.desc)

            new_existing, new_new, new_total_existing, _ = self.space.parameter_count(self.desc)
            self.assertEquals(new_existing, new)
            self.assertEquals(new_new, 0)
            self.assertEquals(new_total_existing, new)

    def test_to(self):
        self.space.to('cuda:0')
        for layer in self.space.layers:
            self.assertEqual(layer.weight.device, torch.device('cuda:0'))
        self.space.cpu()
        for layer in self.space.layers:
            self.assertEqual(layer.weight.device, torch.device('cpu'))

    def test_save(self):
        existed = exists(f'{self.space.name}')
        save_name = 'the_best_name_ever'
        save_path = f'{self.space.name}/{save_name}.pth'
        if exists(save_path): remove(save_path)
        self.assertFalse(exists(save_path))
        self.space.save(name=save_name)
        self.assertTrue(exists(save_path))
        if existed:
            remove(save_path)
        else:
            rmtree(f'{self.space.name}')

    def test__get_value(self):
        d = {1: {2: {3: 4}}}
        v = self.space._get_value(d, 1, 2, 3)
        self.assertEqual(v, 4)
        d = nested_defaultdict(3, dict)
        v = self.space._get_value(d, 1, 2, 3)
        self.assertIsInstance(v, dict)
        self.assertEqual(len(v), 0)

    def test__set_value(self):
        d = {1: {2: {3: 4}}}
        self.space._set_value(d, 'value', 1, 2, 3)
        val = d[1][2][3]
        self.assertIs(val, 'value')

    def test__del_value(self):
        d = {1: {2: {3: 4}}}
        self.space._del_value(d, 1, 2)
        self.assertIsNone(d[1].get(2))

    def test_fill_the_gaps(self):
        for node_id in range(self.description['num_node']):
            self.assertIsNone(self.description['node'][node_id].get('activation'))
        backup = copy.deepcopy(self.description)
        filled = self.space.fill_the_gaps(self.description)
        self.assertEqual(self.description, backup, '`fill_the_gaps` mutates description.')
        for node_id in range(self.description['num_node']):
            self.assertIsNotNone(filled['node'][node_id].get('activation'))

    def test_connect(self):
        for node_id in range(self.description['num_node']):
            self.assertIsInstance(self.description['node'][node_id]['input'], dict)
        backup = copy.deepcopy(self.description)
        connected = self.space.connect(self.description)
        self.assertEqual(self.description, backup, '`connect` mutates description.')
        for node_id in range(self.description['num_node']):
            self.assertIsInstance(connected['node'][node_id]['input'], list)
            self.assertGreater(len(connected['node'][node_id]['input']), 0)

    def test_connect_leafs(self):
        connected = self.space.connect(self.description)
        backup = copy.deepcopy(connected)
        l_connected = self.space.connect_leafs(connected)
        self.assertEqual(connected, backup)
        last_node = max(l_connected['node'])
        self.assertGreaterEqual(len(connected['node'][last_node]['input']),
                                len(l_connected['node'][last_node]['input']))

    def test_pick(self):
        avail = ['one', 'two', 'three', 'four']
        pick = self.space.pick
        self.assertIs(pick('two', avail, 'None'), 'two')
        self.assertIs(pick('five', avail, 'first'), 'one')
        self.assertIs(pick('five', avail, 'last'), 'four')
        self.assertIsNone(pick('five', avail, 'ignore'))
        self.assertIs(len(avail), 1)

    def test_prune(self):
        viable = False
        while not viable:
            description = self.arch.sample()[0]
            connected = self.space.connect(description)
            backup = copy.deepcopy(connected)
            pruned, viable = self.space.prune(connected)
            self.assertEqual(connected, backup)
        self.assertLessEqual(pruned['num_node'], connected['num_node'])
        for node_id in range(self.description['num_node']):
            self.assertLessEqual(len(pruned['node'][node_id]['input']),
                                 len(connected['node'][node_id]['input']))

    def test_resolve_shapes(self):
        connected = self.space.connect(self.description)
        backup = copy.deepcopy(connected)
        resolved = self.space.resolve_shapes(connected, (-1,128))
        self.assertEqual(connected, backup)
        for node_id in range(self.description['num_node']):
            self.assertIn('dim', resolved['node'][node_id])
            self.assertIsInstance(resolved['node'][node_id]['dim'], int)

    def test_preprocess(self):
        backup = copy.deepcopy(self.description)
        self.space.preprocess(self.description, (-1, 128))
        self.assertEqual(backup, self.description)

    def test_prepare(self):
        self.space.prepare(torch.randn(5,128), self.desc)
        self.assertRaises(ValueError, self.space.prepare, torch.randn(5, 5), self.desc)

    def test_forward(self):
        self.space.reset()
        model = self.space.prepare(torch.randn(5, 128), self.desc)
        model.zero_grad()
        out = model(torch.randn(5, 128), self.desc)
        self.assertTrue(torch.is_tensor(out))
        out.mean().backward()
        for l in self.space.layers:
            self.assertIsNotNone(l.weight.grad)
            self.assertIsNotNone(l.bias.grad)