from .mlp_space import TestMLPSpace
from nasframe.searchspaces import RNNSpace
from nasframe.architect import Architect

import torch
import copy


class TestRNNSpace(TestMLPSpace):
    def setUp(self, num_nodes=1, num_states=1, input_dim=128, bias=True,
              add_unnecessary_layers=False, add_layer_after_combine=False,
              shape_resolution_method='min'):
        self.space = RNNSpace(num_nodes, num_states, bias=bias,
                              add_unnecessary_layers=add_unnecessary_layers,
                              add_layer_after_combine=add_layer_after_combine,
                              shape_resolution_method=shape_resolution_method,
                              activations=['relu'])
        self.arch = Architect(self.space)
        while True:
            sample = self.arch.sample()
            self.description = sample[0]['rnn'][0]
            self.desc = self.space.preprocess(sample[0], (-1, input_dim))
            if self.desc is not None:
                break

    def test__count_related_params(self):
        # num_nodes, num_states, input_dim, bias,
        # add_unnecessary_weights,add_weight_after_combine, shape_resolution_strategy
        args = [
            [1, 1, 1, False, False, False, 'min'],
            [10, 10, 128, True, True, True, 'max']
        ]
        for argset in args:
            self.setUp(*argset)
            self.space.cuda()

            existing, new, total_existing, _ = self.space.parameter_count(self.desc)
            self.assertEquals(existing, 0)
            self.assertEquals(total_existing, 0)

            input = torch.randn(5, 6, argset[2])
            model = self.space.prepare(input, self.desc)
            model(input, self.desc)

            new_existing, new_new, new_total_existing, _ = self.space.parameter_count(self.desc)
            self.assertEquals(new_existing, new)
            self.assertEquals(new_new, 0)
            self.assertEquals(new_total_existing, new)

            self.space.cpu()

    def test_connect(self):
        for node_id in range(self.description['num_node']):
            self.assertIsInstance(self.description['node'][node_id]['input'], dict)
        backup = copy.deepcopy(self.description)
        connected = self.space.connect(self.description)
        self.assertEqual(self.description, backup, '`connect` mutates description.')
        for node_id in range(self.description['num_node']):
            self.assertIsInstance(connected['node'][node_id]['input'], list)
            self.assertGreater(len(connected['node'][node_id]['input']), 0)
        for state_id in range(self.description['num_state']):
            self.assertIsInstance(connected['node'][state_id]['input'], list)
            self.assertGreater(len(connected['node'][state_id]['input']), 0)

    def test_connect_leafs(self):
        connected = self.space.connect(self.description)
        backup = copy.deepcopy(connected)
        l_connected = self.space.connect_leafs(connected)
        self.assertEqual(connected, backup)
        self.assertGreaterEqual(len(connected['state'][0]['input']),
                                len(l_connected['state'][0]['input']))

    def test_prune(self):
        viable = False
        while not viable:
            description = self.arch.sample()[0]
            connected = self.space.connect(description)
            backup = copy.deepcopy(connected)
            pruned, viable = self.space.prune(connected)
            self.assertEqual(connected, backup)
        pruned = pruned['rnn'][0]
        connected = connected['rnn'][0]
        self.assertLessEqual(pruned['num_node'], connected['num_node'])
        self.assertLessEqual(pruned['num_state'], connected['num_state'])
        for node_id in range(self.description['num_node']):
            self.assertLessEqual(len(pruned['node'][node_id]['input']),
                                 len(connected['node'][node_id]['input']))
        for state_id in range(self.description['num_node']):
            self.assertLessEqual(len(pruned['state'][state_id]['input']),
                                 len(connected['state'][state_id]['input']))

    def test_prepare(self):
        self.space.prepare(torch.randn(5, 6, 128), self.desc)
        self.assertRaises(ValueError, self.space.prepare, torch.randn(5, 5, 6), self.desc)

    def test_forward(self):
        self.space.reset()
        model = self.space.prepare(torch.randn(5, 6, 128), self.desc)
        model.zero_grad()
        out = model(torch.randn(5, 6, 128), self.desc)
        for state in out:
            self.assertTrue(torch.is_tensor(state))
        torch.stack(out).mean().backward()
        for l in self.space.layers:
            self.assertIsNotNone(l.weight.grad)
            self.assertIsNotNone(l.bias.grad)