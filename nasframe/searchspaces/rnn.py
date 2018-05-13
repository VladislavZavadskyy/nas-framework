from .mlp import *
from functools import wraps


def transforms_description(f):
    """
    Decorator which deepcopies the description and extracts ['rnn'][0] from it.
    Desperately needs to be obsoleted.
    """
    @wraps(f)
    def wrapper(description, *args, **kwds):
        description = copy.deepcopy(description)
        try: desc = description['rnn'][0]
        except KeyError: desc = description
        desc = f(desc, *args, **kwds)
        try: description['rnn'][0] = desc
        except KeyError: description = desc
        return description
    return wrapper


class RNNSpace(MLPSpace):
    """
    Defines a recurrent neural network search space.

    Args:
        max_nodes (int): maximum number of nodes in cell
        max_states (int): maximum number of hidden states
        state_dims (tuple): possible hidden state's dimension numbers
        out_sizes (tuple): possible numbers of output dims relative to the number of input dims (num_out/num_in)
        input_combine_ops (tuple): names of operations to combine inputs, if there's more than one
        activations (tuple): names of nonlinearities to apply to layer outputs
        drop_probs (tuple): possible node input dropout probabilities
        recurrent_drop_probs (tuple): possible recurrent dropout probabilities
        max_node_inputs (int, str, optional): maximum number of layers to be used as input for node or expression
            to be evaluated, which returns this number. If not specified, defaults to maximum possible number of inputs.
        max_node_inputs (int, str, optional): maximum number of layers to be used as input for state or expression
            to be evaluated, which returns this number. If not specified, defaults to maximum possible number of inputs.
        use_zoneout (bool): whether to use zoneout instead of recurrent dropout.
        kwargs: keyword arguments to be passed to MLPSpace constructor.
    """
    def __init__(self, max_nodes, max_states,
                 state_dims=(16,32,64,128),
                 out_sizes=(.2, .5, 1, 2, 5),
                 input_combine_ops=('add','mult','avg','u_gate','concat'),
                 activations=('relu','sigmoid','tanh', 'identity'),
                 drop_probs=(0.,0.1,0.3,0.5),
                 recurrent_drop_probs=(0.,0.1,0.3,0.5),
                 max_node_inputs=None, max_state_inputs=None,
                 use_zoneout=True, **kwargs):

        super().__init__(1, **kwargs)

        self.use_zoneout = use_zoneout

        max_node_inputs = max_node_inputs or f'list(range(1, outer_i+{max_states+2}))'
        max_state_inputs = max_state_inputs or max_nodes

        # region Node space definition
        state_names = ', '.join([f'"state_{i}"' for i in range(max_states)])
        self.node_space = SearchSpace(
            name='node',
            outer={},
            num_inner=int_to_range(max_nodes, 1),
            inner=SearchSpace(
                name='input',
                outer=OrderedDict({
                    'inp_combine': input_combine_ops,
                    'out_size': out_sizes,
                    'activation': activations,
                    'drop_prob': drop_probs,
                }),
                num_inner=int_to_range(max_node_inputs, 1),
                inner={
                    'id': f'["graph_input", {state_names}] + '
                           '["node_%d"%i for i in range(outer_i)]'
                })
        )
        # endregion
        # region State space definition
        self.state_space = SearchSpace(
            name='state',
            outer={},
            num_inner=int_to_range(max_states, 1),
            inner=SearchSpace(
                name='input',
                outer=OrderedDict({
                    'inp_combine': input_combine_ops,
                    'dim': state_dims,
                    'rec_drop_prob': recurrent_drop_probs,
                }),
                num_inner=int_to_range(max_state_inputs, 1),
                inner={
                    'id': [f'node_{i}' for i in range(max_nodes)],
                })

        )
        # endregion

        self.name = 'rnn'
        self.num_inner = [1]
        self.inner = [self.node_space, self.state_space]

        self.dimension_names.update({
            'rec_drop_prob': 'Zoneout' if self.use_zoneout else 'Recurrent dropout',
        })

    # region Curriculum related stuff

    def force_num_states(self, n):
        """
        Signals architect, that number of states should always be ``n``.
        """
        self.state_space.force_num_inner(n)

    def force_num_state_inputs(self, n):
        """
        Signals architect, that number of state inputs should always be ``n``.
        """
        self.state_space.inner.force_num_inner(n)

    def force_num_nodes(self, n):
        """
        Signals architect, that number of nodes should always be ``n``.
        """
        self.node_space.inner.force_num_inner(n)

    def force_num_node_inputs(self, n):
        """
        Signals architect, that number of node inputs should always be ``n``.
        """
        self.node_space.inner.force_num_inner(n)

    def release_num_nodes(self):
        """
        Releases constraints set on number of nodes.
        """
        self.node_space.release_num_inner()

    def release_num_node_inputs(self):
        """
        Releases constraints set on number of node inpus.
        """
        self.node_space.inner.release_num_inner()

    def release_num_states(self):
        """
        Releases constraints set on number of states.
        """
        self.state_space.release_num_inner()

    def release_num_state_inputs(self):
        """
        Releases constraints set on number of state inputs.
        """
        self.state_space.inner.release_num_inner()

    def set_curriculum_complexity(self, i):
        """
        Sets curriculum complexity ``i``.
        """
        self.node_space.force_num_inner(i)
        self.state_space.force_num_inner(i)

    def release_all_constraints(self):
        """
        Releases all constraints set by ``force*`` or ``set_curriculum_complexity`` methods.
        """
        super().release_all_constraints()
        self.release_num_state_inputs()
        self.release_num_states()

    # endregion

    # region Parameter and memory management
    def parameter_count(self, description, create_nonexistent=False):
        """
        Returns tuple of existing, new and total existing parameter counts of the model described by ``description``
        and a list of keys corresponding to used layers.
        if ``create_nonexistent`` is True, creates described layers along the way.
        """
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        if desc['node'][0].get('dim') is None:
            raise ValueError("The shapes are not resolved. "
                             "Please pass the result of `RNNSpace.resolve_shapes` as description next time.")

        (existing, new), used_dict = self._count_related_params(desc, 'node', create_nonexistent)
        (s_existing, s_new), s_used_dict = self._count_related_params(desc, 'state', create_nonexistent)

        used_dict = flatten(used_dict)
        used_dict.update(flatten(s_used_dict))
        existing += s_existing
        new += s_new

        total_existing = 0
        for module in self.layers:
            total_existing += param_count(module)

        return existing, new, total_existing, used_dict
    # endregion

    # region Graph description processing
    def _add_agraph_units(self, description, graph: gv.AGraph):
        """
        Adds units to ``AGraph``.

        Args:
            description (dict): description of a graph
            graph (gv.AGraph): pygraphviz graph to add units to

        Returns:
            gv.AGraph: graph with units added
        """
        description = copy.deepcopy(description)
        try: description = description['rnn'][0]
        except KeyError: description = description

        graph = super()._add_agraph_units(description, graph)

        for i in range(description['num_state']):
            state = description['state'][i]
            state_name = f'state_{i}'
            label = f'<h<sup>{i}</sup><sub>t</sub><BR/>'
            for k, v in state.items():
                if k in ['input', 'num_input', 'inp_combine']:
                    continue
                label += f'{self.dimension_names[k]}: {v}<BR/>'
            label += '>'
            graph.add_node(n=state_name, label=label, **state_attrs)

        for i in range(description['num_state']):
            state = description['state'][i]
            state_name = f'state_{i}_t+1'
            label = f'<h<sup>{i}</sup><sub>t+1</sub><BR/>'
            for k, v in state.items():
                if k in ['input', 'num_input', 'rec_drop_prob', 'dim']:
                    continue
                if k == 'inp_combine':
                    self._add_agraph_unit(graph, state, state_name, v)
                    continue

                label += f'{self.dimension_names[k]}: {v}<BR/>'
            label += '>'
            graph.add_node(n=state_name, label=label, **state_attrs)
        return graph

    def fill_the_gaps(self, description, **kwargs):
        """
        Fills gaps in the description.

        If a certain search space dimension has only one possible value, it is ignored by architect and thus
        is absent in descriptions generated by it. This method fills such description 'gaps' with that only value.

        Args:
            description (dict): graph description
        Returns:
             dict: graph description with default values filled in.
        """
        description = copy.deepcopy(description)
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        desc = SearchSpace._fill_the_gaps(desc, self.inner)

        try: description['rnn'][0] = desc
        except KeyError: description = desc
        return description

    def resolve_shapes(self, description, input, method=None, **kwargs):
        """
        Converts relative shapes to absolute.

        Args:
            desc: description of a graph.
            input: graphs input or its shape.
            method: method to resolve shape ambiguity; either 'min', 'max' or 'mean'

        Returns:
            dict: description with shapes resolved.
        """
        description = copy.deepcopy(description)
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        method = method or self.shape_resolution_method
        desc = SearchSpace._resolve_shapes(desc, input, method, 'node')

        try: description['rnn'][0] = desc
        except KeyError: description = desc
        return description

    def connect(self, description, picking_method=None):
        """
        Resolves graph's connections, by picking requested input names
        from available input names using ``picking_method``.

        Args:
            description (dict): description to be connected
            picking_method (str): name of the input picking method

        Returns:
            dict: connected description

        """
        description = copy.deepcopy(description)
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        picking_method = picking_method or self.input_picking_method

        state_names = ', '.join([f'"state_{i}"' for i in range(desc['num_state'])])
        node_inputs = f'["graph_input", {state_names}] + ["node_%d"%i for i in range(outer_i)]'
        state_inputs = [f'node_{i}' for i in range(desc['num_node'])]

        super()._connect(desc, self.node_space, picking_method, node_inputs)
        super()._connect(desc, self.state_space, picking_method, state_inputs)

        try: description['rnn'][0] = desc
        except KeyError: description = desc
        return description

    @staticmethod
    @transforms_description
    def connect_leafs(desc):
        """
        Connects leaf nodes, which otherwise would be pruned, to the output.
        """
        _, useful_nodes, _ = RNNSpace.walk_graph(desc)
        leaf_nodes = set(desc['node'].keys()).difference(set(useful_nodes))
        desc['state'][0]['input'] = list(set(desc['state'][0]['input']).union(leaf_nodes))
        return desc

    @staticmethod
    def prune(description):
        """
        Returns modified copy of the ``description``, which doesn't contain unused units and boolean value
        indicating whether the description is viable (i.e. contains used units).
        """
        description = copy.deepcopy(description)
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        depends_on_input, useful_nodes, useful_states = RNNSpace.walk_graph(desc)
        if not depends_on_input: return {}, False

        mapping = {'graph_input': 'graph_input'}
        mapping.update(map(lambda i: (f'state_{i[1]}', f'state_{i[0]}'), enumerate(useful_states)))
        mapping.update(map(lambda i: (f'node_{i[1]}', f'node_{i[0]}'), enumerate(useful_nodes)))

        SearchSpace._prune(desc, 'node', mapping)
        SearchSpace._prune(desc, 'state', mapping)

        viable = depends_on_input and desc['num_state'] > 0 and desc['num_node'] > 0
        viable &= all(map(lambda n: len(n['input']) > 0, desc['node'].values()))
        viable &= all(map(lambda s: len(s['input']) > 0, desc['state'].values()))

        try: description['rnn'][0] = desc
        except KeyError: description = desc
        return description, viable

    @staticmethod
    def walk_graph(description):
        """
        Walks described graph backwards to determine which nodes are connected to
        at least one :math:`h_{t+1}` state (useful nodes), which states are being both read from on step  :math:`t`
        and written to on step :math:`t+1`  (useful states) and whether the graph depends on input at all.

        Args:
            description: description of a graph.

        Returns:
            tuple: ((bool) whether graph depends on input,
                    (set) of useful node ids,
                    (set) of useful state ids)
        """
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        persistent_nodes = {'graph_input': set()}

        reachable_from_nodes = SearchSpace._walk_graph(desc, 'node', persistent_nodes)
        while True:
            reachable_from_states = SearchSpace._walk_graph(desc, 'state', persistent_nodes)

            for name, node_set in reachable_from_states.items():
                for input_node in set(node_set):
                    node_set.update(reachable_from_nodes.get(input_node, set()))

            new_persistent = dict(filter(lambda i: any(n in i[1] for n in persistent_nodes),
                                         reachable_from_states.items()))

            new_persistent = dict.fromkeys(new_persistent, set())
            new_persistent.update(persistent_nodes)

            if persistent_nodes.keys() == new_persistent.keys():
                break
            persistent_nodes = new_persistent

        reachable_from_any_state = set(chain.from_iterable(reachable_from_states.values()))
        useful_states = filter(lambda n: n in reachable_from_any_state, persistent_nodes)
        useful_states = filter(lambda name: 'state' in name, useful_states)
        useful_states = set(map(lambda s: int(s.split('_')[1]), useful_states))

        reachable_from_nodes = SearchSpace._walk_graph(desc, 'node', persistent_nodes)
        reachable_from_useful_states = map(lambda i: reachable_from_states[f'state_{i}'], useful_states)
        reachable_from_useful_states = set(chain.from_iterable(reachable_from_useful_states))

        useful_nodes = filter(lambda i: any(n in i[1] for n in persistent_nodes),
                              reachable_from_nodes.items())
        useful_nodes = filter(lambda i: i[0] in reachable_from_useful_states, useful_nodes)
        useful_nodes = set(map(lambda i: int(i[0].split('_')[1]), useful_nodes))

        depends_on_input = 'graph_input' in reachable_from_any_state

        return depends_on_input, useful_nodes, useful_states
    # endregion

    def init_hidden(self, batch_size, dims, device=None):
        """
        Returns list of ``len(dims)`` hidden states initialized with zeros for ``batch_size``.

        Args:
            dims (list, tuple): dimensions of hidden states
        """
        device = device or self.device
        return [wrap(torch.zeros(batch_size, dim),
                     device, requires_grad=False) for dim in dims]

    def cell(self, input, hidden, description):
        """
        Processes one time step of the sequence.

        Args:
             input (torch.Tensor): input tensor
             hidden (list): hidden states
             desc (dict): description of a RNN.

        Returns:
            list: hidden states
        """
        outputs = {'graph_input': input}
        for state_id in range(description['num_state']):
            outputs[f'state_{state_id}'] = hidden[state_id]

        for node_id in range(description['num_node']):
            node = description['node'][node_id]
            node_name = f'node_{node_id}'

            interm_outs = []
            for input_name in node['input']:
                shape = (outputs[input_name].shape[-1], node['dim'])
                if (shape[0] != shape[1] or self.add_unnecessary_layers)\
                        and node['inp_combine'] != 'concat':

                    interm_inp = outputs[input_name]
                    if node['drop_prob'] > 0 and self.training:
                        interm_inp = nn.Dropout(node['drop_prob'])(interm_inp)

                    layer = self.get_layer(input_name, node_name, shape)
                    interm_outs.append(layer(interm_inp))
                else:
                    interm_outs.append(outputs[input_name])

            outputs[node_name] = self.combine_tensors(node['inp_combine'], interm_outs)

            if self.add_layer_after_combine or node['inp_combine'] == 'concat':
                if node['drop_prob'] > 0 and self.training:
                    outputs[node_name] = nn.Dropout(node['drop_prob'])(outputs[node_name])
                layer = self.get_layer('combine', node_name, (outputs[node_name].shape[-1], node['dim']))
                outputs[node_name] = layer(outputs[node_name])

            outputs[node_name] = get_activation(node['activation'])(outputs[node_name])

        h_next = []
        for state_id in range(description['num_state']):
            state = description['state'][state_id]
            state_name = f'state_{state_id}'

            interm_outs = []
            for input_name in state['input']:
                shape = (outputs[input_name].shape[-1], state['dim'])
                if (shape[0] != shape[1] or self.add_unnecessary_layers)\
                        and state['inp_combine'] != 'concat':
                    layer = self.get_layer(input_name, state_name, shape)
                    interm_outs.append(layer(outputs[input_name]))
                else:
                    interm_outs.append(outputs[input_name])

            h_next.append(self.combine_tensors(state['inp_combine'], interm_outs))

            if self.add_layer_after_combine or state['inp_combine'] == 'concat':
                layer = self.get_layer('combine', state_name, (h_next[state_id].shape[-1], state['dim']))
                h_next[state_id] = layer(h_next[state_id])

        return h_next

    def forward(self, inputs, description, hidden=None, return_sequence=True):
        """
        Does ``inputs.size(1)`` forward passes through the described graph.
        Call ``prepare`` first, and use ``forward`` method of the returned module instead.

        Args:
            inputs (torch.Tensor): inputs to be forwarded
            description (dict): description of a graph to forward through
            hidden (list): hidden state, if None initialized with zeros
            return_sequence (bool): if True, will return sequence instead of the last element in it

        Returns:
            list: last hidden state or history of hidden states, depending on ``return_sequence`` value.
        """
        try: desc = description['rnn'][0]
        except KeyError: desc = description

        if isinstance(desc['state'][0]['input'], dict):
            raise ValueError("The description is not connected. "
                             "Please pass the result of `RNNSpace.connect(description)` next time.")

        assert torch.is_tensor(inputs), f'How dare you pass {type(inputs)} to me?'
        assert inputs.dim() == 3, 'Input shape must be [batch, timesteps, input_dim]'

        inputs = inputs.to(self.device)

        state_dims = list(map(lambda s: s['dim'], desc['state'].values()))
        hidden = hidden or self.init_hidden(inputs.shape[0], state_dims)

        dp_masks = []
        for st in range(desc['num_state']):
            dp_masks.append(wrap([desc['state'][st]['rec_drop_prob']], self.device)
                            .expand([desc['state'][st]['dim']])
                            .bernoulli())

        out = []
        for step in range(inputs.shape[1]):
            h_next = self.cell(inputs[:, step], hidden, desc)
            for i in range(len(hidden)):
                if desc['state'][i]['rec_drop_prob'] > 0:
                    if self.use_zoneout:
                        if self.training:
                            h_next[i] = hidden[i] * dp_masks[i] + h_next[i] * (1 - dp_masks[i])
                        else:
                            h_next[i] = hidden[i] * desc['state'][i]['rec_drop_prob'] \
                                        + h_next[i] * (1 - desc['state'][i]['rec_drop_prob'])
                    else:
                        if self.training:
                            h_next[i] = h_next[i] * dp_masks[i] / desc['state'][i]['rec_drop_prob']
            hidden = h_next
            if len(out) == 0:
                for i in range(len(hidden)): out.append(hidden[i].unsqueeze(1))
            else:
                for i in range(len(hidden)):
                    out[i] = torch.cat([out[i], hidden[i].unsqueeze(1)], 1)

        return out if return_sequence else hidden


state_attrs = {
    'fontname': 'Roboto',
    'fontcolor': '#fcfcfc',
    'fillcolor': '#682dd8',
    'color': '#682dd8',
    'style': 'filled',
    'shape': 'box',
    'fontsize': 13,
    'margin': .2,
}
