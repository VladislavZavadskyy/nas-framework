from .base import *


class MLPSpace(SearchSpace):
    """
    Defines a multilayer perceptron search space.

    Args:
        max_nodes (int): maximum number of layers in a graph
        out_sizes (list, tuple): possible numbers of output dims relative to the number of input dims (num_out/num_in)
        input_combine_ops (list, tuple): names of operations to combine inputs, if there's more than one
        activations (list, tuple): names of nonlinearities to apply to layer outputs
        drop_probs (list, tuple): possible node input dropout probabilities
        max_node_inputs (int, str, optional): maximum number of layers to be used as input or expression
            to be evaluated, which returns this number. If not specified, defaults to maximum possible number of inputs.
        kwargs: keyword arguments to be passed to SearchSpace constructor.
    """

    def __init__(self, max_nodes, out_sizes=(.2, .5, 1, 2, 5),
                 input_combine_ops=('add', 'mult', 'avg', 'u_gate', 'concat'),
                 activations=('relu', 'sigmoid', 'tanh'),
                 drop_probs=(0., 0.1, 0.3, 0.5), max_node_inputs=None,
                 **kwargs):

        if isinstance(max_node_inputs, int):
            max_node_inputs = f'min({max_node_inputs}, outer_i+2)'
        elif max_node_inputs is None:
            max_node_inputs = f'list(range(1, outer_i+2))'

        super().__init__(name='node',
                         outer={},
                         num_inner=int_to_range(max_nodes, 1),
                         inner=[SearchSpace(
                             name='input',
                             outer=OrderedDict({
                                 'inp_combine': input_combine_ops,
                                 'out_size': out_sizes,
                                 'activation': activations,
                                 'drop_prob': drop_probs,
                             }),
                             num_inner=int_to_range(max_node_inputs, 1),
                             inner={
                                 'id': f'["graph_input"] + ["node_%d"%i for i in range(outer_i)]',
                             })],
                         **kwargs)

        self.dimension_names = {
            'drop_prob': 'Dropout',
            'activation': 'Activation',
            'inp_combine': 'Input combine',
            'out_size': 'Size multiplier',
            'dim': 'Dimension',
        }

        self.reset()

    # region Curriculum related stuff
    def force_num_nodes(self, n):
        """
        Signals architect, that number of nodes should always be ``n``.
        """
        self.force_num_inner(n)

    def force_num_node_inputs(self, n):
        """
        Signals architect, that number of node inputs should always be ``n``.
        """
        self.inner.force_num_inner(n)

    def release_num_nodes(self):
        """
        Releases constraints set on number of nodes.
        """
        self.release_num_inner()

    def release_num_node_inputs(self):
        """
        Releases constraints set on number of node inpus.
        """
        self.inner.release_num_inner()

    def set_curriculum_complexity(self, i):
        """
        Sets curriculum complexity ``i``.
        """
        self.force_num_inner(i)

    def release_all_constraints(self):
        """
        Releases all constraints set by ``force*`` or ``set_curriculum_complexity`` methods.
        """
        self.release_num_node_inputs()
        self.release_num_nodes()

    # endregion

    # region Layer ops
    def get_layer(self, input_name, unit_name, shape, *args, **kwargs):
        """
        Creates a layer or returns existing.

        Args:
            input_name: input node id.
            unit_name: layer node id.
            shape: weight shape (e.g. tuple of (input_node_dim, output_node_dim)).

        Returns:
            nn.Linear: layer in question.
        """
        key = input_name, unit_name, shape
        constructor = partial(nn.Linear, shape[0], shape[1], self.bias)
        return super().get_layer(*key, constructor=constructor)

    def delete_layer(self, input_name, unit_name, shape, *args, **kwargs):
        """
        Deletes layer.

        Args:
            input_name: input node id.
            unit_name: layer node id.
            shape: weight shape (e.g. tuple of (input_node_dim, output_node_dim)).

        """
        key = input_name, unit_name, shape
        self.delete_layer(*key)

        self.log_info(f'Layer {input_name} -> {unit_name} ({shape}) has been deleted.')

    def move_layer(self, input_name, unit_name, shape, device=None, **kwargs):
        """
        Moves layer to ``device``.

        Args:
            input_name: input node id.
            unit_name: layer node id.
            shape: weight shape (e.g. tuple of (input_node_dim, output_node_dim)).
            device (torch.Device): device to move layer to, if not specified defaults to ``self.device``
        """
        key = input_name, unit_name, shape
        if super().move_layer(*key, device=device):
            self.log_info(f'Layer {input_name} -> {unit_name} ({shape}) moved to {str(device)}.')

    def get_index(self, input_name, unit_name, shape):
        """
        Returns index of layer specified by ``input_name``, ``unit_name`` and ``shape`` is ``self.layers``.
        """
        key = input_name, unit_name, shape
        return self._get_value(self.layer_index, *key)

    def layer_exists(self, input_name, unit_name, shape, *args, **kwargs):
        """
        Returns layer's existence status.
        """
        key = input_name, unit_name, shape
        return super().layer_exists(*key)

    def _init_layer(self, module, **kwargs):
        super()._init_layer(module,
                            weight_init=nn.init.xavier_normal_,
                            bias_init=partial(nn.init.constant_, val=0))

    # endregion

    # region Parameter and memory management
    def parameter_count(self, description, create_nonexistent=False):
        """
        Returns tuple of existing, new and total existing parameter counts of the model described by ``description``
        and a list of keys corresponding to used layers.
        if ``create_nonexistent`` is True, creates described layers along the way.
        """
        if description['node'][0].get('dim') is None:
            raise ValueError("The shapes are not resolved. "
                             "Please pass the result of `MLPSpace.resolve_shapes` as description next time.")

        (existing, new), used_dict = self._count_related_params(description, 'node', create_nonexistent)

        total_existing = 0
        for module in self.layers:
            total_existing += param_count(module)

        return existing, new, total_existing, list(flatten(used_dict))

    def _count_related_params(self, desc, space_name, create_nonexistent=False):
        """
        Returns tuple of (existing and new) parameter counts of the model described by ``desc``
        and a ``dict`` with of keys of used layers.
        if ``create_nonexistent`` is True, creates described layers along the way.
        """
        param_count = [0, 0]  # 0: existing, 1: new
        used_dict = nested_defaultdict(2, dict)

        for unit_id, unit in desc[space_name].items():
            unit_name = f'{space_name}_{unit_id}'
            total_input_shape = 0

            for input_name in unit['input']:
                shape = (self._resolve_shape(input_name, desc, 'None'), unit['dim'])

                if (shape[0] != shape[1] or self.add_unnecessary_layers) \
                        and unit['inp_combine'] != 'concat':

                    is_new = 1 - self.layer_exists(input_name, unit_name, shape)
                    param_count[is_new] += np.prod(shape) + shape[-1] * self.bias

                    used_dict[input_name][unit_name][shape] = True

                    if is_new and create_nonexistent:
                        self.get_layer(input_name, unit_name, shape)

                total_input_shape += shape[0]

            if self.add_layer_after_combine or unit['inp_combine'] == 'concat':
                input_shape = total_input_shape if unit['inp_combine'] == 'concat' else unit['dim']
                shape = (input_shape, unit['dim'])

                is_new = 1 - self.layer_exists('combine', unit_name, shape)
                param_count[is_new] += np.prod(shape) + shape[-1] * self.bias

                used_dict['combine'][unit_name][shape] = True

                if is_new and create_nonexistent:
                    self.get_layer('combine', unit_name, shape)

        return param_count, used_dict

    def reset(self):
        """
        Resets search space to initial state.
        """
        self.usage_count = nested_defaultdict(3, value=0)
        self.last_used = nested_defaultdict(2, dict)
        self.layer_index = nested_defaultdict(2, dict)
        self.layer_created = nested_defaultdict(2, dict)

        self.layers = nn.ModuleList()

        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()

    # endregion

    # region Description processing
    def _add_agraph_unit(self, graph: gv.AGraph, unit, name, combine_op: str):
        """
        Adds a single unit to ``gv.Agraph``.

        Args:
            graph: graph to add unit to.
            unit: unit description.
            name: unit name.
            combine_op: unit's input combination op.

        """
        if len(unit['input']) <= 1:
            graph.add_edge(unit['input'][0], name)
        else:
            if combine_op == 'u_gate':
                if len(unit['input']) < 3:
                    return self._add_agraph_unit(graph, unit, name, self.u_gate_fallback)
                elif len(unit['input']) > 3:
                    unit = copy.deepcopy(unit)
                    unit['input'] = unit['input'][:3]
            op_node = '_'.join(map(str, set(unit['input']))) + combine_op
            graph.add_node(op_node, label=combine_op, **op_attrs)
            for idx, input_name in enumerate(unit['input']):
                if combine_op == 'u_gate':
                    graph.add_edge(input_name, op_node, label='u' if not idx else idx)
                else:
                    graph.add_edge(input_name, op_node)
            graph.add_edge(op_node, name)

    def _add_agraph_units(self, description, graph: gv.AGraph):
        """
        Adds units to ``AGraph``.

        Args:
            description (dict): description of a graph
            graph (gv.AGraph): pygraphviz graph to add units to

        Returns:
            gv.AGraph: graph with units added
        """
        input_dim = ''
        if description.get('input_dim') is not None:
            input_dim = f'<BR/>Dimension: {description["input_dim"]}'
        graph.add_node(n='graph_input', label=f'<Input<sub>t</sub>{input_dim}>', **input_attrs)

        for i in range(description['num_node']):
            node = description['node'][i]
            node_name = f'node_{i}'

            label = f'<'
            for k, v in node.items():
                if k in ['input', 'num_input']:
                    continue
                if k == 'inp_combine':
                    self._add_agraph_unit(graph, node, node_name, v)
                    continue

                label += f'{self.dimension_names[k]}: {v}<BR/>'
            label += '>'
            graph.add_node(n=node_name, label=label, **node_attrs)

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
        return SearchSpace._fill_the_gaps(description, self)

    def resolve_shapes(self, desc, input, method=None, **kwargs):
        """
        Converts relative shapes to absolute.

        Args:
            desc: description of a graph.
            input: graphs input or its shape.
            method: method to resolve shape ambiguity; either 'min', 'max' or 'mean'

        Returns:
            dict: description with shapes resolved.
        """
        method = method or self.shape_resolution_method
        return SearchSpace._resolve_shapes(desc, input, method, 'node')

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

        picking_method = picking_method or self.input_picking_method
        node_inputs = f'["graph_input"] + ["node_%d"%i for i in range(outer_i)]'

        super()._connect(description, self, picking_method, node_inputs)
        return description

    @staticmethod
    def connect_leafs(description):
        """
        Connects leaf nodes, which otherwise would be pruned, to the output.
        """
        desc = copy.deepcopy(description)
        _, useful_nodes = MLPSpace.walk_graph(desc)
        leaf_nodes = set(desc['node'].keys()).difference(set(useful_nodes))
        last_node = desc['node'][max(desc['node'].keys())]
        last_node['input'] = list(set(last_node['input']).union(leaf_nodes))
        return desc

    @staticmethod
    def prune(description):
        """
        Returns modified copy of the ``description``, which doesn't contain unused units and boolean value
        indicating whether the description is viable (i.e. contains used units).
        """
        depends_on_input, useful_nodes = MLPSpace.walk_graph(description)
        if not depends_on_input: return {}, False
        desc = copy.deepcopy(description)

        mapping = {'graph_input': 'graph_input'}
        mapping.update(
            map(lambda i: (f'node_{i[1]}', f'node_{i[0]}'), enumerate(useful_nodes))
        )

        SearchSpace._prune(desc, 'node', mapping)

        viable = depends_on_input and desc['num_node'] > 0
        viable &= all(map(lambda n: len(n['input']) > 0, desc['node'].values()))
        return desc, viable

    @staticmethod
    def walk_graph(description):
        """
        Walks described graph bakwards to determine which nodes are connected
        to the last one (which is assumed to be output).

        Args:
            description (dict): description of a graph.

        Returns:
            tuple: ((bool) whether graph depends on input,
                    (set) of useful node ids)
        """
        persistent_nodes = {'graph_input': set()}
        reachable_nodes = SearchSpace._walk_graph(description, 'node', persistent_nodes)

        useful_nodes = filter(lambda i: 'graph_input' in i[1], reachable_nodes.items())
        useful_nodes = set(map(lambda i: int(i[0].split('_')[1]), useful_nodes))

        output_node_id = max(description['node'].keys())
        depends_on_input = output_node_id in useful_nodes

        return depends_on_input, useful_nodes

    # endregion

    def forward(self, inputs, description):
        """
        Does one forward pass through the described graph.
        Call ``prepare`` first, and use ``forward`` method of the returned module instead.

        Args:
            inputs (torch.Tensor): inputs to be forwarded
            description (dict): description of a graph to forward through
        """
        if isinstance(description['node'][0]['input'], dict):
            raise ValueError("The description is not connected. "
                             "Please pass the result of `MLPSpace.connect(description)` next time.")

        assert torch.is_tensor(inputs), f'How dare you pass {type(inputs)} to me?'
        assert inputs.dim() == 2, 'Input shape must be [batch, input_dim]'

        inputs = inputs.to(self.device)
        outputs = {'graph_input': inputs}

        for node_id in range(description['num_node']):
            node = description['node'][node_id]
            node_name = f'node_{node_id}'

            interm_outs = []
            for input_name in node['input']:
                shape = (outputs[input_name].shape[-1], node['dim'])
                if (shape[0] != shape[1] or self.add_unnecessary_layers) \
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
                shape = (outputs[node_name].shape[-1], node['dim'])
                layer = self.get_layer('combine', node_name, shape)
                outputs[node_name] = layer(outputs[node_name])

            outputs[node_name] = get_activation(node['activation'])(outputs[node_name])

        return outputs[node_name]  # Assuming last node_name belongs to the last node
