from nasframe.utils.misc import *
from nasframe.utils.torch import *

from collections import OrderedDict
from os.path import exists, join
from flatten_dict import flatten
from itertools import chain
from time import time
from torch import nn

import pygraphviz as gv
import numpy as np
import subprocess
import torch
import copy
import gc


class SearchSpace(nn.Module):
    r"""
    Base search space, from which all search spaces should be derived.

    Arguments:
        name (str): the name of search space
        outer (dict): dimensions which can contain only one point
        num_inner (list): possible numbers of points inner dimensions can contain
        inner (dict, list): inner dimensions, which can contain up to ``max(num_inner)`` points
        device (int, str, torch.Device): if ``int``, should be in [-1, num_visible_gpus);
            if ``str``, should abide torch.device naming convention.
        input_picking_method (str): method used for picking inputs, when requested input names are not available;
            choose from ``{'first', 'last' or ignore'}``;
            ``'set'`` can be prepended (or appended), to avoid duplicating
            input names in request.
        u_gate_fallback (str): fallback input combination method, when update gate
            (i.e. :math:`\sigma(u)*\alpha + (1-\sigma(u))*\beta`) cannot be applied
        shape_resolution_method (str): method to resolve absolute shapes given relative to the
            ``{'min', 'max' or 'mean'}`` input absolute shape, across all unit inputs
        add_unnecessary_layers (bool): if units particular input shape and unit shape match and this
            is set to ``False``, layer between them won't be added
        add_layer_after_combine (bool): if combined input shape matches unit shape and this
            is set to ``False``, layer after input combination won't be added
        reuse_parameters (bool): whether to save parameters across descriptions; can reduce training times
        data_parallel (bool): whether or not ``prepare`` method should return data parallel version of this space
        logger (Logger): logger instance
        bias (bool): whether layers should contain biases
    """
    def __init__(self, name, outer, num_inner, inner, device=-1, **kwargs):

        super().__init__()

        self.name = name
        self.outer = outer
        self.num_inner = num_inner
        self.inner = inner
        self.forced_num_inner = None

        self._indicator = nn.Parameter(wrap([0.], device))

        self.u_gate_fallback = kwargs.get('u_gate_fallback', 'avg').strip().lower()
        self.input_picking_method = kwargs.get('input_picking_method', 'set_last').strip().lower()
        self.shape_resolution_method = kwargs.get('shape_resolution_method', 'min').strip().lower()
        self.add_unnecessary_layers = kwargs.get('add_unnecessary_layers', False)
        self.add_layer_after_combine = kwargs.get('add_layer_after_combine', True)
        self.reuse_parameters = kwargs.get('reuse_parameters', True)
        self.data_parallel = kwargs.get('data_parallel', False)
        self.logger = kwargs.get('logger', None)
        self.bias = kwargs.get('bias', True)

        self.layers = nn.ModuleList()

        self.usage_count = None
        self.last_used = None
        self.layer_index = None
        self.layer_created = None

        self.dimension_names = {}

    @property
    def is_cuda(self):
        """
        Returns bool value, indicating whether this search space is located on a CUDA device.
        """
        return self._indicator.is_cuda

    @property
    def device(self):
        """
        Returns ``torch.Device`` on which this search space is located.
        """
        return self._indicator.device

    def merge(self, other):
        """
        Merges this search space with another.
        Useful only is ``self.reuse_parameters`` is ``True``.
        If ``other`` space instance has a layer which this one doesn't, such layers get's added to this space instance.
        If both ``other`` and ``self`` contain same layers, their parameters get averaged and written to this space'
        parameters.
        """
        new_count, modified_count = 0, 0
        flat_index = flatten(self.layer_index)
        for key in flatten(other.layer_index):
            if key in flat_index:
                l1 = self.get_layer(*key)
                l2 = other.get_layer(*key)

                assert l1.weight.shape == l2.weight.shape

                l1.weight.data.add_(l2.weight.data)
                l1.weight.data.div_(2)

                if self.bias:
                    l1.bias.data.add_(l2.bias.data)
                    l1.bias.data.div_(2)

                self.layers[self.get_index(*key)] = l1
                modified_count += 1

                usage_count = other._get_value(other.usage_count, *key)
                usage_count += self._get_value(self.usage_count, *key)

                creation_time = other._get_value(other.layer_created, *key)
                creation_time = min(creation_time, self._get_value(self.layer_created, *key))

                last_used = other._get_value(other.last_used, *key)
                last_used = max(last_used, self._get_value(self.last_used, *key))
            else:
                layer = other.get_layer(*key)
                self._set_value(self.layer_index, len(self.layers), *key)
                self.layers.append(layer)

                usage_count = other._get_value(other.usage_count, *key)
                creation_time = other._get_value(other.layer_created, *key)
                last_used = other._get_value(other.last_used, *key)
                new_count += 1

            self._set_value(self.usage_count, usage_count, *key)
            self._set_value(self.layer_created, creation_time, *key)
            self._set_value(self.last_used, last_used, *key)

        self.log_info(f'Space was merged with another. {new_count} new layers have been added; '
                      f'{modified_count} modified.')

    def log_info(self, str):
        """
        Logs info via ``self.logger.info`` if ``self.logger`` is not None.
        """
        if self.logger is not None:
            self.logger.info(str)

    # region Curriculum related stuff
    def force_num_inner(self, n):
        """
        Signals architect, that number of points in the inner space should always be ``n``.
        """
        max_inner = max(self.eval_(self.num_inner))
        num_inner = min(n, max_inner)
        if n > max_inner:
            self.log_info(f'Attempted to force num_{self.name} to be {n}, '
                          f'but max_{self.name} is {max_inner}, forcing nodes to {num_inner}.')
        else:
            self.log_info(f'num_{self.name} forced to be {num_inner}.')
        self.forced_num_inner = num_inner

    def release_num_inner(self):
        """
        Releases constraints set on number of points in inner search space dimensions.
        """
        self.forced_num_inner = None
        self.log_info(f'num_{self.name} released.')

    def set_curriculum_complexity(self, i):
        """
        Should implement constraining certain parameters to match curriculum complexity ``i``.
        """
        raise NotImplemented()

    def release_all_constraints(self):
        """
        Releases all constraints set by calling ``set_curriculum_complexity``.
        """
        raise NotImplemented()
    # endregion

    # region Graph description processing
    def preprocess(self, description, input_shape, connect_leafs=False):
        """
        Description preprocessing pipeline.
        Consists of filling the gaps, then connecting, then connecting leaf units if ``connect_leafs``,
        then pruning and, if graph is viable after all that, resolving shapes.

        Args:
            description (dict): description to preprocess
            input_shape (iterable): shape of the input, which will be used ``descriptions``
            connect_leafs (bool): wether to include ``connect_leafs`` in the pipeline

        Returns:
            dict: preprocessed description.
        """
        description = self.fill_the_gaps(description)
        description = self.connect(description)
        if connect_leafs:
            description = self.connect_leafs(description)
        description, viable = self.prune(description)

        if not viable: return None

        description = self.resolve_shapes(description, input_shape)
        return description

    @staticmethod
    def _fill_the_gaps(description, space, **kwargs):
        """
        Recursively fills gaps in the description.

        If a certain search space dimension has only one possible value, it is ignored by architect and thus
        is absent in descriptions generated by it. This method fills such description 'gaps' with that only value.
        Args:
            description (dict): graph description
            space (SearchSpace): space or sub-space to fill gaps in
        Returns:
             dict: graph description with default values filled in.
        """
        description = copy.deepcopy(description)

        spaces = space if isinstance(space, (list,tuple)) else [space]

        for space in spaces:
            if isinstance(space, SearchSpace):
                for k, dim in space.outer.items():
                    dim = space.eval_(dim, **kwargs)
                    if not isinstance(dim, (list, tuple)):
                        add_if_doesnt_exist(description, k, dim)
                    elif len(dim) < 2:
                        add_if_doesnt_exist(description, k, dim[0])

                unit_type = space.name
                if isinstance(space.inner, (list, tuple, SearchSpace)):
                    for i in description[unit_type].keys():
                        description[unit_type][i] = \
                            SearchSpace._fill_the_gaps(description[unit_type][i], space.inner, outer_i=i)
                elif isinstance(space.inner, dict):
                    assert isinstance(description[unit_type], dict), 'Description mustn\'t be connected for this method'
                    for i in description[unit_type].keys():
                        for k, dim in space.inner.items():
                            dim = space.eval_(dim, **kwargs)
                            if not isinstance(dim, (list, tuple)):
                                add_if_doesnt_exist(description[unit_type][i], k, dim)
                            elif len(dim) < 2:
                                add_if_doesnt_exist(description[unit_type][i], k, dim[0])

        return description

    @staticmethod
    def connect_leafs(description):
        """
        Connects leaf nodes, which would be pruned otherwise, to the output.
        """
        raise NotImplemented()

    def _connect(self, description, space, picking_method='set_random', input_expr=None):
        """
        Resolves graph's connections, by picking requested input names
        from available input names using ``picking_method``.

        Args:
            description (dict): description to be connected
            space (SearchSpace): search space corresponding to the description passed
            picking_method (str): name of the input picking method
            input_expr (str): expression which needs to be evaluated in order to obtain
                the list of available inputs

        Returns:
            dict: connected description

        """
        unit_type = space.name
        for outer_i in range(description[f'num_{unit_type}']):
            unit = description[unit_type][outer_i]
            unit_name = f'{unit_type}_{outer_i}'

            if input_expr is None:
                available_inputs = space.eval_(space.inner['id'], **locals())
            else:
                available_inputs = space.eval_(input_expr, **locals())

            requested_inputs = list(map(lambda v: v['id'], unit['input'].values()))
            chosen_inputs = []

            if 'set' in picking_method:
                requested_inputs = list(set(requested_inputs))

            unsatisfied = []
            for input_name in requested_inputs:
                if input_name in available_inputs:
                    idx = available_inputs.index(input_name)
                    chosen_inputs.append(available_inputs.pop(idx))
                else:
                    unsatisfied.append(input_name)

            if len(available_inputs) > 0 and len(unsatisfied) > 0:
                for input_name in unsatisfied:
                    input_name = self.pick(input_name, available_inputs, picking_method)
                    if input_name is not None:
                        chosen_inputs.append(input_name)

            unit['input'] = chosen_inputs
            unit['num_input'] = len(chosen_inputs)

        return description

    @staticmethod
    def pick(input_name, avail, picking_method):
        """
        Picks one input name from the ``avail`` list, using ``picking_method``.
        Pops chosen input name from ``avail``.

        Args:
            input_name (str): name of the requested input
            avail (list): names of available inputs
            picking_method (str): name of the picking method to be used

        Returns:
            str: name of the picked input, or None if ``avail`` is empty or requested input name is not available
            and ``picking_method == 'ignore'``.
        """
        if len(avail) > 0:
            if input_name in avail:
                return avail.pop(avail.index(input_name))
            if 'first' in picking_method:
                return avail.pop(0)
            elif 'last' in picking_method:
                return avail.pop(len(avail)-1)
            elif 'ignore' in picking_method:
                return None
            else:
                raise NotImplementedError(f"Picking method '{picking_method}' is not implemented.")
        else: return None

    @staticmethod
    def prune(description):
        """
        Returns modified copy of the ``description``, which doesn't contain unused units and boolean value
        indicating whether the description is viable (i.e. contains used units).
        """
        raise NotImplemented()

    @staticmethod
    def _prune(description, unit_type, mapping):
        """
        Filters ``description`` with respect to ``unit_type`` given ``mapping`` of the names of useful units.
        ``mapping`` should be a ``dict`` with old names as keys and new names as values.
        """
        for unit_id in list(description[unit_type].keys()):
            unit_name = f'{unit_type}_{unit_id}'
            if unit_name not in mapping:
                del description[unit_type][unit_id]

        for unit_id in list(description[unit_type].keys()):
            unit_name = f'{unit_type}_{unit_id}'

            inpt = description[unit_type][unit_id]['input']
            inpt = list(filter(lambda i: i is not None, map(mapping.get, inpt)))

            description[unit_type][unit_id]['input'] = inpt
            description[unit_type][unit_id]['num_input'] = len(inpt)

            new_name = mapping[unit_name]
            if unit_name != new_name:
                new_id = int(new_name.split('_')[1])
                description[unit_type][new_id] = description[unit_type][unit_id]
                del description[unit_type][unit_id]

        description[f'num_{unit_type}'] = len(description[unit_type].keys())

    @staticmethod
    def _walk_graph(description, unit_type, persistent_units):
        """
        Walks described graph backwards to determine which units are reachable from ``persistent_units``.

        Args:
            description (dict): description of a graph.
            unit_type (str): unit type w.r.t. to which results should be computed
            persistent_units (dict): mapping of node names useful a priori to sets of their inputs

        Returns:
            dict: mapping of unit names to unit names reachable from them
        """

        if isinstance(description[unit_type][0]['input'], dict):
            raise ValueError("The description is not connected. "
                             "Please pass the result of `connect` method next time.")

        reachable_units = dict(persistent_units)

        for unit_id in sorted(description[unit_type].keys()):
            unit_name = f'{unit_type}_{unit_id}'
            unit_input = set(description[unit_type][unit_id]['input'])
            for input_name in set(unit_input):
                unit_input.update(reachable_units.get(input_name, set()))
            reachable_units[unit_name] = unit_input

        return reachable_units

    @staticmethod
    def _resolve_shapes(desc, input, method='min', unit_type='node'):
        """
        Recursively convert relative shapes to absolute.

        Args:
            desc: description of a graph
            input: graphs input tensor or its shape
            method: method to resolve shape ambiguity; either 'min', 'max' or 'mean'
            unit_type: type of unit to resolve shapes for

        :return: description with shapes resolved.
        """
        desc = copy.deepcopy(desc)
        if torch.is_tensor(input):
            input_dim = tuple(input.shape)[-1]
        elif isinstance(input, (tuple, list, torch.Size)):
            input_dim = tuple(input)[-1]
        else:
            raise ValueError(f'Input must be a Tensor or shape, got {type(input)}.')

        if isinstance(desc[unit_type][0]['input'], dict):
            raise ValueError("The description is not connected. "
                             f"Please pass the result of `connect` method next time.")

        desc['input_dim'] = input_dim
        for id in desc[unit_type].keys():
            unit_name = f'{unit_type}_{id}'
            SearchSpace._resolve_shape(unit_name, desc, method)

        return desc

    @staticmethod
    def _resolve_shape(unit_name, desc, method):
        """
        Recursive call of ``_resolve_shapes`` method.

        Args:
            unit_name: name of the unit to resolve shape for
            desc: description of a graph
            method: method to resolve shape ambiguity; either 'min', 'max' or 'mean'

        Returns:
            int: node's absolute output dim.
        """
        unit_type = unit_name.split('_')[0]
        if unit_name == 'graph_input':
            return desc['input_dim']

        unit_id = int(unit_name.split('_')[1])
        if desc[unit_type][unit_id].get('dim') is not None:
            return desc[unit_type][unit_id]['dim']

        unit = desc[unit_type][unit_id]
        mapped = list(map(lambda n: SearchSpace._resolve_shape(n, desc, method), unit['input']))

        if unit['inp_combine'] == 'concat':
            func = sum
        elif method == 'min':
            func = min
        elif method == 'max':
            func = max
        elif method == 'mean':
            func = np.mean
        else:
            raise NotImplementedError(f'Method {method} is not implemented.')

        unit['dim'] = int(max(1, int(func(mapped) * unit['out_size'])))
        return unit['dim']

    def draw(self, description, path):
        """
        Draws a graph given description.

        Args:
            path: path to which the resulting image should be saved.
            description: description of the graph to draw.
        """
        try:
            make_dirs(os.path.dirname(path))
        except: pass
        graph = gv.AGraph(directed=True, strict=True, rankdir='LR')

        self._add_agraph_units(description, graph)

        graph.layout(prog='dot')
        graph.draw(path)

    def _add_agraph_units(self, description, graph):
        """
        Adds units to ``AGraph``.

        Args:
            description (dict): description of a graph
            graph (gv.AGraph): pygraphviz graph to add units to

        Returns:
            gv.AGraph: graph with units added
        """
        raise NotImplemented()

    # endregion

    # region Layer operations
    def get_layer(self, *key, constructor=None, **kwargs):
        """
        Creates a layer or returns existing.

        Args:
            *key: key of the layer in (possibly nested) layer_index
            constructor (callable): constructor for the layers that doesn't exists, doesn't take any arguments

        Returns:
            nn.Module: layer in question.
        """
        index = self.get_index(*key)
        if index is None:
            layer = constructor().to(self.device)
            self._init_layer(layer)
            self._set_value(self.layer_index, len(self.layers), *key)
            self.layers.append(layer)
            self._set_value(self.layer_created, time(), *key)
        else:
            self.move_layer(*key)
            layer = self.layers[index]
        prev_use_count = self._get_value(self.usage_count, *key)
        self._set_value(self.usage_count, prev_use_count+1, *key)
        self._set_value(self.last_used, time(), *key)
        return layer

    def get_index(self, *key):
        """
        Convenience method which gets layer index given ``key``.
        """
        return self._get_value(self.layer_index, *key)

    def layer_exists(self, *key, **kwargs):
        """
        Returns True, if layer with identity ``key`` exists.
        """
        return key in flatten(self.layer_index)

    def delete_layer(self, *key, **kwargs):
        """
        Deletes layer with identity ``key``.
        """
        index = self.get_index(*key)

        self._del_value(self.layer_index, *key)
        self._del_value(self.layer_created, *key)
        self._del_value(self.usage_count, *key)
        self._del_value(self.last_used, *key)
        del self.layers[index]

    def move_layer(self, *key, device=None):
        """
        Moves layer specified by ``key`` to ``device``. If ``device`` is not specified, defaults to ``self.device``.
        """
        index = self.get_index(*key)

        device = device or self.device
        if self.layers[index].weight.device != device:
            self.layers[index] = self.layers[index].to(device)
            return True
        return False

    @staticmethod
    def _get_value(dictionary, *key):
        """
        Gets value specified by ``key`` from a (possibly nested) ``dictionary``.
        """
        for k in key[:-1]:
            dictionary = dictionary[k]
        if isinstance(dictionary, defaultdict):
            return dictionary[key[-1]]
        else:
            return dictionary.get(key[-1])

    @staticmethod
    def _set_value(dictionary, value, *key):
        """
        Sets ``value`` corresponding to ``key`` in a (possibly nested) ``dictionary``.
        """
        for k in key[:-1]:
            dictionary = dictionary[k]
        dictionary[key[-1]] = value

    @staticmethod
    def _del_value(dictionary, *key):
        """
        Deletes (last level of) ``key`` from a (possibly nested) ``dictionary``.
        """
        for k in key[:-1]:
            dictionary = dictionary[k]
        del dictionary[key[-1]]

    def _init_layer(self, module, **kwargs):
        """
        Initializes module weights and bias, if present.

        Args:
            module (nn.Module): module whose parameters should be initialized
            weight_init (callable): inplace initialization function for parameters with ``ndim > 1``
            bias_init (callable): inplace initialization function for parameters with ``ndim <= 1``
        """
        weight_init = kwargs.get('weight_init', nn.init.xavier_normal_)
        bias_init = kwargs.get('bias_init', partial(nn.init.constant_, val=0))
        for p in module.parameters():
            if p.dim() > 1:
                weight_init(p)
            else:
                bias_init(p)
    # endregion

    # region Parameter and memory management

    def parameter_count(self, description, create_nonexistent=False):
        """
        Returns tuple of existing, new and total existing parameter counts of the model described by ``description``
        and a list of keys corresponding to used layers.
        if ``create_nonexistent`` is True, creates described layers along the way.
        """
        raise NotImplemented()

    def _count_related_params(self, desc, space_name, create_nonexistent=False):
        """
        Returns tuple of (existing and new) parameter counts of the model described by ``desc``
        and a ``dict`` with of keys of used layers.
        if ``create_nonexistent`` is True, creates described layers along the way.
        """
        raise NotImplemented()

    def prepare(self, input, description, freeze=False, drop_unused=False):
        """
        Prepares search space to evaluate description.

        Args:
            input (torch.Tensor): tensor containing sample input
            description (dict): description of a graph
            freeze (bool): whether to freeze the model using ``torch.jit.trace`` (experimental)
            drop_unused (bool): whether to delete unused in the description layers

        Returns:
            nn.Module: model ready for forwarding data through
        """
        assert torch.is_tensor(input) or isinstance(input, np.ndarray)
        # if input.shape[-1] != description['input_dim']:
        #     raise ValueError('Shapes were resolved for different input shape.')

        # Create needed layers
        if not self.reuse_parameters: self.reset()
        used_keys = self.parameter_count(description, create_nonexistent=True)[-1]

        if drop_unused:
            unused = set(flatten(self.layer_index)).difference(set(used_keys))
            for k in unused:
                self.delete_layer(*k)
            self.log_info(f'Deleted {len(unused)} layers, {len(self.layer_index)} remain.')

        for key in used_keys:
            self.move_layer(*key)

        if freeze:
            @torch.jit.trace(input)
            def f(input, description=description):
                return self(input, description)
            model = f
        else:
            model = self

        if self.data_parallel: return nn.DataParallel(model)
        else: return model

    def reset(self):
        """
        Resets search space to initial state.
        """
        raise NotImplemented()

    def save(self, prefix='./', name=None):
        """
        Saves search space to path composed of ``prefix``/``self.name``/``name``.pth.

        Returns:
            str: save location
        """
        name = name or 'unnamed'
        filename = f'{name}.pth'
        prefix = join(prefix, f'{self.name}')
        if not exists(prefix): make_dirs(prefix)
        path = join(prefix, filename)

        logger = self.logger
        self.logger = None
        torch.save(self, path)
        self.logger = logger

        return path
    # endregion

    def combine_tensors(self, op_name, tensors):
        """
        Combines tensor list into one tensor with given op.

        Args:
            op_name: name of the operation to perform on tensor list in order to combine it's elements
            tensors: list of tensors to combine

        Returns:
            torch.Tensor: combined tensors
        """
        if len(tensors) > 1:
            if op_name == 'add':
                return torch.stack(tensors,0).sum(0)
            if op_name == 'mult':
                return torch.stack(tensors,0).prod(0)
            if op_name == 'avg':
                return torch.stack(tensors, 0).mean(0)
            if op_name == 'concat':
                return torch.cat(tensors, -1)
            if op_name == 'u_gate':
                if len(tensors) >= 3:
                    u = nn.functional.sigmoid(tensors[0])
                    return u*tensors[1] + (1-u)*tensors[2]
                else:
                    return self.combine_tensors(self.u_gate_fallback, tensors)
        else:
            return tensors[0]

    def forward(self, *input):
        """
        Call ``prepare`` first, and use ``forward`` method of the returned module instead.
        """
        raise NotImplemented()

    @staticmethod
    def eval_(val, **context):
        """
        Evaluates strings and returns everything else unchanged.
        """
        locals().update(context)
        if isinstance(val, str):
            return eval(val)
        else:
            return copy.deepcopy(val)

    def __repr__(self, indent=1):
        string = '  '*(indent-1) + f'{self.name} search space:\n'
        for k, v in self.outer.items():
            k = self.dimension_names.get(k, k)
            string += '  '*indent + f'{k}: {v},\n'

        if isinstance(self.inner, dict):
            string += '  ' * indent + f"{self.num_inner}x{{\n"
            for k, v in self.inner.items():
                k = self.dimension_names.get(k, k)
                string += '  ' * indent + f'    {k}: {v!r},\n'
            string += '  ' * indent + '}\n'
        else:
            assert isinstance(self.inner, (list, tuple, SearchSpace))
            space = self.inner if isinstance(self.inner, (list, tuple)) else [self.inner]
            for s in space:
                string += '  ' * indent + f"{self.num_inner}x{{\n"
                string += s.__repr__(indent=indent+1)

        return string


size_map = {
    torch.FloatTensor: 32,
    torch.DoubleTensor: 64,
    torch.CharTensor: 8,
    torch.ByteTensor: 8,
    torch.ShortTensor: 16,
    torch.IntTensor: 32,
    torch.LongTensor: 64,
}

input_attrs = {
    'fontname': 'Roboto',
    'fontcolor': '#ffffff',
    'fillcolor': '#2d77d8',
    'color': '#2d77d8',
    'style': 'filled',
    'shape': 'box',
    'fontsize': 13,
    'margin': .3,
}

op_attrs = {
    'fontname': 'Roboto',
    'fontcolor': '#353535',
    'fillcolor': '#ffc6dc',
    'color': '#ffc6dc',
    'style': 'filled',
    'shape': 'oval',
    'fontsize': 12,
    'margin': .05,
}

node_attrs = {
    'fontname': 'Roboto',
    'fontcolor': '#383838',
    'fillcolor': '#f4f4f4',
    'color': '#828282',
    'style': 'filled',
    'shape': 'box',
    'fontsize': 13,
    'margin': .2,
}
