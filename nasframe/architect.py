from nasframe.utils.misc import make_dirs, add_if_doesnt_exist, add_increment
from nasframe.utils.torch import wrap
from nasframe.searchspaces import SearchSpace

from torch.distributions import Categorical
from collections import defaultdict
from os.path import join, exists

import torch.nn as nn
import torch
import copy


class Architect(nn.Module):
    """
    Architect neural network.

    Args:
        search_space (SearchSpace): search space to which this architect instance belongs
        state_dims (tuple): graph encoder cell dimensions
        cell_type (str): 'lstm' or 'gru'; graph encoder cell type
    """
    def __init__(self, search_space, state_dims=(128, 128), cell_type='LSTM'):
        super().__init__()

        self.search_space = copy.deepcopy(search_space)
        self.search_space.reset()

        self.cells = []

        cell = nn.LSTMCell if cell_type.lower() == 'lstm' else nn.GRUCell
        for i, d in enumerate(state_dims[:-1]):
            self.cells.append(cell(d, state_dims[i+1]))

        self.cells = nn.ModuleList(self.cells)

        self.embedding_index = {'state_zero': 0}
        self.policies = {}
        self.values = {}

        self.initialize(self.search_space, [])

        self.embedding = nn.Embedding(len(self.embedding_index), state_dims[0])
        self._policy = nn.ModuleList(list(self.policies.values()))
        self._values = nn.ModuleList(list(self.values.values()))

    @property
    def is_cuda(self):
        """
        Returns bool value, indicating whether this architect is located on a CUDA device.
        """
        return next(self.parameters()).is_cuda

    @property
    def device(self):
        """
        Returns ``torch.Device`` on which this architect is located.
        """
        return next(self.parameters()).device

    def reset(self):
        """
        Initializes all parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
            else:
                nn.init.constant_(p, 0)

    def init_hidden(self, num_samples):
        """
        Initializes hidden by passing :math:`state_0` embedding and
        zero-initialized hidden state to graph encoder
        """
        states = []
        for cell in self.cells:
            num_states = 1 + 1*isinstance(cell, nn.LSTMCell)
            zeros = wrap(torch.zeros(num_samples, cell.hidden_size), self.device)
            states.append(tuple([zeros.clone()]*num_states))
        input = self.embedding(wrap([0], self.device, dtype=torch.long))
        states = self(input, states)
        return states

    def save(self, prefix='./', name=None, include_space=False):
        """
        Saves architect to a file.

        Args:
            prefix (str): prefix directory of a future save
            name (str): save name
            include_space (bool): whether to save the search space along with this architect instance

        Returns:
            str: path to which architect was saved (``prefix``/architect/``name``.pth)
        """
        name = name or 'unnamed'
        filename = f'{name}.pth'
        prefix = join(prefix, 'architect')
        if not exists(prefix): make_dirs(prefix)
        path = join(prefix, filename)

        space = self.search_space
        if not include_space:
            self.search_space = None
        torch.save(self, path)
        self.search_space = space
        return path

    def forward(self, input, hidden, key=None):
        """
        Perform a forward path through graph encoder.

        Returns:
            new hidden state if key is None
            else tuple of (logits, values, hidden)
        """
        for cell_id, cell in enumerate(self.cells):
            hidden[cell_id] = cell(input, hidden[cell_id])
            if not isinstance(hidden[cell_id], (list, tuple)):
                hidden[cell_id] = [hidden[cell_id]]
            input = hidden[cell_id][0]

        if key is not None:
            logits = self.policies[key](hidden[-1][0])
            values = self.values[key](hidden[-1][0])
            return logits, values, hidden

        return hidden

    def act(self, inputs, hidden, explore, key, output=None, pick=None):
        """
        Get action given encoded graph predicted to the moment.

        Args:
            inputs (torch.Tensor): encoded graph representation
            explore(bool): whether to explore or exploit only
            hidden (list): list of encoder cell(s) hidden states
            key (str): key corresponding to policy and value layers
            output (defaultdict(list), optional): if provided, chosen logprobs, chosen values and
                entropies will be appended to those lists, instead of being returned
            pick (int, optional): index of action to pick instead of sampling or argmax.

        Returns:
            action, hidden state, chosen logprobs, chosen values, entropies
        """
        assert key is not None, '`key` must not be None for act. ' \
                                'If you want to update hidden only, call `forward` instead'
        logits, values, hidden = self(inputs, hidden, key)
        distribution = Categorical(logits=logits)

        if pick is not None:
            assert pick < logits.size(1)
            action = wrap([pick], self.device, dtype=torch.long)
        elif explore:
            action = distribution.sample()
        else:
            action = distribution.probs.max(1)[1]

        chosen_logs = distribution.log_prob(action).unsqueeze(1)
        chosen_values = values.gather(1, action.unsqueeze(1))
        entropies = distribution.entropy()

        if output is not None:
            assert isinstance(output, defaultdict) and output.default_factory == list, \
                '`output` must be a defaultdict with `list` default_factory.'

            output['logprob'].append(chosen_logs)
            output['values'].append(chosen_values)
            output['entropies'].append(entropies)

            return action, hidden

        return action, hidden, chosen_logs, chosen_values, entropies

    def sample(self, explore=True):
        """
        Samples a graph description, using current policy.

        Args:
            explore: whether to explore or exploit only.

        Returns:
            tuple: sampled graph, log probabilities and predicted
                values of chosen actions, entropies of output distributions
        """
        output = defaultdict(list)
        representations = {}
        description = {}

        hidden = self.init_hidden(1)

        self._subsample(hidden, explore, self.search_space, [], representations, output, description)

        return (description, torch.cat(output['logprob']).transpose(0, 1),
                torch.cat(output['values']), torch.cat(output['entropies']))

    def evaluate_description(self, description):
        """
        Picks described actions in order to obtain log probabilities, predicted values
        and the entropies of output distributions

        Args:
            description (dict): description to evaluate

        Returns:
            tuple: sampled graph, log probabilities and predicted
                values of chosen actions, entropies of output distributions
        """
        description = copy.deepcopy(description)
        output = defaultdict(list)
        representations = {}

        hidden = self.init_hidden(1)

        self._subsample(hidden, False, self.search_space, [], representations, output, description)

        return (description, torch.cat(output['logprob']).transpose(0, 1),
                torch.cat(output['values']), torch.cat(output['entropies']))

    def _subsample(self, hidden, explore, search_space, names, representations, output, description, outer_i=None):
        """
        The recursive workhorse of `sample` method.

        Args:
            hidden (list): previous encoder hidden state
            explore (bool): whether to explore the search/action space or exploit only
            search_space (SearchSpace): current level of search space
            names (list): names of search spaces up to current level
            output (defaultdict(list), optional): dict of lists to append outputs to
            description (dict): description being generated
            outer_i (int): outer iteration ordinal (used in recursion, `None` on depth `0`)

        Returns:
            list: next hidden state
        """
        name = search_space.name
        names = copy.deepcopy(names)
        names.append(name)
        # Those checks were introduced to reuse this method to evaluate model output for already existing description.
        if description.get(name) is None:
            description[name] = {}

        # region num inner prediction
        num_inner = self.search_space.eval_(search_space.num_inner, **locals())

        # region forcing facilitation
        if description.get(f'num_{name}') is None:
            forced_inner = self.search_space.eval_(search_space.forced_num_inner, **locals())
            max_available = max(num_inner) if isinstance(num_inner, (list, tuple)) else num_inner
            assert forced_inner is None or isinstance(forced_inner, int) and 0 < forced_inner <= max_available

            if forced_inner is not None:
                try: forced_inner = num_inner.index(forced_inner)
                except ValueError:
                    raise ValueError(f'Number of inner search spaces "{forced_inner}" '
                                      'is not present in original search space.')
        else:
            forced_inner = num_inner.index(description[f'num_{name}'])
        # endregion

        index = self.embedding_index[f'{name}_start']
        index = wrap([index], self.device, dtype=torch.long)
        input = self.embedding(index)

        if len(num_inner) > 1:
            key = f'{"_".join(names[:-1])}_{len(num_inner)}_{name}s'
            action, hidden = self.act(input, hidden, explore, key, output, forced_inner)
            num_inner = num_inner[action.item()]
        else:
            hidden = self(input, hidden)
            num_inner = num_inner[forced_inner] if forced_inner is not None else num_inner[0]

        if description.get(f'num_{name}') is None:
            description[f'num_{name}'] = num_inner
        # endregion

        # region inner space prediction
        index = self.embedding_index[f'{num_inner}_{name}s']
        index = wrap([index], self.device, dtype=torch.long)
        input = self.embedding(index)

        encoded_flag = False
        for i in range(int(num_inner)):
            if description[name].get(i) is None:
                description[name][i] = {}

            if isinstance(search_space.inner, dict):
                for k, v in search_space.inner.items():
                    v = self.search_space.eval_(v, **locals())
                    key = f'{"_".join(names[:-1])}_{len(v)}_{k}s'

                    if isinstance(v, (list, tuple)) and len(v) > 1:
                        pick = description[name][i].get(k)
                        if pick is not None:
                            try: pick = v.index(pick)
                            except ValueError:
                                raise ValueError(f'Point "{pick}" is not present in '
                                                 f'{k} dimension of the search space.')

                        action, hidden = self.act(input, hidden, explore, key, output, pick)

                        choice = v[action.item()]

                        if pick is None: description[name][i][k] = choice
                        else: assert choice == description[name][i][k]

                        if k == 'id':
                            if choice in representations:
                                input = representations[choice]
                                continue

                        index = self.embedding_index[f'{k}_{choice}']
                        index = wrap([index], self.device, dtype=torch.long)
                        input = self.embedding(index)
                    else:
                        if description[name][i].get(k) is None:
                            description[name][i][k] = v[0]
                        else: assert v[0] == description[name][i][k]

            else:
                assert isinstance(search_space.inner, (list, tuple, SearchSpace)), \
                    'Inner search space must be either dict, SearchSpace or list of SearchSpaces.'

                if not encoded_flag:
                    hidden = self(input, hidden)
                    encoded_flag = True

                spaces = [search_space.inner] if isinstance(search_space.inner, SearchSpace) else search_space.inner
                for space in spaces:
                    input = self._subsample(hidden, explore, space, names, representations,
                                            output, description[name][i], i)
                    hidden = self(input[-1][0], hidden)

        index = self.embedding_index[f'{name}_inner_done']
        index = wrap([index], self.device, dtype=torch.long)
        input = self.embedding(index)
        # endregion

        # region outer keys prediction
        for k, v in search_space.outer.items():
            v = self.search_space.eval_(v, **locals())
            key = f'{"_".join(names[:-1])}_{len(v)}_{k}s'

            if isinstance(v, (list, tuple)) and len(v) > 1:
                pick = description.get(k)
                if pick is not None:
                    try: pick = v.index(pick)
                    except ValueError:
                        raise ValueError(f'Point "{pick}" is not present in '
                                         f'{k} dimension of the search space.')

                action, hidden = self.act(input, hidden, explore, key, output, pick)

                choice = v[action.item()]

                if pick is None: description[k] = choice
                else: assert choice == description[k]

                if k == 'id':
                    if choice in representations:
                        input = representations[choice]
                        continue

                index = self.embedding_index[f'{k}_{choice}']
                index = wrap([index], self.device, dtype=torch.long)
                input = self.embedding(index)
            else:
                if description[name][i].get(k) is None:
                    description[name][i][k] = v[0]
                else: assert v[0] == description[name][i][k]
        # endregion

        index = self.embedding_index[f'{name}_end']
        index = wrap([index], self.device, dtype=torch.long)
        input = self.embedding(index)

        hidden = self(input, hidden)
        if len(names) > 2:
            repr_key = f'{names[-2]}' if outer_i is None else f'{names[-2]}_{outer_i}'
            representations[repr_key] = hidden[-1][0]
        return hidden

    def initialize(self, search_space, names, outer_i=None):
        """
        Recursively initializes architect, creating value and policy layers and embedding index.

        Args:
            search_space (SearchSpace): current level of search space
            names (list): names of search spaces up to current level
            outer_i (int): outer iteration ordinal (used in recursion, `None` on depth `0`)

        """
        name = search_space.name
        names = copy.deepcopy(names)
        names.append(name)
        output_dim = self.cells[-1].hidden_size

        num_inner = self.search_space.eval_(search_space.num_inner, **locals())
        if len(num_inner) > 1:
            key = f'{"_".join(names[:-1])}_{len(num_inner)}_{name}s'
            add_if_doesnt_exist(self.policies, key, nn.Linear(output_dim, len(num_inner)))
            add_if_doesnt_exist(self.values, key, nn.Linear(output_dim, len(num_inner)))

        add_increment(self.embedding_index, f'{name}_start')
        add_increment(self.embedding_index, f'{name}_end')

        self.adapt(search_space.outer.items(), names, outer_i)

        for i in range(max(num_inner)):
            add_increment(self.embedding_index, f'{i+1}_{name}s')
            if isinstance(search_space.inner, (list, tuple)):
                for space in search_space.inner: self.initialize(space, names, i)
            elif isinstance(search_space.inner, SearchSpace):
                self.initialize(search_space.inner, names, i)
            else:
                assert isinstance(search_space.inner, dict), \
                    'Inner search space must be either list, dict or SearchSpace.'
                self.adapt(search_space.inner.items(), names, outer_i)
        add_increment(self.embedding_index, f'{name}_inner_done')

    def adapt(self, items, names, outer_i):
        """
        Creates layers and updates embedding index given search space dimensions.

        Args:
            items (iterable): iterable of tuples corresponding to search space dimensions
            names (list): names of search spaces up to current level
            outer_i (int): outer iteration ordinal (used in recursion, `None` on depth `0`)

        Returns:

        """
        output_dim = self.cells[-1].hidden_size
        for k, v in items:
            v = self.search_space.eval_(v, **locals())
            if isinstance(v, (list, tuple)) and len(v) > 1:
                key = f'{"_".join(names[:-1])}_{len(v)}_{k}s'
                add_if_doesnt_exist(self.policies, key, nn.Linear(output_dim, len(v)))
                add_if_doesnt_exist(self.values, key, nn.Linear(output_dim, len(v)))
                for v_ in v:
                    add_increment(self.embedding_index, f'{k}_{v_}')
