import numpy as np
from collections import defaultdict
from nasframe.utils import convert_keys_to_int, wrap

import torch
import json


class Storage:
    """
    Class for storing all the data, needed for architect training.
    """
    def __init__(self):
        self.descriptions = []
        self.log_probs = []
        self.pred_values = []
        self.entropies = []
        self.rewards = []
        self.advantages = []
        self.param_counts = []

    def find(self, description):
        """
        Finds ``description`` in self.

        Returns:
            index if description is stored in this instance of Storage, None otherwise
        """
        if description in self.descriptions:
            return self.descriptions.index(description)
        else:
            return None

    def append(self, description, log_probs, pred_values, entropies, reward=None, param_count=None):
        """
        Appends data to the storage.
        """
        self.descriptions.append(description)
        self.log_probs.append(wrap(log_probs))
        self.pred_values.append(wrap(pred_values))
        self.entropies.append(wrap(entropies))
        self.rewards.append(wrap(reward))
        self.param_counts.append(wrap(param_count))
        self._on_change()

    def reward(self, description, amount):
        """
        Sets or overwrites the reward for given description.
        """
        index = self.descriptions.index(description)
        self.rewards[index] = torch.tensor(amount)
        self._on_change()

    def update(self, description, log_probs, pred_values, entropies):
        """
        Updates the associated description with new log probs, predicted values and entropies.
        """
        if not isinstance(description, dict):
            raise ValueError('Description must be a dict.')

        index = self.descriptions.index(description)
        self.log_probs[index] = log_probs.detach()
        self.pred_values[index] = pred_values.detach()
        self.entropies[index] = entropies.detach()
        self._on_change()

    def _update_advantages(self):
        """
        Calculates advantages from rewards and predicted values.
        """
        advantages = list(map(lambda i: (i[0].expand_as(i[1])-i[1]).view(-1),
                              zip(self.rewards, self.pred_values)))
        adv = torch.cat(advantages)
        isnan = adv != adv
        adv = adv[~isnan]
        if adv.numel() > 0: mean, std = adv.mean(), adv.std()
        else: mean, std = [np.nan]*2
        self.advantages = list(map(lambda a: ((a-mean)/(std+1e-5)).detach(), advantages))

    def _on_change(self):
        """
        Should be called on data change.
        """
        self._update_advantages()

    def filter_na(self):
        """
        Filters out items with ``reward != reward``.
        """
        isnan = enumerate(map(lambda i: i != i, self.rewards))
        isnan = map(lambda i: i[0], filter(lambda i: i[1], isnan))

        for index in list(isnan)[::-1]:
            del self[index]

    def best(self):
        """
        Returns description and corresponding best reward.
        """
        self.filter_na()
        idx = int(np.argmax(self.rewards))
        return self.descriptions[idx], self.rewards[idx]

    @staticmethod
    def from_json(arch=None, path=None, data=None, input_shape=None):
        """
        Reads description-reward pairs from json and constructs a Storage from it.

        Args:
            arch (Architect): architect instance to collect needed data
            path (str): path to json file
            data (list): pre-loaded list of description-reward pairs
            input_shape (list, tuple, torch.Size): shape of the input

        Returns:
            Storage: storage instance containing data from given json
        """
        if not ((path is None) ^ (data is None)):
            raise ValueError('Either path or json must be not None, but not both.')

        if path is not None:
            with open(path) as f:
                data = json.load(f)

        storage = Storage()
        data = list(map(lambda i: (convert_keys_to_int(i[0]), i[1]), data))

        for description, reward in data:
            if arch is not None:
                _, logps, values, entropies = arch.evaluate_description(description)

                param_count = None
                if input_shape is not None:
                    desc = arch.search_space.preprocess(description, input_shape)
                    if desc is not None:
                        param_count = sum(arch.search_space.parameter_count(desc)[:2]) / 1e6
            else:
                logps, values, entropies, param_count = [None]*4

            storage.append(description, logps, values, entropies, reward, param_count)

        return storage

    def to_json(self, path):
        """
        Dumps descriptions and corresponding rewards into a JSON-formatted file on ``path``.
        """
        with open(path, 'w+') as f:
            json.dump(list(zip(self.descriptions, map(lambda t: t.item(), self.rewards))), f)

    def __getitem__(self, item):
        if isinstance(item, (int, np.int_)):
            return (self.descriptions[item], self.log_probs[item],
                    self.pred_values[item], self.entropies[item],
                    self.rewards[item].expand_as(self.log_probs[item]),
                    self.advantages[item], self.param_counts[item])
        if hasattr(item, '__iter__'):
            return list(map(self.__getitem__, item))
        if isinstance(item, slice):
            return self[range(*item.indices(len(self)))]
        else:
            raise ValueError(f'Cannot index Storage with {type(item)}.')

    def __delitem__(self, item):
        if isinstance(item, int):
            del self.descriptions[item]
            del self.log_probs[item]
            del self.pred_values[item]
            del self.entropies[item]
            del self.rewards[item]
            del self.param_counts[item]
        else:
            raise ValueError(f'Cannot delete with {type(item)} as index.')
        self._on_change()

    def __len__(self):
        return len(self.descriptions)


class CurriculumStorage:
    """
    Class for storing all the data, needed for architect curriculum training.

    Args:
        max_complexity: maximum complexity of the curriculum.
    """
    def __init__(self, max_complexity):
        self.storages = defaultdict(Storage)
        self.current_complexity = 1
        self.max_complexity = max_complexity

    @property
    def current_storage(self):
        """
        Returns storage corresponding to current complexity.
        """
        return self.storages[self.current_complexity]

    @property
    def descriptions(self):
        """
        Returns descriptions corresponding to current complexity.
        """
        return self.current_storage.descriptions

    @property
    def log_probs(self):
        """
        Returns :math:`log(probs)` corresponding to current complexity.
        """
        return self.current_storage.log_probs

    @property
    def pred_values(self):
        """
        Returns predicted values corresponding to current complexity.
        """
        return self.current_storage.pred_values

    @property
    def entropies(self):
        """
        Returns entropies corresponding to current complexity.
        """
        return self.current_storage.entropies

    @property
    def rewards(self):
        """
        Returns rewards corresponding to current complexity.
        """
        return self.current_storage.rewards

    @property
    def advantages(self):
        """
        Returns advantages corresponding to current complexity.
        """
        return self.current_storage.advantages

    @property
    def param_counts(self):
        """
        Returns parameter counts corresponding to current complexity.
        """
        return self.current_storage.param_counts

    def flatten(self):
        """
        Flattens current instance into a Storage instance, which contains all the data from current instance.
        """
        flat = Storage()
        for storage in self.storages.values():
            for point in storage:
                description, logps, values, entropies,\
                reward, _, param_count = point
                reward = reward.mean().item()
                flat.append(description, logps, values, entropies, reward, param_count)
        return flat

    def set_complexity(self, n):
        """
        Sets current curriculum complexity to ``n``.
        """
        n = min(self.max_complexity, n)
        self.current_complexity = n

    def reward(self, description, ammount):
        """
        Sets or overwrites the reward for given description.
        """
        self.current_storage.reward(description, ammount)

    def append(self, description, log_probs, pred_values, entropies, reward=None, parameter_count=None):
        """
        Appends data to the storage.
        """
        self.current_storage.append(description, log_probs, pred_values, entropies, reward, parameter_count)

    def update(self, description, log_probs, pred_values, entropies):
        """
        Updates the associated description with new log probs, predicted values and entropies.
        """
        self.current_storage.update(description, log_probs, pred_values, entropies)

    def find(self, description):
        """
        Finds ``description`` in self.

        Returns:
            dict: mapping of levels to indices
        """
        results = {}
        for i in range(1,self.max_complexity+1):
            if description in self.storages[i].descriptions:
                results[i] = self.storages[i].descriptions.index(description)
        return results

    @staticmethod
    def from_json(arch=None, path=None, data=None, input_shape=None, max_complexity=None):
        """
        Reads description-reward pairs from json and constructs a Storage from it.

        Args:
            arch (Architect): architect instance to collect needed data
            path (str): path to json file
            data (dict): dictionary with pre-loaded data
            input_shape (list, tuple, torch.Size): shape of the input
            max_complexity (int): maximum complexity of constructed storage

        Returns:
            CurriculumStorage: storage instance containing data from given json
        """
        if not ((path is None) ^ (data is None)):
            raise ValueError('Either path or json must be not None, but not both.')

        if path is not None:
            with open(path) as f:
                data = json.load(f)

        assert isinstance(data, dict)
        data = convert_keys_to_int(data)

        storage = CurriculumStorage(max_complexity or max(data.keys()))

        for level in data:
            storage.storages[level] = Storage.from_json(
                arch=arch, data=data[level], input_shape=input_shape)

        return storage

    def to_json(self, path):
        """
        Dumps descriptions and corresponding rewards into a JSON-formatted file on ``path``.
        """
        storages = dict(self.storages)

        for level in storages:
            descriptions = storages[level].descriptions
            rewards = map(lambda t: t.item(), storages[level].rewards)
            storages[level] = list(zip(descriptions, rewards))

        with open(path, 'w+') as f:
            json.dump(storages, f)

    def best(self):
        """
        Returns description and corresponding best reward.
        """
        best = map(lambda s: s.best(), self.storages.values())
        return max(best, key=lambda i: i[1])

    def filter_na(self):
        """
        Filters out items with ``reward != reward`` across levels.
        """
        for storage in self.storages.values():
            storage.filter_na()

    def __getitem__(self, item):
        return self.current_storage[item]

    def __delitem__(self, item):
        del self.current_storage[item]

    def __len__(self):
        return len(self.current_storage)