import time

from flatten_dict import flatten

from nasframe.utils.torch import *
from nasframe import FeedForwardCoach
from torch.utils.data import DataLoader
from nasframe.utils import make_dirs, Bunch, get_tqdm
from nasframe.utils import get_logger, FileLock, logger
from nasframe import Architect, ArchitectCoach
from nasframe.coaches.base import LossIsNoneError
from nasframe.storage import CurriculumStorage, Storage
from nasframe.searchspaces import get_space_type

from tensorboardX import SummaryWriter
from functools import partial, wraps
from os.path import exists

from shutil import rmtree
from os.path import join

from PIL import Image

import gc
import os
import yaml
import json

import multiprocessing as mp


class GenericModel(nn.Module):
    """
    Generic model template to be used as a wrapper for search space.

    Args:
        space (MLPSpace): search space being wrapped

    """
    def __init__(self, space):
        super().__init__()
        self.space = space
        self.cell = None

    @property
    def device(self):
        """
        Returns:
            ``torch.Device`` on which ``self.space`` is located.
            Should be also the device of any other instance's module.

        """
        return self.space.device

    def prepare(self, inputs, description):
        """
        A wrapper for the ``space.prepare`` call.
        Useful when inputs passed to this wrapper have a shape different from the one of inputs passed to search space.

        """
        self.cell = self.space.prepare(inputs, description)
        return self

    def forward(self, inputs, description):
        self.cell(inputs, description)


def hash_description(description):
    """
    Attempts to create a consistent description representation and hash it.
    Usually fails.

    """
    description = json.loads(json.dumps(description))
    flat_desc = flatten(description)
    for k in list(flat_desc):
        if isinstance(flat_desc[k], (list, tuple)):
            flat_desc[k] = frozenset(flat_desc[k])
        new_k = str(tuple(map(str, k)))
        flat_desc[new_k] = str(flat_desc[k])
        del flat_desc[k]
    tuples = frozenset(flat_desc.items())
    return hash(tuples)


def worker_init(description, log_dir):
    """
    Initializes a worker process.

    Args:
        description (dict): description for the worker to evaluate
        log_dir (str): path to a directory where logs shall be stored

    Returns:
        logger: logger instance
        summary_writer (SummaryWriter): summary writer assigned to this worker
        description_dir (str): path of a directory, to which passed description has been assigned

    """
    hashsum = hash_description(description)
    description_dir = join(log_dir, 'descriptions', str(hashsum))

    if exists(description_dir):
        return description, 'exists'
    make_dirs(description_dir)

    with open(join(description_dir, 'description.json'), 'w+') as f:
        json.dump(description, f)

    logger = get_logger(f'description_{hashsum}', join(description_dir, 'training.log'))
    summary_writer = SummaryWriter(description_dir)

    return logger, summary_writer, description_dir


def prepare(config, resume, input_shape, gpu_idx, num_gpus):
    """
    Initializes training session.

    Args:
        config (dict, str): configuration dictionary or path to YAML file.
        resume (boot): whether to attempt to resume a previous session
        input_shape (list, tuple, torch.Size): shape of the inputs
        num_gpus (int): number of GPUs to use
        gpu_idx (str, optional): string of comma separated dpu indices. if ``None``, ``range(num_gpus)`` is used.

    Returns:
        Architect and an ArchitectCoach
    """
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f)

    log_dir = config['log_dir']
    if not resume:
        for path in os.listdir(log_dir):
            if path == 'model.pth':
                continue
            path = join(log_dir, path)
            if os.path.isdir(path):
                rmtree(path)
            else:
                os.remove(path)

    make_dirs(join(log_dir, 'descriptions'))

    if exists(join(log_dir, 'architect')):
        for path in os.listdir(join(log_dir, 'architect')):
            if path.startswith('events.out.tfevents'):
                os.remove(join(log_dir, 'architect', path))
    else:
        make_dirs(join(log_dir, 'architect'))

    type_name = config['searchspace'].pop('type')
    space_proto = get_space_type(type_name)(**config['searchspace'])

    arch_logger = get_logger('arch_coach', join(log_dir, 'architect', 'training.log'))
    load_architect = config['architect_training']['load_architect'] and resume

    if exists(join(log_dir, 'architect', 'checkpoint.pth')) and load_architect:
        arch = torch.load(join(log_dir, 'architect', 'checkpoint.pth'))
        arch.search_space = space_proto
        arch_logger.info('Restored architect from checkpoint.')
    else:
        arch = Architect(space_proto)

    curriculum = config['architect_training'].get('curriculum', False)

    max_complexity = config['architect_training'].get('max_curriculum_complexity')
    assert max_complexity is not None

    summary_writer = SummaryWriter(join(log_dir, 'architect'))

    if not exists(join(log_dir, 'description_reward.json')):
        with open(join(log_dir, 'description_reward.json'), 'w+') as f:
            json.dump({} if curriculum else [], f)

    if resume:
        if curriculum:
            storage = CurriculumStorage.from_json(arch=arch, path=join(log_dir, 'description_reward.json'),
                                                  space=space_proto, input_shape=input_shape,
                                                  max_complexity=max_complexity)

            flat_storage = storage.flatten()
            arch_logger.info(f'Loaded {len(flat_storage)} descriptions, '
                             f'{len(storage.storages)} curriculum levels total.')
        else:
            storage = Storage.from_json(arch=arch, path=join(log_dir, 'description_reward.json'), space=space_proto,
                                        input_shape=input_shape)
            arch_logger.info(f'{len(storage)} descriptions loaded.')

            flat_storage = storage

        hashums = list(map(hash_description, flat_storage.descriptions))
        for d in os.listdir(join(log_dir, 'descriptions')):
            try:
                if int(d) not in hashums: rmtree(join(log_dir, 'descriptions', d))
            except ValueError:
                continue

        for path in os.listdir(join(log_dir, 'descriptions')):
            if path.startswith('events.out.tfevents'):
                os.remove(join(log_dir, 'descriptions', path))

        description_writer = SummaryWriter(join(log_dir, 'descriptions'))
        for i, hashsum in enumerate(hashums):
            description_dir = join(log_dir, 'descriptions', str(hashsum))
            if not exists(description_dir):
                os.mkdir(description_dir)
                description = flat_storage.descriptions[i]
                desc = space_proto.preprocess(description, input_shape)
                space_proto.draw(desc, join(description_dir, 'graph.png'))

                img = np.array(Image.open(join(description_dir, 'graph.png')))
                description_writer.add_image(f'descriptions/{hashsum}', img)

                with open(join(description_dir, 'description.json'), 'w+') as f:
                    json.dump(description, f)
    else:
        storage = CurriculumStorage(max_complexity) if curriculum else Storage()

    arch_coach = ArchitectCoach(arch, storage,
                                scheduler=None,
                                logger=arch_logger,
                                epochs_to_increase=np.inf,
                                tensorboard=summary_writer,
                                **config['architect_training'])

    if not exists(join(log_dir, space_proto.name, 'merged_space.pth')):
        space_proto.save(log_dir, 'merged_space')

    FileLock(join(log_dir, space_proto.name, 'merged_space.pth.lock')).purge()
    FileLock(join(log_dir, 'description_reward.json')).purge()

    gpu_indices = range(num_gpus) if gpu_idx is None else map(int, gpu_idx.split(','))
    return config['architect_training'], space_proto, arch, arch_coach, list(gpu_indices), log_dir


def sample_loop(arch, storage, space_proto, input_shape,
                desired_storage_len, append_deterministic=True):
    """
    Description sampling loop.

    Samples description given an architect and a space prototype, then perprocesses them by applying
    ``space_proto.preprocess`` method with default parameters, if description is viable after that,
    appends it to the list and puts it into storage. Loop stops when the storage length reaches
    ``desired_storage_len``.

    Args:
        arch (Architect): architect instance
        storage (Storage, CurriculumStorage): storage instance
        space_proto (SeachSpace): search space prototype
        input_shape (list, tuple, torch.Size): input shape
        desired_storage_len (int): the length storage should have when the loop is completed
        append_deterministic (bool): whether to append description obtained by deterministically choosing the actions
            with the highest probability in the architect's output distribution. If it would be the only description in
            resulting evaluation list, this argument will be ignored.

    Returns:
        A list of sampled but not evaluated descriptions.
    """
    evaluation_list = []
    while len(storage) < desired_storage_len:
        description, logps, values, entropies = arch.sample()
        desc = space_proto.preprocess(description, input_shape)
        if desc is not None:
            num_param = sum(space_proto.parameter_count(desc)[:2]) / 1e6
            storage.append(description, logps, values, entropies, None, num_param)
            evaluation_list.append(description)

    deterministic_is_viable = False
    if append_deterministic and len(evaluation_list) > 0:
        description, logps, values, entropies = arch.sample(explore=False)
        desc = space_proto.preprocess(description, input_shape)

        if desc is not None:
            num_param = sum(space_proto.parameter_count(desc)[:2]) / 1e6
            storage.append(description, logps, values, entropies, None, num_param)
            evaluation_list.append(description)
            deterministic_is_viable = True

    return evaluation_list, deterministic_is_viable


def evaluate(descriptions, worker_fn, gpu_indices):
    """
    Launch the workers to evaluate descriptions in parallel.

    Args:
        descriptions (list): descriptions to be evaluated
        worker_fn (callable): worker callable
        gpu_indices (list): list of gpu indices to run on

    Returns:
        list of (description, reward, device_idx) tuples
    """
    if len(descriptions) == 0: return []

    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:

        device_queue = manager.Queue()
        for idx in gpu_indices:
            device_queue.put(idx)

        worker_fn = partial(worker_fn, device_queue=device_queue)
        with ctx.Pool(len(descriptions), maxtasksperchild=1) as pool:
            result = pool.map(worker_fn, descriptions)

    return result


def train_curriculum(config, worker, input_shape,
                     resume, num_gpus, gpu_idx):
    """
    Curriculum architect training procedure.
    Includes sampling descriptions with complexity :math:`i`, evaluating them, train architect and start over again
    with complexity :math:`i+1`.
    When ``max_compexity`` is reached, disables curriculum, flattens :class:`CurriculumStorage` and trains plainly
    from then on.

    Args:
        config (dict, str): configuration dictionary or path to YAML file.
        worker (callable): worker callable
        input_shape (tuple, list, torch.Size): input's shape
        resume (bool): whether  to try resuming the previous session
        num_gpus (int): number of GPUs to use
        gpu_idx (str, optional): string of comma separated dpu indices. if ``None``, ``range(num_gpus)`` is used.

    Returns:
        On keyboard interrupt returns storage filled with all that's benn found and evaluated.
    """
    training_config, space_proto, arch, arch_coach, gpu_indices, log_dir =\
        prepare(config, resume, input_shape, gpu_idx, num_gpus)

    load_architect = training_config['load_architect'] and resume
    epochs_per_loop = training_config['epochs_per_loop']
    architect_lr_decay = training_config['lr_decay']
    assert 0 < architect_lr_decay < 1

    storage_surplus_factor = training_config.get('storage_surplus_factor', 1)
    assert storage_surplus_factor >= 1

    storage = arch_coach.storage
    summary_writer = arch_coach.summary_writer

    loops = 0
    points_per_epoch = arch_coach.batch_size * arch_coach.epoch_steps

    while True:
        try:
            curriculum_complexity = loops + 1
            desired_storage_len = len(storage) + points_per_epoch

            if curriculum_complexity <= storage.max_complexity:

                arch_coach.stats.curriculum_complexity = curriculum_complexity
                arch.search_space.set_curriculum_complexity(curriculum_complexity)
                storage.set_complexity(curriculum_complexity)
                desired_storage_len = points_per_epoch*storage_surplus_factor

            elif curriculum_complexity == storage.max_complexity + 1:

                curriculum_complexity = 0
                arch_coach.curriculum = False

                arch.search_space.release_all_constraints()
                arch_coach.storage = arch_coach.storage.flatten()

            else:

                curriculum_complexity = 0

            evaluation_list, deterministic_is_viable = sample_loop(
                arch, storage, space_proto, input_shape, desired_storage_len)

            worker_fn = partial(worker, current_complexity=curriculum_complexity)
            result = evaluate(evaluation_list, worker_fn, gpu_indices)
            logger.debug(f'Architect training: Evaluated {len(result)} descriptions.')

            accuracies = []
            for description, reward in result:
                if reward == 'exists':
                    for level, index in storage.find(description).items():
                        r = storage.storages[level].rewards[index]
                        if not np.isnan(r.mean().item()):
                            reward = r.mean().item()
                            accuracies.append(reward)
                            storage.reward(description, float(reward))
                elif reward is not None:
                    accuracies.append(reward)
                    storage.reward(description, float(reward))
            logger.debug(f'Architect training: Updated storage with {len(accuracies)} items.')

            # Since last reward corresponds to deterministic description
            if len(result) > 0:
                if result[-1][1] is None:
                    deterministic_is_viable = False

                if deterministic_is_viable:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies[:-1]), loops)
                    summary_writer.add_scalar('deterministic_acc', accuracies[-1], loops)
                else:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies), loops)
                    logger.debug(f'Architect training: Deterministic description is not viable.')

            storage.filter_na()
            if len(storage) < desired_storage_len and curriculum_complexity != 0:
                logger.debug(f'Architect training: not enough samples evaluated, rerunning the loop.')
                continue

            try:
                if len(result) > 0 or not load_architect:
                    logger.debug(f'Architect training: beginning the training.')
                    arch_coach.train(epochs_per_loop)
                    arch.save(log_dir, 'checkpoint')
                arch_coach.decay_lr(architect_lr_decay)
                loops += 1
            except ValueError as e:
                if 'Storage does not contain enough' in str(e):
                    continue

        except KeyboardInterrupt:
            return storage


def train_plain(config, worker, input_shape,
                resume, num_gpus, gpu_idx):
    """
        Architect training procedure.
        Includes sampling descriptions, evaluating them, train architect and start over again
        with complexity i+1.

        Args:
            config (dict, str): configuration dictionary or path to YAML file.
            worker (callable): worker callable
            input_shape (tuple, list, torch.Size): input's shape
            resume (bool): whether  to try resuming the previous session
            num_gpus (int): number of GPUs to use
            gpu_idx (str, optional): string of comma separated dpu indices. if ``None``, ``range(num_gpus)`` is used.

        Returns:
            On keyboard interrupt returns storage filled with all that's benn found and evaluated.
        """
    training_config, space_proto, arch, arch_coach, gpu_indices, log_dir =\
        prepare(config, resume, input_shape, gpu_idx, num_gpus)

    load_architect = training_config['load_architect']
    epochs_per_loop = training_config['epochs_per_loop']
    architect_lr_decay = training_config['lr_decay']
    assert 0 < architect_lr_decay < 1

    storage = arch_coach.storage
    summary_writer = arch_coach.summary_writer

    loops = 0
    points_per_epoch = arch_coach.batch_size * arch_coach.epoch_steps

    while True:
        try:
            desired_storage_len = len(storage) + points_per_epoch
            evaluation_list, deterministic_is_viable = sample_loop(
                arch, storage, space_proto, input_shape, desired_storage_len)

            result = evaluate(evaluation_list, worker, gpu_indices)

            accuracies = []
            for description, reward in result:
                if reward == 'exists':
                    index = storage.find(description)
                    r = storage.rewards[index]
                    if not np.isnan(r.mean().item()):
                        reward = r.mean().item()
                        accuracies.append(reward)
                        storage.reward(description, float(reward))
                elif reward is not None:
                    accuracies.append(reward)
                    storage.reward(description, float(reward))

            # Since last reward corresponds to deterministic description
            if len(result) > 0:
                if result[-1][1] is None:
                    deterministic_is_viable = False

                if deterministic_is_viable:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies[:-1]), loops)
                    summary_writer.add_scalar('deterministic_acc', accuracies[-1], loops)
                else:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies), loops)
                    summary_writer.add_scalar('deterministic_acc', -1, loops)

            storage.filter_na()
            if len(storage) < points_per_epoch:
                continue

            try:
                if len(result) > 0 or not load_architect:
                    arch_coach.train(epochs_per_loop)
                    arch.save(log_dir, 'checkpoint')
                arch_coach.decay_lr(architect_lr_decay)
                loops += 1
            except ValueError as e:
                if 'Storage does not contain enough' in str(e):
                    continue

        except KeyboardInterrupt:
            return storage


def acquire_device(worker):
    """
    Worker decorator which acquires a device from device queue and releases it when
    the worker is done working.

    Keyword Args:
        device_queue (Queue): a blocking queue with available devices.
    """
    @wraps(worker)
    def wrapper(*args, **kwds):
        device_queue = kwds.pop('device_queue')

        idx = device_queue.get()
        logger.debug(f'Acquired device {idx}.')
        result = worker(*args, **kwds, device_idx=idx)

        device_queue.put(idx)
        logger.debug(f'Released device {idx}.')
        return result
    return wrapper


@acquire_device
def generic_worker(description, device_idx, current_complexity, config, space_type, reward_metric):
    """
    Concurrent description evaluator.

    Args:
        description (dict): description to be evaluated
        device_idx (int): a dedicated CUDA devices index
        current_complexity (int): current curriculum complexity level.
        config (dict, str): configuration dictionary or path to YAML file.
        space_type (str): the name of root search space.
        reward_metric (str): key for the returned evaluation stats dictionary.

    Returns:
        Tuple of (description, mean ``reward_metric`` value).

        If loss is NaN, than mean ``reward_metric`` value = 0.

        If OOM was raised and ``not adaptive_batch_size`` or using ``min_batch_size`` causes OOM -- mean_auc=None.
    """
    description = dict(description)

    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f)

    log_dir = config['log_dir']
    data_dir = config['data_dir']

    logger, summary_writer, description_dir = worker_init(description, log_dir)
    logger.debug(f'Worker {device_idx}: initialization done.')

    config = config.get('child_training', config)

    batch_size = config.pop('batch_size')
    keep_data_on_device = config.pop('keep_data_on_device')
    adaptive_batch_size = config.pop('adaptive_batch_size')

    if adaptive_batch_size:
        min_batch_size = config.pop('min_batch_size')
        max_batch_size = config.pop('max_batch_size')
        batch_size_decay = config.pop('batch_size_decay')

        assert isinstance(min_batch_size, int)
        assert isinstance(max_batch_size, int)
        assert 0 < batch_size_decay < 1

        config['initial_lr'] *= max_batch_size / batch_size
        batch_size = max_batch_size
    else:
        min_batch_size = batch_size

    datasets = torch.load(join(data_dir, 'preprocessed.pth'))

    # A bit ugly, but should fix OOMs due to race conditions
    while True:
        try:
            with torch.cuda.device(device_idx):
                torch.cuda.init()
                break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.debug(f'Worker {device_idx}: got OOM during init, trying again in 1 second.')
                time.sleep(1)
            else:
                raise e

    with torch.cuda.device(device_idx):
        save_path = join(log_dir, space_type, 'merged_space.pth')
        model_path = join(log_dir, 'model.pth')

        model = torch.load(model_path)
        logger.debug(f'Worker {device_idx}: model loaded.')
        with FileLock(save_path):
            model.space = torch.load(save_path)
            logger.debug(f'Worker {device_idx}: search space loaded.')
            model = model.cuda()
            logger.debug(f'Worker {device_idx}: placed on device.')
        model.space.logger = logger

        desc = model.space.preprocess(description, (-1, model.space_input_size))

        model.space.draw(desc, join(description_dir, 'graph.png'))
        img = np.array(Image.open(join(description_dir, 'graph.png')))
        summary_writer.add_image('graph', img)

        if keep_data_on_device:
            for key in datasets.keys():
                datasets[key].tensors = tuple(map(
                    lambda t: t.cuda(device_idx), datasets[key].tensors))
            gc.collect()

        def cleanup():
            for f in os.listdir(description_dir):
                if f.startswith('events.out.tfevents'):
                    os.remove(join(description_dir, f))

        while min_batch_size <= batch_size:
            try:
                loaders = Bunch()
                for k in datasets.keys():
                    loaders[k] = DataLoader(datasets[k], batch_size, shuffle=True,
                                            pin_memory=not keep_data_on_device)

                space_coach = FeedForwardCoach(model, loaders,
                                               logger=logger,
                                               log_dir=log_dir,
                                               tensorboard=summary_writer,
                                               tqdm=get_tqdm(position=device_idx),
                                               **config)
                logger.debug(f'Worker {device_idx}: beginning training.')
                space_coach.train_until_convergence(description=desc)
                logger.debug(f'Worker {device_idx}: beginning evaluation')
                stats = space_coach.evaluate(desc, loaders.validation)
                mean_reward = np.mean(stats[reward_metric])

                with FileLock(save_path):
                    if exists(save_path):
                        other = torch.load(save_path)
                        model.space.merge(other.to(model.space.device))
                    model.space.cpu().save(log_dir, 'merged_space')

                with FileLock(join(log_dir, 'description_reward.json')):
                    with open(join(log_dir, 'description_reward.json'), 'r') as f:
                        existing = json.load(f)

                    if current_complexity is not None:
                        assert isinstance(existing, dict)
                        if str(current_complexity) not in existing:
                            existing[str(current_complexity)] = []

                        existing[str(current_complexity)].append([description, mean_reward])
                    else:
                        assert isinstance(existing, list)
                        existing.append([description, mean_reward])

                    with open(join(log_dir, 'description_reward.json'), 'w+') as f:
                        json.dump(existing, f)

                cleanup()
                return description, mean_reward

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if adaptive_batch_size:
                        batch_size = int(batch_size*batch_size_decay)
                        config['initial_lr'] *= batch_size_decay
                        logger.info(f'Out of memory, decreasing batch size to {batch_size}.')
                    else:
                        logger.info('Out of memory on fixed batch size. Terminating.')
                        cleanup()
                        return description, None
                else:
                    logger.error(e)
                    raise e

            except LossIsNoneError:
                logger.info('Loss is NaN. Terminating.')
                cleanup()
                return description, 0., device_idx

        else:
            logger.info('Out of memory with minimum batch size. Terminating.')
            cleanup()
            return description, None


def baseline_worker(device_idx, config, reward_metric, name):
    """
    Concurrent description evaluator.

    Args:
        device_idx (int): a dedicated CUDA devices index
        config (dict, str): configuration dictionary or path to YAML file.
        reward_metric (str): key for the returned evaluation stats dictionary.
        name (str): baseline name

    Returns:
        float: achieved ``reward_metric`` value
    """
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f)

    log_dir = config['log_dir']
    data_dir = config['data_dir']

    baseline_dir = join(log_dir, 'baselines', name)

    if exists(baseline_dir):
        rmtree(baseline_dir)
    make_dirs(baseline_dir)

    logger = get_logger(f'baseline_{name}', join(baseline_dir, 'training.log'))
    summary_writer = SummaryWriter(baseline_dir)

    logger.debug(f'Worker {device_idx}: initialization done.')

    config = config.get('child_training', config)

    batch_size = config.pop('batch_size')
    keep_data_on_device = config.pop('keep_data_on_device')
    adaptive_batch_size = config.pop('adaptive_batch_size')

    if adaptive_batch_size:
        min_batch_size = config.pop('min_batch_size')
        max_batch_size = config.pop('max_batch_size')
        batch_size_decay = config.pop('batch_size_decay')

        assert isinstance(min_batch_size, int)
        assert isinstance(max_batch_size, int)
        assert 0 < batch_size_decay < 1

        config['initial_lr'] *= max_batch_size / batch_size
        batch_size = max_batch_size
    else:
        min_batch_size = batch_size

    datasets = torch.load(join(data_dir, 'preprocessed.pth'))

    with torch.cuda.device(device_idx):
        model_path = join(log_dir, 'model.pth')

        model = torch.load(model_path)
        logger.debug(f'Worker {device_idx}: model loaded.')

        model = model.cuda()
        logger.debug(f'Worker {device_idx}: placed on device.')

        if keep_data_on_device:
            for key in datasets.keys():
                datasets[key].tensors = tuple(map(
                    lambda t: t.cuda(device_idx), datasets[key].tensors))
            gc.collect()

        while min_batch_size <= batch_size:
            try:
                loaders = Bunch()
                for k in datasets.keys():
                    loaders[k] = DataLoader(datasets[k], batch_size, shuffle=True,
                                            pin_memory=not keep_data_on_device)

                space_coach = FeedForwardCoach(model, loaders,
                                               logger=logger,
                                               log_dir=log_dir,
                                               tensorboard=summary_writer,
                                               tqdm=get_tqdm(position=device_idx),
                                               **config)
                logger.debug(f'Worker {device_idx}: beginning training.')
                space_coach.train_until_convergence(description={})
                logger.debug(f'Worker {device_idx}: beginning evaluation')
                stats = space_coach.evaluate({}, loaders.validation)
                mean_reward = np.mean(stats[reward_metric])

                return mean_reward

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if adaptive_batch_size:
                        batch_size = int(batch_size*batch_size_decay)
                        config['initial_lr'] *= batch_size_decay
                        logger.info(f'Out of memory, decreasing batch size to {batch_size}.')
                    else:
                        logger.info('Out of memory on fixed batch size.')
                        return None
                else:
                    logger.error(e)
                    raise e

            except LossIsNoneError:
                logger.info('Loss is NaN.')
                return np.nan

        else:
            logger.info('Out of memory with minimum batch size.')
            return None