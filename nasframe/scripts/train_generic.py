from flatten_dict import flatten

from nasframe.utils.torch import *
from nasframe.utils import make_dirs
from nasframe.utils import get_logger, FileLock
from nasframe import Architect, ArchitectCoach
from nasframe.storage import CurriculumStorage, Storage

from tensorboardX import SummaryWriter
from functools import partial
from os.path import exists

from shutil import rmtree
from os.path import join

from PIL import Image

import os
import json
import copy

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
        device_idx (int): CUDA device index for the worker to work on
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


def prepare(space_proto, curriculum, log_dir, resume,
            architect_batch_size, architect_steps_per_epoch,
            input_shape, load_architect, max_curriculum_complexity=None):
    """
    Initializes training session.

    Args:
        space_proto (SearchSpace): search space prototype
        curriculum (bool): whether to prepare for a curriculum training session
        log_dir (str): path to the log directory
        resume (boot): whether to attempt to resume a previous session
        architect_batch_size (int): number of samples processed per training step
        architect_steps_per_epoch (int):  number of steps per epoch of architect training
        input_shape (list, tuple, torch.Size): shape of the inputs
        load_architect (bool): if True architect will be loaded, and won't be trained
            if no graphs were evaluated on current loop iteration.
        max_curriculum_complexity (int, optional): maximum level of curriculum complexity,
            can be ommited if ``curriculum`` is ``False``

    Returns:
        Architect and an ArchitectCoach
    """
    if not resume:
        if exists(log_dir):
            rmtree(log_dir)

    make_dirs(join(log_dir, 'descriptions'))

    if exists(join(log_dir, 'architect')):
        for path in os.listdir(join(log_dir, 'architect')):
            if path.startswith('events.out.tfevents'):
                os.remove(join(log_dir, 'architect', path))
    else:
        make_dirs(join(log_dir, 'architect'))

    summary_writer = SummaryWriter(join(log_dir, 'architect'))
    arch_logger = get_logger('arch_coach', join(log_dir, 'architect', 'training.log'))

    if (exists(join(log_dir, 'architect', 'checkpoint.pth'))
            and resume and load_architect):
        arch = torch.load(join(log_dir, 'architect', 'checkpoint.pth'))
        arch.search_space = space_proto
        arch_logger.info('Restored architect from checkpoint.')

    if curriculum:
        assert max_curriculum_complexity is not None
        storage = CurriculumStorage(max_curriculum_complexity)
    else:
        storage = Storage()

    arch = Architect(space_proto)
    arch_coach = ArchitectCoach(arch, storage,
                                scheduler=None,
                                logger=arch_logger,
                                curriculum=curriculum,
                                epochs_to_increase=np.inf,
                                tensorboard=summary_writer,
                                batch_size=architect_batch_size,
                                steps_per_epoch=architect_steps_per_epoch)

    if not exists(join(log_dir, 'description_reward.json')):
        with open(join(log_dir, 'description_reward.json'), 'w+') as f:
            json.dump({}, f)
    elif resume:
        if curriculum:
            storage = CurriculumStorage.from_json(
                join(log_dir, 'description_reward.json'),
                arch, space_proto, input_shape, max_curriculum_complexity)

            flat_storage = storage.flatten()
            arch_logger.info(f'Loaded {len(flat_storage)} descriptions, '
                             f'{len(storage.storages)} curriculum levels total.')
        else:
            storage = Storage.from_json(arch, space=space_proto, input_shape=input_shape,
                                        path=join(log_dir, 'description_reward.json'))
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

        arch_coach.storage = storage

    if not exists(join(log_dir, space_proto.name, 'merged_space.pth')):
        space_proto.save(log_dir, 'merged_space')

    FileLock(join(log_dir, space_proto.name, 'merged_space.pth.lock')).purge()
    FileLock(join(log_dir, 'description_reward.json')).purge()

    return arch, arch_coach


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
        append_deterministic (bool): whehter to append description obtained by deterministically choosing the actions
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
    descriptions = copy.deepcopy(descriptions)

    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:

        device_queue = manager.Queue()
        for idx in gpu_indices:
            device_queue.put(idx)

        worker_fn = partial(worker_fn, device_queue=device_queue)

        eval_list = list(map(manager.dict, descriptions))
        with ctx.Pool(len(gpu_indices)) as pool:
            result = pool.map(worker_fn, eval_list)

    return result


def train_curriculum(space_proto, worker, input_shape,
                     max_complexity, log_dir, resume,
                     architect_steps_per_epoch,
                     architect_batch_size,
                     architect_lr_decay,
                     storage_surplus_factor,
                     num_gpus, gpu_idx,
                     epochs_per_loop,
                     load_architect):
    """
    Curriculum architect training procedure.
    Includes sampling descriptions with complexity :math:`i`, evaluating them, train architect and start over again
    with complexity :math:`i+1`.
    When ``max_compexity`` is reached, disables curriculum, flattens :class:`CurriculumStorage` and trains plainly
    from then on.

    Args:
        space_proto (SearchSpace): search space prototype
        worker (callable): worker callable
        input_shape (tuple, list, torch.Size): input's shape
        max_complexity (int): maximum  curriculum complexity
        log_dir (str): path to the log directory
        resume (bool): whether  to try resuming the previous session
        architect_batch_size (int): number of samples processed per training step
        architect_steps_per_epoch (int):  number of steps per epoch of architect training
        architect_lr_decay (float): architect learning rate decay factor,
            which is applied every ``epochs_per_loop`` epochs.
        storage_surplus_factor (float): the factor by which number of stored
              data points (description-rewards) must surpass minimum required
              at the current curriculum level
        num_gpus (int): number of GPUs to use
        gpu_idx (str, optional): string of comma separated dpu indices. if ``None``, ``range(num_gpus)`` is used.
        epochs_per_loop (int): number of architect training epochs per one loop
        load_architect (bool): if True architect will be loaded, and won't be trained
            if no graphs were evaluated on current loop iteration.

    Returns:
        On keyboard interrupt returns storage filled with all that's benn found and evaluated.
    """
    arch, arch_coach = prepare(space_proto,
                               curriculum=True,
                               input_shape=input_shape,
                               log_dir=log_dir, resume=resume,
                               architect_batch_size=architect_batch_size,
                               architect_steps_per_epoch=architect_steps_per_epoch,
                               max_curriculum_complexity=max_complexity,
                               load_architect=load_architect)

    storage = arch_coach.storage
    summary_writer = arch_coach.summary_writer

    if gpu_idx is None:
        gpu_indices = range(num_gpus)
    else:
        gpu_indices = map(int, gpu_idx.split(','))
    gpu_indices = list(gpu_indices)

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

                arch.curriculum = False
                storage.set_complexity(0)
                curriculum_complexity = 0

                arch.search_space.release_all_constraints()
                arch.storage.storages[0] = arch.storage.flatten()

            else:

                curriculum_complexity = 0

            evaluation_list, deterministic_is_viable = sample_loop(
                arch, storage, space_proto, input_shape, desired_storage_len)

            worker_fn = partial(worker, current_complexity=curriculum_complexity)
            result = evaluate(evaluation_list, worker_fn, gpu_indices)

            accuracies = []
            for description, reward, _ in result:
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

            # Since last reward corresponds to deterministic description
            if len(result) > 0:
                if result[-1][1] is None:
                    deterministic_is_viable = False

                if deterministic_is_viable:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies[:-1]), loops)
                    summary_writer.add_scalar('deterministic_acc', accuracies[-1], loops)
                else:
                    summary_writer.add_scalar('stochastic_acc', np.mean(accuracies), loops)

            storage.filter_na()
            if len(storage) < desired_storage_len and curriculum_complexity != 0:
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


def train_plain(space_proto, worker, input_shape,
                log_dir, resume, architect_steps_per_epoch,
                architect_batch_size, architect_lr_decay,
                num_gpus, gpu_idx, epochs_per_loop,
                load_architect):
    """
        Architect training procedure.
        Includes sampling descriptions, evaluating them, train architect and start over again
        with complexity i+1.

        Args:
            space_proto (SearchSpace): search space prototype
            worker (callable): worker callable
            input_shape (tuple, list, torch.Size): input's shape
            log_dir (str): path to the log directory
            resume (bool): whether  to try resuming the previous session
            architect_batch_size (int): number of samples processed per training step
            architect_steps_per_epoch (int):  number of steps per epoch of architect training
            architect_lr_decay (float): architect learning rate decay factor,
                which is applied every ``epochs_per_loop`` epochs.
            num_gpus (int): number of GPUs to use
            gpu_idx (str, optional): string of comma separated dpu indices. if ``None``, ``range(num_gpus)`` is used.
            epochs_per_loop (int): number of architect training epochs per one loop
            load_architect (bool): if True architect will be loaded, and won't be trained
                if no graphs were evaluated on current loop iteration.

        Returns:
            On keyboard interrupt returns storage filled with all that's benn found and evaluated.
        """
    arch, arch_coach = prepare(space_proto,
                               curriculum=False,
                               input_shape=input_shape,
                               load_architect=load_architect,
                               log_dir=log_dir, resume=resume,
                               architect_batch_size=architect_batch_size,
                               architect_steps_per_epoch=architect_steps_per_epoch)

    storage = arch_coach.storage
    summary_writer = arch_coach.summary_writer

    if gpu_idx is None:
        gpu_indices = range(num_gpus)
    else:
        gpu_indices = map(int, gpu_idx.split(','))
    gpu_indices = list(gpu_indices)

    loops = 0
    points_per_epoch = arch_coach.batch_size * arch_coach.epoch_steps

    while True:
        try:
            desired_storage_len = len(storage) + points_per_epoch
            evaluation_list, deterministic_is_viable = sample_loop(
                arch, storage, space_proto, input_shape, desired_storage_len)

            result = evaluate(evaluation_list, worker, gpu_indices)

            accuracies = []
            for description, reward, _ in result:
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
