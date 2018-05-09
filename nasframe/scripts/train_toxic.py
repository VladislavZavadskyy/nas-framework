from .train_generic import *

from PIL import Image
from torch import nn

from nasframe.utils import Bunch, logger, get_tqdm
from nasframe.coaches.base import LossIsNoneError
from nasframe import RNNSpace, FeedForwardCoach
from nasframe.utils.filelock import FileLock
from nasframe.loaders import TextLoader
from torch.utils.data import DataLoader

import pandas as pd
import click
import gc


@click.command()
@click.option('-g', '--num-gpus', help=
              'Number of GPUs to use.',
              default=3, type=int)
@click.option('-n', '--max-nodes', help=
              'Max number of nodes in one graph.',
              default=15, type=int)
@click.option('-s', '--max-states', help=
              'Max number of states.',
              default=5, type=int)
@click.option('-v', '--val-fraction', help=
              'Fraction of training data set to be used for validation.',
              default=.2, type=float, )
@click.option('-r', '--resume', help=
              'If set, will attempt to resume previous NAS session.',
              is_flag=True, default=False)
@click.option('--child-lr', help=
              'Initial learning rate for children networks w.r.t. --child-batch-size.',
              type=float, default=1e-3)
@click.option('--child-batch-size', help=
              'Batch size used when training children, '
              'given that --adaptive-batch-size is not set.',
              default=256, type=int)
@click.option('--adaptive-batch-size', help=
              'If set, will attempt to use --max-batch-size first, '
              'then, on OOM encounters, will decay it with --batch-size-decay '
              'factor, until --min-batch-size is reached.',
              is_flag=True, default=True)
@click.option('--min-batch-size', help=
              'Minimum batch size (see --adaptive-batch-size help).',
              type=int, default=32)
@click.option('--max-batch-size', help=
              'Maximum batch size (see --adaptive-batch-size help).',
              type=int, default=512)
@click.option('--batch-size-decay', help=
              'Batch size decay factor (see --adaptive-batch-size help).',
              type=float, default=.5)
@click.option('--architect-batch-size', help=
              'Size of description batch used for training the architect.',
              default=8, type=int)
@click.option('--architect-epoch-steps', help=
              'Steps (number of batches) per epoch of architect training.',
              default=4, type=int)
@click.option('--data-dir', help=
              'Path to Jigsaw Toxic Comment Challenge data sets.',
              default='data/toxic', type=click.Path(True, file_okay=False))
@click.option('--gpu-idx', help=
              'GPU indices as comma separated values. '
              'If not set, range(--num-gpus) will be used.',
              default=None, type=str)
@click.option('--log-dir', help=
              'Path, where intermediate data and logs are going to be stored',
              default='logs', type=click.Path(file_okay=False))
@click.option('--storage-surplus-factor', help=
              'For curriculum training, the factor by which number of stored '
              'data points (description-rewards) must surpass minimum required'
              'at the current curriculum level.',
              type=float, default=1.3)
@click.option('--keep-data-on-device', help=
              'If set, will attempt to store data sets in the device '
              'memory instead of RAM. This may require a lot of it.',
              is_flag=True, default=False)
@click.option('--force-perprocess', help=
              'Will force preprocessing, even if preprocessed data exists.',
              is_flag=True, default=False)
@click.option('--dont-load-architect', help=
              'Will train architect anew, even if checkpoint exists.',
              is_flag=True, default=False)
@click.option('--search-space-kwargs', help=
              'Path to a .json with desired search space keyword arguments.',
              type=click.Path(True, dir_okay=False), default='spaceconf.json')
def cli(data_dir, val_fraction, child_batch_size, force_perprocess, max_states,
        max_nodes, search_space_kwargs, num_gpus, log_dir, resume,
        storage_surplus_factor, architect_batch_size, architect_epoch_steps,
        adaptive_batch_size, min_batch_size, max_batch_size, child_lr,
        keep_data_on_device, batch_size_decay, gpu_idx, dont_load_architect):
    """
    Preforms neural architecture search on Jigsaw Toxic Comment dataset.
    """
    assert max_states >= 1, 'Maximum number of states has to be >= 1.'
    assert max_nodes >= 1, 'Maximum number of nodes has to be >= 1.'
    assert child_batch_size >= 1, 'Child batch size has to be >= 1.'
    assert architect_batch_size >= 1, 'Architect batch size has to be >= 1.'
    assert architect_epoch_steps >= 1, 'Architect steps pre epoch to be >= 1.'
    assert 0 < val_fraction < 1, 'Validation data fraction has to be in range (0,1).'
    assert num_gpus >= 1, 'Number of GPUs has to be >= 1.'
    assert min_batch_size >= 1, 'Min batch size can\'t be less than 1.'
    assert min_batch_size <= min_batch_size, 'Max batch size must be >= min_batch_size.'
    assert child_lr > 0, 'Child learning rate has to be > 0.'
    assert 0 < batch_size_decay < 1, 'Batch size decay has to be in range (0,1)'

    load_architect = not dont_load_architect

    train_toxic(data_dir, val_fraction, child_batch_size, force_perprocess, max_states,
                max_nodes, search_space_kwargs, num_gpus, log_dir, resume,
                storage_surplus_factor, architect_batch_size, architect_epoch_steps,
                adaptive_batch_size, min_batch_size, max_batch_size, child_lr,
                keep_data_on_device, batch_size_decay, gpu_idx, load_architect)


def train_toxic(data_dir, val_fraction, child_batch_size, force_perprocess, max_states,
                max_nodes, search_space_kwargs, num_gpus, log_dir, resume,
                storage_surplus_factor, architect_batch_size, architect_epoch_steps,
                adaptive_batch_size, min_batch_size, max_batch_size, child_lr,
                keep_data_on_device, batch_size_decay, gpu_idx, load_architect):
    """
    Trains architect (performs NAS) of Jigsaw Toxic Comment dataset.
    See cli help, for parameter description.
    """
    if force_perprocess or not exists(join(data_dir, 'preprocessed.pth')):
        data_train = pd.read_csv(join(data_dir, 'train.csv')).fillna('Nan').sample(frac=1)

        label_cols = [c for c in data_train.columns if c not in {'comment_text', 'id'}]

        val_points = int(len(data_train) * val_fraction)
        data_val = data_train.iloc[:val_points]
        data_train = data_train.iloc[val_points:]

        train_comments = data_train.comment_text.values
        train_labels = data_train[label_cols].values

        val_comments = data_val.comment_text.values
        val_labels = data_val[label_cols].values

        if exists(join(data_dir, 'loader')):
            loader = TextLoader.load(join(data_dir, 'loader'))
        else:
            loader = TextLoader(data_train.comment_text, verbose=True)
            loader.load_embeddings(
                300, embeddings_path=join(data_dir, '..', 'crawl-300d-2M.vec'))
            loader.fit_embeddings()

            loader.embedding_dict = None
            loader.corpus = None

            loader.save(join(data_dir, 'loader'))

        datasets = Bunch()
        datasets.train = loader.make_torch_dataset(train_comments, train_labels, labels_dtype=torch.float)
        datasets.validation = loader.make_torch_dataset(val_comments, val_labels, labels_dtype=torch.float)

        embedding = loader.torch_embedding

        torch.save([datasets, embedding], join(data_dir, 'preprocessed.pth'))

        del loader
        del data_val, data_train
        del val_labels, train_labels
        del val_comments, train_comments

        gc.collect()

    if search_space_kwargs is not None:
        with open(search_space_kwargs) as f:
            kwargs = json.load(f)
    else:
        kwargs = {}

    space_proto = RNNSpace(max_nodes, max_states, **kwargs)

    worker_fn = partial(
        toxic_worker,
        log_dir=log_dir,
        data_dir=data_dir,
        initial_lr=child_lr,
        keep_data_on_device=keep_data_on_device,
        adaptive_batch_size=adaptive_batch_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_size=child_batch_size,
        batch_size_decay=batch_size_decay,
    )

    storage = train_curriculum(
        space_proto, worker_fn,
        input_shape=(-1, 300),
        max_complexity=max_nodes,
        log_dir=log_dir,
        resume=resume,
        num_gpus=num_gpus,
        gpu_idx=gpu_idx,
        epochs_per_loop=10,
        architect_lr_decay=.9,
        load_architect=load_architect,
        storage_surplus_factor=storage_surplus_factor,
        architect_steps_per_epoch=architect_epoch_steps,
        architect_batch_size=architect_batch_size
    )

    best_description, best_auc = storage.best()
    logger.info(f'Best achieved ROC AUC score on validation data set: {best_auc:.5f}.')

    with open(join(log_dir, "descriptions", "best.json"), 'w+') as f:
        json.dump(best_description, f)
    logger.info('Corresponding description is stored in '
                f'{join(log_dir, "descriptions", "best.json")}.')


def toxic_worker(description, device_queue, log_dir, data_dir,
                 batch_size, initial_lr, current_complexity,
                 adaptive_batch_size, min_batch_size,
                 max_batch_size, batch_size_decay,
                 keep_data_on_device):
    """
    Concurrent description evaluator.

    Args:
        description (dict): description to be evaluated
        device_queue (Queue): a queue with available CUDA devices indices
        log_dir (str): path to log directory
        data_dir (str): path to the data directory
        batch_size (int): default batch size
        initial_lr (float):  ``model``'s initial earning rate
        current_complexity (int): current curriculum complexity level.
        adaptive_batch_size (bool): whether to use adaptive batch size.
        min_batch_size (int): the lowest possible batch size if using adaptive batch size.
        max_batch_size (int): the highest (and default) possible batch size if using adaptive batch size.
        batch_size_decay (float): factor to decay batch size by, if OOM has been encountered.
        keep_data_on_device (bool): if ``True`` ``datasets`` will be transferred to device memory.

    Returns:
        Tuple of (description, mean_auc, device_idx).

        If loss is nan -- mean_auc = -1.

        If OOM was raised and ``not adaptive_batch_size`` or using ``min_batch_size`` causes OOM -- mean_auc=None.
    """
    description = dict(description)
    device_idx = device_queue.get()
    datasets, embedding = torch.load(join(data_dir, 'preprocessed.pth'))

    with torch.cuda.device(device_idx):
        logger, summary_writer, description_dir = worker_init(description, log_dir)
        save_path = join(log_dir, 'rnn', 'merged_space.pth')
        with FileLock(save_path):
            space = torch.load(save_path)

        space.logger = logger
        model = ToxicModel(space, embedding).cuda(device_idx)

        data_shape = [
            batch_size,
            datasets.train.tensors[0].shape[-1],
            model.embedding.embedding_dim
        ]
        desc = space.preprocess(description, data_shape)

        model.set_linear(desc['rnn'][0]['state'][0]['dim'] * 2)

        space.draw(desc, join(description_dir, 'graph.png'))
        img = np.array(Image.open(join(description_dir, 'graph.png')))
        summary_writer.add_image('graph', img)

        if adaptive_batch_size:
            initial_lr *= max_batch_size/batch_size
            batch_size = max_batch_size

        if keep_data_on_device:
            for key in datasets.keys():
                datasets[key].tensors = tuple(map(
                    lambda t: t.cuda(device_idx), datasets[key].tensors))
            gc.collect()

        loaders = None
        while min_batch_size <= batch_size:
            try:
                loaders = Bunch()
                for k in datasets.keys():
                    loaders[k] = DataLoader(datasets[k], batch_size, shuffle=True,
                                            pin_memory=not keep_data_on_device)

                space_coach = FeedForwardCoach(model, loaders,
                                               criterion='bcewithlogits',
                                               initial_lr=initial_lr,
                                               logger=logger,
                                               log_dir=log_dir,
                                               scheduler_kwargs={'mode': 'max'},
                                               scheduler_metric='auc',
                                               tensorboard=summary_writer,
                                               tqdm=get_tqdm(position=device_idx))

                space_coach.train_until_convergence(description=desc)
                stats = space_coach.evaluate(desc, loaders.validation)

                with FileLock(save_path):
                    if exists(save_path):
                        other = torch.load(save_path)
                        space.merge(other.to(space.device))
                    space.cpu().save(log_dir, 'merged_space')

                with FileLock(join(log_dir, 'description_reward.json')):
                    with open(join(log_dir, 'description_reward.json'), 'r') as f:
                        existing = json.load(f)
                    assert isinstance(existing, dict)

                    if str(current_complexity) not in existing:
                        existing[str(current_complexity)] = []

                    existing[str(current_complexity)].append([description,
                                                         np.mean(stats.auc)])
                    with open(join(log_dir, 'description_reward.json'), 'w+') as f:
                        json.dump(existing, f)

                return description, np.mean(stats.auc), device_idx

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if adaptive_batch_size:
                        batch_size = int(batch_size*batch_size_decay)
                        initial_lr *= batch_size_decay
                        logger.info(f'Out of memory, decreasing batch size to {batch_size}.')
                    else:
                        return description, None, device_idx
                else:
                    logger.error(e)
                    raise e
            except LossIsNoneError:
                return description, -1, device_idx
            finally:
                for name in list(locals()):
                    del locals()[name]
                torch.cuda.empty_cache()
                gc.collect()
                device_queue.put(device_idx)
        else:
            logger.info('Out of memory encountered on min batch size.')
            for name in list(locals()):
                if name not in ['description', 'device_idx']:
                    del locals()[name]
            torch.cuda.empty_cache()
            gc.collect()
            device_queue.put(device_idx)
            return description, None, device_idx


class ToxicModel(GenericModel):
    """
    A (one of many possible) model, for Jigsaw Toxic Comment dataset.
    This one uses concatenated 1D average and max pooling after unrolled
    ``RNNSpace`` cell output as an input to the last linear.

    Args:
        space (RNNSpace): a space
    """
    def __init__(self, space, embedding):
        super().__init__(space)
        d = self.space.device
        self.embedding = embedding.to(d)
        self.avg_pool = nn.AdaptiveAvgPool1d(1).to(d)
        self.max_pool = nn.AdaptiveMaxPool1d(1).to(d)

    def set_linear(self, size):
        """
        Sets ``self.linear`` to a new linear layer with weight matrix of shape ``(size, 6)``.

        Args:
            size: output of the penultimate layer (state_0 shape)

        """
        self.linear = nn.Linear(size, 6).to(self.space.device)

    def prepare(self, inputs, description):
        """
        A wrapper for the ``space.prepare`` function.

        It's needed since actual input shape differs from
        passed as inputs are indices passed through embedding.

        Args:
            inputs:
            description:

        Returns:

        """
        if torch.is_tensor(inputs):
            inputs = self.embedding(inputs)
        elif isinstance(inputs, (list, tuple, torch.Size)):
            if inputs[-1] != self.embedding.embedding_dim:
                inputs = list(inputs) + [self.embedding.embedding_dim]
        self.cell = self.space.prepare(inputs, description)
        return self

    def forward(self, inputs, description):
        embedded = self.embedding(inputs)
        if self.cell is None:
            raise ValueError('Prepare must be called prior to forward.')
        rnn_out = self.cell(inputs=embedded, description=description, return_sequence=True)[0]
        avgs = self.avg_pool(rnn_out.transpose(1, 2)).squeeze()
        maxes = self.avg_pool(rnn_out.transpose(1, 2)).squeeze()
        out = self.linear(torch.cat((avgs, maxes), -1))
        return out
