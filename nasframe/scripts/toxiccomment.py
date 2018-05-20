from .generic import *

from torch import nn

from nasframe.utils import Bunch, logger
from nasframe.loaders import TextLoader


import pandas as pd
import yaml
import gc


def train_toxic(num_gpus, val_fraction, resume, config_path, gpu_idx, force_perprocess):
    """
    Trains architect (performs NAS) of Jigsaw Toxic Comment dataset.
    See cli help, for parameter description.
    """

    with open(config_path) as f:
        config = yaml.load(f)

    data_dir = config['data_dir']
    log_dir = config['log_dir']

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

        torch.save(datasets, join(data_dir, 'preprocessed.pth'))
        torch.save(embedding, join(data_dir, 'embedding.pth'))

        del loader
        del data_val, data_train
        del val_labels, train_labels
        del val_comments, train_comments

        gc.collect()
    else:
        embedding = torch.load(join(data_dir, 'embedding.pth'))

    model = ToxicModel(None, embedding)
    make_dirs(join(log_dir))
    torch.save(model, (join(log_dir, 'model.pth')))

    worker_fn = partial(
        generic_worker,
        config=config_path,
        space_type='rnn',
        reward_metric='auc'
    )

    input_shape = config['child_training'].pop('input_shape')
    storage = train_curriculum(
        config, worker_fn,
        input_shape=input_shape,
        resume=resume,
        num_gpus=num_gpus,
        gpu_idx=gpu_idx)

    best_description, best_auc = storage.best()
    logger.info(f'Best achieved ROC AUC score on validation data set: {best_auc:.5f}.')

    with open(join(log_dir, "descriptions", "best.json"), 'w+') as f:
        json.dump(best_description, f)
    logger.info('Corresponding description is stored in '
                f'{join(log_dir, "descriptions", "best.json")}.')


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
        self.embedding = embedding
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.linear = None

    @property
    def input_size(self):
        return self.embedding.embedding_dim

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

        output_dim = description['rnn'][0]['state'][0]['dim'] * 2
        if self.linear is None or self.linear.weight.size(0) != output_dim:
            self.linear = nn.Linear(output_dim, 6).to(self.space.device)

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