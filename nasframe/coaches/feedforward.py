from .base import CoachBase, LossIsNoneError
from torch.nn.utils import clip_grad_norm
from nasframe.utils.misc import selfless, add_if_doesnt_exist
from nasframe.utils.torch import *

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import functional as F


class FeedForwardCoach(CoachBase):
    """
    Class which facilitates training of feed forward search spaces or models, based on such spaces,
    given they follow interface defined in scripts.GenericModel.

    Args:
        space (nn.Module): SearchSpace or GenericModel and their descendants
        loaders (Bunch): bunch of data loaders containing 'train' and 'validation' keys
        criterion (str): name of criterion type to use for loss calculation
        criterion_kwargs (dict): keyword arguments passed to the criterion constructor
        **kwargs: keyword arguments passed to CoachBase constructor
    """
    def __init__(self, space, loaders=None, criterion=None, criterion_kwargs=None, **kwargs):

        add_if_doesnt_exist(kwargs, 'steps_per_epoch', len(loaders.train))
        super().__init__(space, **kwargs)
        self.loaders = loaders

        self.space = space
        self.model = None

        self.criterion_kwargs = criterion_kwargs or {}

        if criterion is not None:
            self.criterion = get_criterion(criterion, **self.criterion_kwargs)
        else:
            self.criterion = None

        self.metrics.update({
            'loss': {
                'fn': 'np_mean',
                'kwargs': ['losses']
            },
            'lr': {
                'fn': self.get_current_lr,
                'kwargs': []
            }
        })

        if 'bce' in criterion.lower():
            self.metrics['auc'] = {
                'fn': 'np_mean',
                'kwargs': ['auc']
            }

    def update_optim(self, params):
        """
        Creates optimizer for given parameters, retaining current learning rate.
        
        Args:
            params: parameters to optimize.

        """
        optim = get_optimizer(self.optim_name)
        lr = self.current_lr or self.initial_lr
        self.optim = optim(params, lr=lr, **self.optimizer_kwargs)

        if self.sched_name is not None:
            self.scheduler = get_scheduler(self.sched_name)
            self.scheduler = self.scheduler(self.optim, **self.scheduler_kwargs)
        else:
            self.scheduler = None

    def train(self, epochs=1, steps=None, *args, **kwargs):
        """
        Train for ``epochs`` epochs, each of ``steps`` batches.

        Args:
            epochs (int): number of epochs to train
            steps (int, optional): number of batches in one epoch

        """
        assert 'description' in kwargs, 'You should consider providing description of the graph to train.'
        steps = steps or self.epoch_steps
        self.clear_stats()
        self.model = None
        for epoch in range(epochs):
            self._training_loop(steps, *args, **kwargs)
            self._finalize_epoch()

    def train_until_convergence(self, *args, **kwargs):
        """
        Trains until the model has converged.
        """
        assert 'description' in kwargs, 'You should consider providing description of the graph to train.'
        steps = kwargs.get('steps', self.epoch_steps)
        self.clear_stats()
        self.model = None
        while not self.is_converged:
            self._training_loop(steps, break_on_convergence=True, *args, **kwargs)
            self._finalize_epoch()

    def clear_stats(self, *args, **kwargs):
        self.stats.losses = []
        if 'auc' in self.metrics:
            self.stats.auc = []

    def _training_loop(self, steps, *args, **kwargs):
        """
        A training loop, which wraps ``steps`` calls to ``_training_loop_body``.
        If ``steps`` is not specified, defaults to  ``len(loaders.train)``.
        """
        self.stats.step = 0
        steps = min(self.epoch_steps, steps) if steps is not None else self.epoch_steps

        progress_bar = self.tqdm(self.loaders.train, desc=f'Training {self.name} on {str(self.space.device)}')

        for i, (input, labels) in enumerate(progress_bar):
            if self.stats.step >= steps: break

            self._training_loop_body(**selfless(locals()), **kwargs)

            if np.isnan(self.stats.losses[-1]) and self.raise_on_nan:
                raise LossIsNoneError('Loss is None.')
            if np.isnan(self.stats.losses[-1]) and self.break_on_nan:
                break

            if self.stats.accumulated_step % self.log_every == 0 and self.stats.accumulated_step > 0:
                self._log(**selfless(locals()))

            self.stats.accumulated_step += 1
            self.stats.step += 1

            if self.scheduler is not None and self.scheduler_scale == 'step':
                metric = self._evaluate_metric(self.metrics[self.scheduler_metric])
                self.scheduler.step(metric)
                self.update_convergence_status(metric)
                if self.is_converged and kwargs.get('break_on_convergence', False):
                    break

    def _training_loop_body(self, input, labels, description,
                            progress_bar=None, *args, **kwargs):
        """
        The boilerplate of training loop.
        Roughly consists of getting the loss, backpropagating and
        performing an optimizer step.
        """
        loss = self.get_loss(description, input, labels)

        self.optim.zero_grad()
        loss.backward()

        if self.grad_norm is not None and self.grad_norm > 0:
            clip_grad_norm(self.model.parameters(), self.grad_norm)

        self.optim.step()

        self.stats.losses.append(loss.item())

        if progress_bar is not None:
            progress_bar.set_postfix_str(f'Loss: {loss.item():5.3f} | Epoch: {self.stats.epoch}')

    def evaluate(self, description, loader=None, *args, **kwargs):
        """
        Evaluates trained model on data from ``loader`` or
        ``self.loaders.validation`` if the former is not specified.

        Args:
            description (dict): description to evaluate.
            loader (DataLoader, optional): loader with validation data

        Returns:
            Model performance stats collected during evaluation
        """
        self.clear_stats()
        loader = loader or self.loaders.validation
        if self.model is None:
            raise ValueError('You should train the model first.')
        self.model = self.model.eval()

        progress_bar = self.tqdm(loader, desc=f'Evaluating {self.name} on {str(self.space.device)}')
        for i, (input, labels) in enumerate(progress_bar):
            with torch.no_grad():
                loss = self.get_loss(description, input, labels, *args, **kwargs)
                self.stats.losses.append(loss.item())

                progress_bar.set_postfix_str(f'Loss: {loss.item():5.3f}')

        if self.logger is not None:
            self.logger.info(f'{self.name} evaluation loss is: '
                             f'{np.mean(self.stats.losses):5.3f}')
            if 'auc' in self.stats:
                self.logger.info(f'{self.name} evaluation ROC AUC score is: '
                                 f'{np.mean(self.stats.auc):5.3f}')
        return self.stats

    def get_loss(self, description, inputs, labels, mean=True):
        """
        A method which handles calculating loss and all that comes with it.

        Args:
             description (dict): description of the graph to evaluate
             inputs (Tensor): pretty self-explanatory
             labels (Tensor): labels, corresponding to inputs
             mean (bool): whether or not to return mean value of losses.

        Returns:
            loss.
        """
        inputs = wrap(inputs, self.space.device)
        labels = wrap(labels, self.space.device)

        if self.model is None:
            self.model = self.space.prepare(inputs, description)
            self.model.train()
            self.update_optim(self.model.parameters())

        out = self.model(inputs=inputs, description=description)
        loss = self.criterion(out, labels)

        if 'auc' in self.metrics:
            out = F.sigmoid(out.detach())
            try:
                self.stats.auc.append(roc_auc_score(get_np(labels), get_np(out)))
            except ValueError:
                pass

        return loss.mean() if mean else loss