from nasframe.utils.misc import Bunch, selfless, add_if_doesnt_exist, get_tqdm
from nasframe.utils.torch import *

from tensorboardX import SummaryWriter


class LossIsNoneError(BaseException):
    """
    Intended to be raised when loss tensor is overflown.
    """
    pass


class CoachBase:
    """
    A base class for coach classes, which encapsulate training methods and variables and facilitate
    training code reuse.

    Args:
        model (nn.Module): model to be trained
        name (str): name of the model to be trained
        optimizer (str): a name of the optimizer type to be used
        optimizer_kwargs (dict): keyword arguments passed to optimizer constructor
        initial_lr (float): initial learning rate
        grad_clip_norm (float): gradient clipping norm, if 0, gradients don't get clipped
        scheduler (str): name of scheduler type to be used
        scheduler_kwargs (dict): keyword arguments passed to scheduler constructor
        scheduler_metric (str): key of ``metrics`` dict, to be passed to ``scheduler.step``
        scheduler_scale (str): 'step' or 'epoch', at what frequency ``scheduler.step`` gets called
        steps_per_epoch (int): number of training steps per one epoch
        metrics (dict): dictionary of metrics descriptions
        log_dir (str): path to the log directory
        tensorboard (str, bool, SummaryWriter): if True, SummaryWriter is instantiated in the ``log_dir``, if str,
            assumed to be a path at which SummaryWrtier should be instantiated, if SummaryWriter, well, then it's
            just a SummaryWriter.
        log_every (int): number of training steps defining the frequency of logging
        logger (Logger): a logger instance to be used for logging
        break_on_nan (bool): whether to break training loop if np.nan is encountered in loss
        raise_on_nan (bool): whether to raise LossIsNoneException if np.nan is encountered in loss,
            this has higher priority than ``break_on_nan``
        tqdm (callable, optional): tqdm constructor, if None, default is used
    """
    def __init__(self, model, name='model',
                 optimizer='adam', optimizer_kwargs=None, initial_lr=1e-3, grad_clip_norm=.0,
                 scheduler='reduceonplateau', scheduler_kwargs=None, scheduler_metric='loss',
                 scheduler_scale='step', steps_per_epoch=None, metrics=None, log_dir='logs',
                 tensorboard=None, log_every=10, logger=None, break_on_nan=True, raise_on_nan=True,
                 tqdm=None, convergence_patience_mult=2, convergence_patience=None, **kwargs):

        if logger is not None and len(kwargs) > 0:
            kwargs = ', '.join(kwargs.keys())
            logger.info(f'Keyword arguments {kwargs} were provided, but won\'t be used.')

        self.model = model
        self.name = name

        self.grad_norm = grad_clip_norm
        self.initial_lr = initial_lr
        self.optim_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.scheduler_kwargs = scheduler_kwargs or {}
        self.sched_name = scheduler
        self.scheduler_metric = scheduler_metric
        self.scheduler_scale = scheduler_scale

        self.epoch_steps = steps_per_epoch

        if self.epoch_steps is not None:
            default_patience = self.epoch_steps // 3 if scheduler_scale == 'step' else 1
            add_if_doesnt_exist(self.scheduler_kwargs, 'patience', default_patience)

        if isinstance(metrics, dict):
            self.metrics = metrics
        elif metrics is None:
            self.metrics = {}
        else:
            raise ValueError(f'`metrics` must be a dict or None, got {type(metrics)}.')

        self.log_every = log_every
        self.logger = logger

        # region convergence detection
        self.convergence = Bunch()

        if convergence_patience is None:
            self.convergence.patience = self.scheduler_kwargs.get('patience', 100 if scheduler_scale == 'step' else 1)
            self.convergence.patience = int(self.convergence.patience*convergence_patience_mult)
        else:
            self.convergence.patience = convergence_patience

        self.convergence.threshold = self.scheduler_kwargs.get('threshold', 1e-3)
        self.convergence.mode = self.scheduler_kwargs.get('mode', 'min')
        # endregion

        self.log_dir = log_dir
        if isinstance(tensorboard, bool) and tensorboard:
            self.summary_writer = SummaryWriter(log_dir)
        elif isinstance(tensorboard, str):
            self.summary_writer = SummaryWriter(tensorboard)
        else:
            self.summary_writer = tensorboard

        self.break_on_nan = break_on_nan
        self.raise_on_nan = raise_on_nan

        self.tqdm = tqdm if tqdm is not None else get_tqdm()

        self.reset()

    @property
    def current_lr(self):
        """
        Returns the current learning rate (assuming there's only one optimizer parameter group)
        """
        if self.optim is not None:
            return self.optim.param_groups[0]['lr']

    @property
    def step(self):
        """
        Returns current epoch step
        """
        return self.stats.step

    @property
    def accumulated_step(self):
        """
        Returns step accumulated across epochs
        """
        return self.stats.accumulated_step

    @property
    def is_converged(self):
        """
        Returns True if the model is converged
        """
        return self.convergence.indicator

    def get_current_lr(self):
        """
        Returns the current learning rate (assuming there's only one optimizer parameter group)
        """
        return self.current_lr

    def decay_lr(self, factor):
        """
        Multiplies current learning rate by ``factor``.
        """
        for group in self.optim.param_groups:
            group['lr'] *= factor

    def update_convergence_status(self, metric):
        """
        Updates the convergence status, given a ``metric`` value.
        """
        rel_epsilon = 1. - self.convergence.threshold
        if self.convergence.mode == 'min':
            is_better = metric*rel_epsilon < self.convergence.best_val
        elif self.convergence.mode == 'max':
            is_better = metric * rel_epsilon > self.convergence.best_val
        else:
            raise ValueError(f'Unknown convergence mode {self.convergence.mode}')

        if is_better:
            self.convergence.best_val = metric
            self.convergence.counter = 0
            self.convergence.indicator = False
        else:
            self.convergence.counter += 1

        if self.convergence.counter >= self.convergence.patience:
            self.convergence.indicator = True

    def reset(self):
        """
        Resets all counters and stats to their initial values
        """
        self.stats = Bunch({})
        self.stats.step = 0
        self.stats.accumulated_step = 0
        self.stats.epoch = 0

        optim = get_optimizer(self.optim_name)
        self.optim = optim(self.model.parameters(), lr=self.initial_lr, **self.optimizer_kwargs)

        if self.sched_name is not None:
            self.scheduler = get_scheduler(self.sched_name)
            self.scheduler = self.scheduler(self.optim, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        self.convergence.counter = 0
        self.convergence.indicator = False
        self.convergence.best_val = -np.inf if self.convergence.mode == 'max' else np.inf

    def train(self, epochs=1, steps=None, *args, **kwargs):
        """
        Train for ``epochs`` epochs, each of ``steps`` batches.

        Args:
            epochs (int): number of epochs to train
            steps (int, optional): number of batches in one epoch

        """
        steps = steps or self.epoch_steps
        self.clear_stats()
        self.model.train()
        for epoch in range(epochs):
            self._training_loop(steps, *args, **kwargs)
            self._finalize_epoch()

    def train_until_convergence(self, *args, **kwargs):
        """
        Trains until the model has converged.
        """
        steps = kwargs.get('steps', self.epoch_steps)
        self.clear_stats()
        self.model.train()
        while not self.is_converged:
            self._training_loop(steps, break_on_convergence=True, *args, **kwargs)
            self._finalize_epoch()

    def _training_loop(self, steps, *args, **kwargs):
        """
        The training loop, which trains for ``steps`` steps.
        """
        self.stats.step = 0
        steps = min(self.epoch_steps, steps) if steps is not None else self.epoch_steps

        progress_bar = self.tqdm(range(steps), desc=f'Training {self.name} on {str(self.model.device)}')
        for i in progress_bar:
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

    def _training_loop_body(self, *args, **kwargs):
        """
        The boilerplate of training loop.
        Should consist roughly of getting the loss, backpropagating and
        performing an optimizer step.
        """
        raise NotImplemented()

    def clear_stats(self, *args, **kwargs):
        """
        Clears accumulated stats.
        """
        raise NotImplemented()

    def _finalize_epoch(self, *args, **kwargs):
        """
        Finalizes an epoch of training.
        """
        if self.scheduler is not None and self.scheduler_scale == 'epoch':
            metric = self._evaluate_metric(self.metrics[self.scheduler_metric])
            self.scheduler.step(metric)
            self.update_convergence_status(metric)

        self.stats.epoch += 1

    def _log(self, *args, **kwargs):
        """
        Logs some info about training.
        """
        if self.metrics is not None:
            evaluated_metrics = {}
            for name, values in self.metrics.items():
                evaluated_metrics[name] = self._evaluate_metric(values)

                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(f"{self.name}/{name}",
                                                   evaluated_metrics[name],
                                                   self.accumulated_step)

            if self.logger is not None:
                metric_string = ' | '.join([f'{k}: {v:6.4f}' for k, v in evaluated_metrics.items()])
                self.logger.info(f'Training {self.name}, epoch {self.stats.epoch},'
                                 f' step {self.stats.step}: {metric_string}')

    def _evaluate_metric(self, metric, **kwargs):
        """
        Evaluates a metric description.
        """
        metric_kwargs = {}
        for kwarg in metric['kwargs']:
            if kwarg in self.stats:
                metric_kwargs[kwarg] = self.stats[kwarg]
            elif kwarg in kwargs:
                metric_kwargs[kwarg] = kwargs[kwarg]

        if isinstance(metric['fn'], str):
            arg = metric_kwargs[metric['kwargs'][0]][-self.log_every:]
            if len(arg) == 0: return np.nan

            if 'np' in metric['fn']:
                if 'mean' in metric['fn']:
                    arg = np.mean(arg)
            elif 'last' in metric['fn']:
                arg = arg[-1]
            else:
                if 'stack' in metric['fn']:
                    arg = torch.stack(arg)
                if 'mean' in metric['fn']:
                    arg = torch.mean(arg)
                if 'sum' in metric['fn']:
                    arg = torch.sum(arg)
            return arg

        return metric['fn'](**metric_kwargs)
