import numpy as np

import torch
from torch import nn

from functools import partial

from torch.nn.modules import loss
from torch.nn.modules import activation

import torch.optim.lr_scheduler as sched


class Identity(nn.Module):
    """
    A ``torch.nn.Module`` which returns arguments passed to it unchanged, on instance (``forward``) call.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def get_optimizer(name):
    """
    Returns optimizer constructor given the name.

    Args:
        name (str): optimizer name

    Returns:
        Corresponding optimizer from torch.optim module.

    Available optimizers:
        ASGD, Adadelta, Adagrad, Adam, SparseAdam, Adamax, LBFGS, RMSprop, Rprop, SGD
    """
    return {
        'asgd': torch.optim.ASGD,
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': partial(torch.optim.Adam, amsgrad=True),
        'adamax': torch.optim.Adamax,
        'lbfgs': torch.optim.LBFGS,
        'rmsprop': torch.optim.RMSprop,
        'rprop': torch.optim.Rprop,
        'sgd': torch.optim.SGD,
        'sparseadam': torch.optim.SparseAdam
    }[name.strip().lower()]


def get_activation(name, **kwargs):
    """
    Returns instantiated activation given the name.

    Args:
        name (str): activation name
        kwargs (dict): keyword arguments passed to activation constructor

    Returns:
        Corresponding activation instance from torch.nn module

    Available activations:
        ELU, SELU, GLU, ReLU, ReLU6, PReLU, RReLU, LeakyReLU,
        Sigmoid, LogSigmoid, Softmax, Softmax2d, LogSoftmax,
        Softmin, Softplus, Softshrink, Softsign, Tanhshrink,
        Tanh, Hardshrink, Hardtanh, identity/linear/none
    """
    return {
        'elu': activation.ELU(**kwargs),
        'glu': activation.GLU(**kwargs),
        'hardshrink': activation.Hardshrink(**kwargs),
        'hardtanh': activation.Hardtanh(**kwargs),
        'leakyrelu': activation.LeakyReLU(**kwargs),
        'logsigmoid': activation.LogSigmoid(),
        'logsoftmax': activation.LogSoftmax(**kwargs),
        'prelu': activation.PReLU(**kwargs),
        'rrelu': activation.RReLU(**kwargs),
        'relu': activation.ReLU(**kwargs),
        'relu6': activation.ReLU6(**kwargs),
        'selu': activation.SELU(**kwargs),
        'sigmoid': activation.Sigmoid(),
        'softmax': activation.Softmax(**kwargs),
        'softmax2d': activation.Softmax2d(),
        'softmin': activation.Softmin(**kwargs),
        'softplus': activation.Softplus(**kwargs),
        'softshrink': activation.Softshrink(**kwargs),
        'softsign': activation.Softsign(),
        'tanh': activation.Tanh(),
        'tanhshrink': activation.Tanhshrink(),
        'identity': Identity(),
        'linear': Identity(),
        'none': Identity(),
    }[name.strip().lower()]


def get_criterion(name, **kwargs):
    """
    Returns criterion instance given the name.

    Args:
        name (str): criterion name
        kwargs (dict): keyword arguments passed to criterion constructor

    Returns:
        Corresponding criterion from torch.nn module

    Available criteria:
        BCE, BCEWithLogits, CosineEmbedding, CrossEntropy, HingeEmbedding, KLDiv,
        L1, MSE, MarginRanking, MultilabelMargin, MultilabelSoftmargin, MultiMargin,
        NLL, PoissoNLL, SmoothL1, SoftMargin, TripletMargin

    """
    return {
        'bce': loss.BCELoss(**kwargs),
        'bcewithlogits': loss.BCEWithLogitsLoss(**kwargs),
        'cosineembedding': loss.CosineEmbeddingLoss(**kwargs),
        'crossentropy': loss.CrossEntropyLoss(**kwargs),
        'hingeembedding': loss.HingeEmbeddingLoss(**kwargs),
        'kldiv': loss.KLDivLoss(**kwargs),
        'l1': loss.L1Loss(**kwargs),
        'mse': loss.MSELoss(**kwargs),
        'marginranking': loss.MarginRankingLoss(**kwargs),
        'multilabelmargin': loss.MultiLabelMarginLoss(**kwargs),
        'multilabelsoftmargin': loss.MultiLabelSoftMarginLoss(**kwargs),
        'multimargin': loss.MultiMarginLoss(**kwargs),
        'nll': loss.NLLLoss(**kwargs),
        'poissonnll': loss.PoissonNLLLoss(**kwargs),
        'smoothl1': loss.SmoothL1Loss(**kwargs),
        'softmargin': loss.SoftMarginLoss(**kwargs),
        'tripletmargin': loss.TripletMarginLoss(**kwargs)
    }[name.strip().lower()]


def get_scheduler(name):
    """
    Returns scheduler constructor given the name.

    Args:
        name (str): scheduler name

    Returns:
        Corresponding scheduler from torch.optim module

    Available criteria:
        CosineAnnealing, Exponential, Lambda, MultiStep, ReduceOnPlateau, Step

    """
    return {
        'cosineannealing': sched.CosineAnnealingLR,
        'exponential': sched.ExponentialLR,
        'lambda': sched.LambdaLR,
        'multistep': sched.MultiStepLR,
        'reduceonplateau': sched.ReduceLROnPlateau,
        'step': sched.StepLR
    }[name.strip().lower()]


def wrap(inputs, device=-1, **kwargs):
    """
    Makes a tensor or copies (and detaches) an existing one and puts it on `device`.

    Args:
        inputs: torch.Tensor, list or numpy.ndarray or None, in the last case, converted to np.nan
        device: GPU index or -1 for CPU or device name (e.g. 'cpu', 'cuda:0') or ``torch.Device``
        kwargs: optional kwargs passed to ``torch.tensor`` constructor.

    Returns:
        ``torch.Tensor`` with inputs as data.
    """
    if isinstance(inputs, map):
        inputs = list(inputs)
    if isinstance(inputs, list):
        inputs = list(map(lambda i: np.nan if i is None else i, inputs))
    if torch.is_tensor(inputs):
        inputs = inputs.detach()
    if inputs is None:
        inputs = np.nan

    if isinstance(device, int):
        device = 'cpu' if device == -1 else f'cuda:{device}'

    out = torch.tensor(inputs, **kwargs).to(device)

    return out


def get_np(inputs: torch.Tensor):
    """
    Attempts to wrap inputs in a ``numpy.ndarray``.
    If torch.Tensor passed, places it onto CPU and calls ``tensor.numpy()``

    Args:
        inputs (iterable): inputs to be wrapped.

    Returns:
        ``numpy.ndarray`` containing inputs.
    """
    if torch.is_tensor(inputs):
        if inputs.is_cuda:
            inputs = inputs.cpu()
        return inputs.numpy()
    return np.array(inputs)


def param_count(module):
    """
    Counts parameters in passed module.

    Args:
        module (torch.nn.Module): module to count parameters in.

    Returns:
        Parameter count.
    """
    return sum(map(lambda p: p.numel(), module.parameters()))
