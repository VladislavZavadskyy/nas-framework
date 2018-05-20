# Neural Architecture Search Framework 
 [![Docs](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://vladislavzavadskyy.github.io/nas-framework)

NAS Framework, as the name suggests, is a framework which facilitates
[neural architecture search](https://arxiv.org/abs/1611.01578) on various datasets.
It provides a simple and flexible way to define a search space of
arbitrary complexity and an Architect class, which works without modifications
in any search space defined following the template.

An Architect is a recurrent neural network,
which generates computational graph descriptions, by recursively creating
a representation of computational graph predicted to the moment and
choosing an action (i.e. point in a particular dimension of the search
space) following policy which receives that representation as an input.

The architect is trained used reinforcement learning, specifically an Actor-Critic-like variation of
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm on various datasets.

The package is currently in alpha and supports Multilayer Perceptron and Recurrent Neural Network search spaces out of the box.
If your needs can be satisfied by those two search spaces, then all you need to do, in order to perform a neural architecture search, 
is to make a `torch.utils.data.Dataset` with your data and to modify the `toxic_worker` in `scripts.train_toxic` a bit.

## Installation
Install the package via pip, by running:
```
pip install git+https://github.com/VladislavZavadskyy/nas-framework
```

## Running the demo
There's a demo included, which performs a search of RNN architecture on the
[Jigsaw Toxic Comment dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
To run it, follow these steps:
1. Create a directory named `data`.  
2. Download `train.csv.zip` from [kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and unpack it to the `data/toxic`.
3. Download pretrained [fasttext embeddings](https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip), unpack them and place into `data` directory.
3. Run `nas toxic` (append `--help` option to see available arguments).

During the search a `logs` (or one specified with `--log-dir`) directory will be created, 
which will contain information about the search process. You can also view descriptions being evaluated, 
child network training progress and other info by running a tensorboard server in that directory.

## Getting the best found description
To get the best foung description, run `nas find_best` with path to `description_reward.json` as an argument.
See `nas find_best --help` for options.
