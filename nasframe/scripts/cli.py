import click


@click.group()
def nas():
    """
    Neural Architecture Search Framework's command line interface
    """
    pass


@nas.command()
@click.argument('filename',
                type=click.Path(exists=True, dir_okay=False))
@click.option('-c', '--config', help=
              'Path to the YAML config file',
              type=click.Path(exists=True, dir_okay=False),
              default='config.yml')
@click.option('-p', '--preprocess',  help=
              'Whether to pre-process description prior to saving',
              is_flag=True)
@click.option('-d', '--draw', help=
              'Whether to draw an image of the best described graph',
              is_flag=True)
@click.option('--save-to', help=
              'Path to the folder, in which description '
              '(and image, if -d is set) are going to be saved',
              type=click.Path(file_okay=False), default='best')
def find_best(filename, config, preprocess, draw, save_to):
    """
    Finds the best description
    """
    from nasframe.storage import Storage, CurriculumStorage
    from nasframe.searchspaces import get_space_type
    from nasframe.utils import make_dirs

    from os.path import join

    import json
    import yaml

    with open(filename) as f:
        data = json.load(f)

    if isinstance(data, list):
        storage = Storage.from_json(data=data)
    elif isinstance(data, dict):
        storage = CurriculumStorage.from_json(data=data)
    else:
        raise ValueError('Format, in which data is stored, is not understood.')

    description, reward = storage.best()
    click.echo(f'Best reward is {reward}')

    with open(config) as f:
        config = yaml.load(f)

    type_name = config['searchspace'].pop('type')
    space = get_space_type(type_name)(**config['searchspace'])

    input_shape = config['child_training'].pop('input_shape')
    if preprocess:
        description = space.preprocess(description, input_shape)
        pcount = space.parameter_count(description)
        click.echo(f'Parameter count: {pcount[1]/1e+6:.2f} million')

    make_dirs(save_to)
    with open(join(save_to, 'description.json'), 'w+') as f:
        json.dump(description, f)
    click.echo(f'Description saved to {join(save_to, "description.json")}')

    if draw:
        if not preprocess:
            description = space.preprocess(description, input_shape)

        space.draw(description, (join(save_to, 'graph.png')))
        click.echo(f'Graph image is saved to {join(save_to, "graph.png")}')


@nas.command()
@click.option('-g', '--num-gpus', help=
              'Number of GPUs to use.',
              default=3, type=int)
@click.option('-v', '--val-fraction', help=
              'Fraction of training data set to be used for validation.',
              default=.2, type=float, )
@click.option('-r', '--resume', help=
              'If set, will attempt to resume previous NAS session.',
              is_flag=True, default=False)
@click.option('-c', '--config-path', help=
              'Path, to the config YAML file.',
              default='config.yml', type=click.Path(exists=True, dir_okay=False))
@click.option('--gpu-idx', help=
              'GPU indices as comma separated values. '
              'If not set, range(--num-gpus) will be used.',
              default=None, type=str)
@click.option('--force-perprocess', help=
              'Will force preprocessing, even if preprocessed data exists.',
              is_flag=True, default=False)
def toxic(num_gpus, val_fraction, resume, config_path, gpu_idx, force_perprocess):
    """
    Preforms neural architecture search on Jigsaw Toxic Comment dataset
    """
    assert 0 < val_fraction < 1, 'Validation data fraction has to be in range (0,1).'
    assert num_gpus >= 1, 'Number of GPUs has to be >= 1.'

    from .toxiccomment import train_toxic
    train_toxic(num_gpus, val_fraction, resume, config_path, gpu_idx, force_perprocess)