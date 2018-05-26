import logging, os, time, copy, yaml

from collections import defaultdict
from functools import partial


def now():
    """
    Returns:
        Current formatted date-time.
    """
    return time.strftime("%d.%m.%Y_%H:%M:%S")


def get_logger(name=__file__, file=None, file_level='INFO', stdout_level='DEBUG'):
    """
    Gets logger by name if it exists, otherwise initializes one.

    Args:
        name (str):  logger name
        file (str, optional):  path to the log file
        stdout_level (str): stdout logging level
        file_level (str):  file logging level

    Returns:
        A logger instance
    """
    logger = logging.getLogger(name)

    if os.path.exists('config.yml'):
        with open('config.yml') as f:
            config = yaml.load(f)
        file_level = config.get('file_loglevel', file_level)
        stdout_level = config.get('stdout_loglevel', stdout_level)

    if getattr(logger, '_init_done__', False):
        logger.setLevel(file_level)
        return logger

    min_level = min(map(logging._nameToLevel.get, [file_level, stdout_level]))

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(min_level)

    formatter = logging.Formatter("%(levelname)s %(asctime)s: %(message)s",
                                  datefmt='%d/%m/%Y %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(stdout_level)

    del logger.handlers[:]
    logger.addHandler(handler)

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()


def delete_file(path):
    """
    Deletes a file, if it exists.

    Args:
        path (str): path of a file to be deleted

    """
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Path {path} is no more.")


def make_dirs(path):
    """
    Makes dirs, including intermediate one if they don't exist.
    Args:
        path: path that needs to be created

    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created path {path}.")


def selfless(variables):
    """
    Returns a copy of a dict of variables, without ``self`` in it.

    Args:
        variables (dict or callable): named variables
    """
    if not isinstance(variables, dict)\
            and hasattr(variables,'__call__'):
        variables = variables()
    variables = copy.copy(variables)
    try:
        variables.pop('self')
    except KeyError:
        pass
    return variables


class Bunch:
    """
    Convenience class, which can be thought of as an extension to pythons ``dict``
    which allows accessing it's items by class attribute access operator ``.``

    Args:
        dict_or_bunch (dict or Bunch, optional): a collection to initialize items from.
    """
    def __init__(self, dict_or_bunch=None):
        if dict_or_bunch is not None:
            self.update(dict_or_bunch)

    def update(self, object):
        """
        Updates itself with an object.

        Args:
            object: ``Bunch``, ``dict``, ``list`` or ``tuple`` of items (each of length 2) or
                    any object, in which case ``vars(object)`` will be used.

        """
        if isinstance(object, dict):
            pass
        elif isinstance(object, (tuple, list)) \
                and all(len(o) == 2 for o in object):
            object = dict(object)
        else:
            object = vars(object)
        self.__dict__.update(object)

    def __getitem__(self, item):
        return self.__dict__.get(item, None)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, item):
        return item in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def pprint(self, _indent_level=1):
        """
        Pretty prints a (possibly nested) Bunch.

        Args:
            _indent_level (int): level of indentation to be used on current recursion level

        """
        print("Bunch of: {")
        for k, v in self.items():
            if type(v) is Bunch:
                print(f'{4*_indent_level*" "}{k}: ', end="")
                v.pprint(_indent_level+1)
            else:
                lines = str(v).split('\n')
                if len(lines)==1:
                    print(f'{4*_indent_level*" "}{k}: {v},')
                else:
                    print(f'{4*_indent_level*" "}{k}: ')
                    for line in lines:
                        print(f'{4*(_indent_level+1)*" "}{line}')
        print(f"{4*(_indent_level-1)*' '}}}")

    def as_dict(self):
        dict_ = {}
        for k, v in self.items():
            if isinstance(v, Bunch):
                dict_[k] = v.as_dict()
            else:
                dict_[k] = v
        return dict_

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def keys(self):
        return self.__dict__.keys()


def add_if_doesnt_exist(dictionary, key, value):
    """
    Mutates ``dictionary`` to contain a ``key`` if it doesn't already.

    Args:
        dictionary (dict): dictionary to add key to
        key (hashable): a ``dict`` key
        value (any): a value corresponding to the ``key``

    """
    if key not in dictionary:
        dictionary[key] = value


def add_increment(dictionary, key):
    """
    If ``key`` doesn't exist in ``dictionary``, increments maximum
    existing value by one, and assigns a `key`` to it.

    Args:
        dictionary (dict): dictionary to be modified
        key (hashable): a key to assign incremented value to
    """
    add_if_doesnt_exist(dictionary, key, max(dictionary.values())+1)


def bool_val(string):
    """
    Returns a boolean value of a string.

    Examples:
        >>> bool_val('yes')
        True
        >>> bool_val('True')
        True
        >>> bool_val('0')
        False
    Args:
        string (str): string to be evaluated

    """
    val = string.lower() in ['true', 't', 'y', 'yes']
    try: val |= bool(int(string))
    except ValueError: pass
    return val


def return_value(value):
    """
    Returns a value passed to it.

    Args:
        value (any): value to be returned

    Returns:
        ``value``
    """
    return value


def _nested_defaultdict(levels, factory, value):
    """
    A recursive call for ``nested_defaultdict``.

    Args:
        levels (int): depth of nestedness
        factory (callable, optional): factory for the leaf ``defaultdict``
        value (any, optional): a value, which will be used for the default factory if ``factory`` is not specified

    Returns:
        A factory which produces a defaultdict
    """
    if levels == 0:
        if factory is not None:
            return factory
        else:
            return partial(return_value, value)

    ddict = _nested_defaultdict(levels - 1, factory, value)
    return partial(defaultdict, ddict)


def nested_defaultdict(levels, factory=None, value=None):
    """
    Produces a nested defaultdict with ``defaultdict(factory)`` as a leaf.
    If ``factory`` is not specified, creates a factory that returns ``value``

    Args:
        levels (int): depth of nestedness
        factory (callable, optional): factory for the leaf ``defaultdict``
        value (any, optional): a value, which will be used for the default factory if ``factory`` is not specified

    Returns:
        A nested defaultdict
    """
    return _nested_defaultdict(levels, factory, value)()


def convert_keys_to_int(dictionary):
    """
    Tries to convert each key of a (possibly nested) ``dict`` to ``int``.
    Useful when loading from json.

    Args:
        dictionary (dict): dictionary to be processed

    Returns:
        A copy of ``dictionary`` with numeric-like keys converted to ``int``
    """
    dictionary = copy.deepcopy(dictionary)
    for k in list(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            dictionary[k] = convert_keys_to_int(dictionary[k])

        try:
            k_new = int(k)
            dictionary[k_new] = dictionary[k]
            del dictionary[k]
        except ValueError:
            pass

    return dictionary


def get_tqdm(try_notebook=False, leave=False, **kwargs):
    """
    Returns tqdm progressbar constructor.

    Args:
        try_notebook (bool): if ``True`` will attempt to import ``tqdm_notebook`` and return it instead
        leave (bool): whether to leave the bar, after it's closed
        redirect (bool): whether to redirect stdout and stderror to a bar constructed with returned constructor

    Returns:
        A tqdm progressbar constructor
    """
    from tqdm import tqdm, tqdm_notebook
    if try_notebook:
        try:
            import IPython
            return tqdm_notebook
        except ImportError:
            pass
    return partial(tqdm, leave=leave, **kwargs)


def int_to_range(value, shift=0):
    """
    Converts ``value`` to ``range(shift, value + shift))`` if ``value`` is ``int``,
    returns ``value`` unchanged otherwise.
    """
    return list(range(shift, value + shift)) if isinstance(value, int) else value
