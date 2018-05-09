import numpy as np
import pickle, re
import torch

from spacy.lang.en.stop_words import STOP_WORDS
from nasframe.utils.keras_preprocessing import Tokenizer, pad_sequences
from multiprocessing import Pool

from torch.utils.data import TensorDataset, Dataset
from torch import LongTensor

from nasframe.utils.torch import wrap

try:
    from IPython import display
    ipython = True
except ImportError:
    ipython = False


class TextLoader:
    """
    A text dataset loader and preprocessor.

    Args:
        corpus: text corpus
        task_type (str): task for which the dataset should be prepared ('classification' or 'prediction')
        labels (numpy.array, optional): labels for a 'classification' task
        filters (str): string of characters to be filtered out from the ``corpus``
        oov_token (str): token used in place of anything that is out of the vocabulary,
            if None, such tokens are dropped out.
        vocab_size (int): size of the vocabulary
        max_length (int): maximum sequence length
        verbose (bool): whether to print information about how the preprocessing is going

    """
    def __init__(self, corpus, task_type='classification', labels=None,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token=None,
                 vocab_size=20000, max_length=100, verbose=False):

        self.corpus = corpus
        self.task_type = task_type.lower()
        self.labels = labels
        self.vocab_size = vocab_size if oov_token is None else vocab_size + 1
        self.oov_token = oov_token
        self.max_length = max_length
        self.verbose = verbose
        self.filters = filters

        self.embedding = None
        self.embedding_dict = None
        self.embedding_stats = {'mean': .0, 'std': .0, 'dims': 0}

        self.sequences = None
        self._original_index = None

        self.dataset = None
        self.word_mapping = None

        self.fit_tokenizer()

    @property
    def word_index(self) -> dict:
        """
        Word index used for converting text into numeric sequences.
        """
        return self.tokenizer.word_index

    @property
    def torch_embedding(self) -> torch.nn.Embedding:
        """
        Returns:
            torch.nn.Embedding with pretrained weights if ``fit_emeddings`` was called,
            randomly initialized otherwise.
        """
        embedding = torch.nn.Embedding(self.vocab_size,
                                       self.embedding_stats['dims'])
        if self.embedding is not None:
            embedding.weight = torch.nn.Parameter(torch.FloatTensor(self.embedding))
        return embedding

    @staticmethod
    def load(path):
        """
        Loads an existing loader state, saved via ``save`` method, and returns a ``TextLoader`` instance.

        Args:
            path: path to the saved state

        """
        loader = TextLoader('')
        with open(path, 'rb') as f:
            loader.__dict__.update(pickle.load(f))
        return loader

    def fit_tokenizer(self):
        """
        Fits a ``Tokenizer`` instance on a text corpus.

        """
        self._maybe_print("Fitting tokenizer...")

        self.tokenizer = Tokenizer(num_words=self.vocab_size,
                                   oov_token=self.oov_token,
                                   filters=self.filters)

        corpus = [self.corpus] if isinstance(self.corpus, str) else self.corpus
        self.tokenizer.fit_on_texts(corpus)
        self._original_index = self.tokenizer.word_index
        self.sequences = None

    def load_embeddings(self, embedding_dim, *, embeddings_path=None, embedding_dict=None):
        """
        Loads embeddings from a vec-like file.

        Args:
            embedding_dim (int): embedding dimensionality
            embeddings_path (str, optional): path to a file where embeddings are stored
            embedding_dict (dict, optional): if specified, loading from a file is omitted

        """
        if embedding_dict is None:
            self._maybe_print("Reading embeddings file...")
            embedding_dict = {}
            with open(embeddings_path, 'r', encoding='UTF-8') as f:
                for i, line in enumerate(f):
                    key, value = read_vec(line)
                    if value.shape[0] < embedding_dim: continue
                    embedding_dict[key] = value

        emb_arr = np.array(list(embedding_dict.values()))
        self.embedding_dict = embedding_dict
        self.embedding_stats['mean'] = emb_arr.mean()
        self.embedding_stats['std'] = emb_arr.std()
        self.embedding_stats['dims'] = embedding_dim
        self.embedding = None
        self._maybe_print('Done, now call fit_embeddings.')

    def fit_embeddings(self, correct_spelling=False, max_dl=2,
                       ignore_stop_words=False, ignore_chars=True,
                       ignore_matching=re.compile('\d'), reserved_words=None,
                       num_threads=8, tqdm=None):
        """
        Fits preloaded embeddings to match with ``self.wrod_index``.

        Args:
            correct_spelling (bool): whether to correct spelling errors
            max_dl (int, optional): if ``correct_spelling`` is True, maximum Damerau-Levenshtein distance for a proposed
                spelling correction to be considered as a correction, instead of a different word
            ignore_stop_words (bool): if True, stop words are thrown out
            ignore_chars (bool): if True, single-character tokens are thrown out
            ignore_matching (re.Pattern, optional): if a token matches this pattern, it's thrown out
            reserved_words (iterable, optional): any token in this collection is guaranteed
                a randomly initialized embedding
            num_threads (int, optional): number of threads used for spell correcting
            tqdm (callable, optional): tqdm constructor for showing progress on spell correcting

        """

        if correct_spelling:
            import enchant
            en = enchant.Dict("en_US")
            if self.word_mapping is None:
                from jellyfish import damerau_levenshtein_distance as dl

                def map_word(item, max_dl=max_dl):
                    try:
                        suggestion = en.suggest(item[0])[0]
                    except:
                        return item[0], item[0]
                    if dl(item[0], suggestion) <= max_dl:
                        return item[0], suggestion
                    return item[0], item[0]

                with Pool(num_threads) as pool:
                    if tqdm is not None:
                        self.word_mapping = dict(tqdm(pool.imap(map_word,
                                            self.word_index.items()),
                                            total=len(self.word_index.items())))
                    else:
                        self.word_mapping = dict(pool.imap(map_word,
                                                 self.word_index.items()))
                    pool.close()
                    pool.join()

        self._maybe_print("Fitting embeddings...")

        self.tokenizer.word_index = self._original_index
        index_items = list(self.tokenizer.word_index.items())
        index_items.sort(key=lambda item: item[1])
        self.vocab_size = min(self.vocab_size, len(index_items))

        self.embedding = np.random.normal(self.embedding_stats['mean'],
                                          self.embedding_stats['std'],
                                          size=(self.vocab_size+1,
                                                self.embedding_stats['dims']))

        new_index, i = {}, 1
        print("Fitting embeddings...")
        for j, (word, _) in enumerate(index_items):
            if i >= self.vocab_size: break

            if new_index.get(word) is not None: continue

            if reserved_words is not None and word in reserved_words:
                new_index[word] = i = i + 1
                continue

            if word in STOP_WORDS:
                if ignore_stop_words:
                    new_index[word] = -1
                    continue

            elif ignore_chars and len(word) <= 1:
                new_index[word] = -1
                continue

            if ignore_matching and len(ignore_matching.findall(word)) > 0:
                new_index[word] = -1
                continue

            if self.verbose and i % 99 == 0:
                print(f"{i}/{self.vocab_size}", end='\r')

            vector = self.embedding_dict.get(word)
            if vector is not None:
                self.embedding[i] = vector
                new_index[word] = i
                i+=1
            elif not ignore_chars and len(word) <= 1:
                new_index[word] = i = i + 1
                continue
            elif correct_spelling and not en.check(word):
                suggestion = self.word_mapping.get(word)
                if suggestion is None:
                    new_index[word] = i = i+1
                    continue
                vector = self.embedding_dict.get(suggestion)
                if vector is not None:
                    idx = new_index.get(suggestion)
                    if idx is not None:
                        new_index[word] = idx
                    else:
                        self.embedding[i] = vector
                        new_index[suggestion] = i
                        i+=1
                else:
                    new_index[word] = i = i + 1
            else:
                new_index[word] = i = i + 1

        if self.oov_token is not None:
            new_index[self.oov_token] = max(new_index.values())+1
        self.embedding = self.embedding[:max(new_index.values())+1]

        self.tokenizer.word_index = new_index
        self.sequences = None

    def tokenize(self, sequence=None):
        """
        Tokenizes a sequence and maps it to corresponding indices from ``word_index``.

        Args:
            sequence (iterable, optional): sequence to be tokenized, if None ``self.corpus`` is used

        Returns:
            Indices, if sequence was provided, else sets ``self.sequences`` and returns None
        """
        if sequence is None:
            corpus = [self.corpus] if isinstance(self.corpus, str) else self.corpus
        else:
            corpus = [sequence] if isinstance(sequence, str) else sequence

        if 'class' in self.task_type:
            indices = self.tokenizer.texts_to_sequences(corpus)
            indices = pad_sequences(indices, maxlen=self.max_length)
        else:
            indices = self.tokenizer.texts_to_sequences(corpus)

        if sequence is None: self.sequences = indices
        else: return indices

    def make_torch_dataset(self, corpus=None, labels=None, labels_dtype=torch.long):
        """
        Makes a torch.utils.data.Dataset from ``corpus`` or ``self.sequences`` if ``corpus`` is None.

        Args:
            corpus: corpus of text to make a dataset of
            labels (iterable): labels, if there are any
            labels_dtype: data type of the labels, if there are any

        Returns:
            An instance of torch.utils.data.Dataset with corpus indices and labels is there are any as data.
        """
        if corpus is not None or self.sequences is None:
            self.tokenize(corpus)

        sequence_tensor = LongTensor(self.sequences)
        if 'predict' in self.task_type:
            self.dataset = TensorDataset(sequence_tensor[:-1],
                                         sequence_tensor[1:])
        elif 'class' in self.task_type:
            labels = labels if labels is not None else self.labels
            labels_tensor = wrap(labels, dtype=labels_dtype)
            self.dataset = TensorDataset(sequence_tensor, labels_tensor)
        else:
            class MonoDataset(Dataset):
                def __init__(self, data_tensor):
                    self.data_tensor = data_tensor

                def __getitem__(self, index):
                    return self.data_tensor[index]

                def __len__(self):
                    return self.data_tensor.size(0)

            self.dataset = MonoDataset(sequence_tensor)

        return self.dataset

    def save(self, path):
        """
        Saves current state into a file.

        Args:
            path: path to save to.

        """
        with open(path, 'wb+') as f:
            pickle.dump(vars(self), f)
        self._maybe_print("Saved.")

    def _maybe_print(self, string):
        """
        Prints ``string`` if ``self.verbose``.
        """
        if self.verbose:
            print(string)


def read_vec(string) -> (str, np.ndarray):
    """
    Parses one line of a vec-like file.

    Args:
        string: a line to parse.

    Returns:
        a tuple of (word, embedding)
    """
    word, *arr = string.strip(" \n\t").split(" ")
    return word, np.asarray(arr, dtype='float32')
