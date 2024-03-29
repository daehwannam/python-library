
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import torch
from torch.utils.data.sampler import Sampler

from ..klass import subclass, implement
from ..iteration import iterate, slice_by_max_size
from ..hflib.acceleration import Acceleratable


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


@subclass
class EpochRepeatingDataLoader(Acceleratable):
    '''
    Example:

    >>> examples = ['example-A', 'example-B', 'example-C', 'example-D']
    >>> batch_size = 2
    >>> shuffle = False

    >>> num_epoch_repeats_1 = 3
    >>> data_loader_1 = EpochRepeatingDataLoader(
    ...     torch.utils.data.DataLoader(
    ...         SimpleDataset(examples),
    ...         batch_size=batch_size,
    ...         shuffle=shuffle),
    ...     num_epoch_repeats=num_epoch_repeats_1)
    >>> len(data_loader_1)
    6
    >>> list(data_loader_1)
    [['example-A', 'example-B'], ['example-C', 'example-D'], ['example-A', 'example-B'], ['example-C', 'example-D'], ['example-A', 'example-B'], ['example-C', 'example-D']]

    >>> num_epoch_repeats_2 = 0.5
    >>> data_loader_2 = EpochRepeatingDataLoader(
    ...     torch.utils.data.DataLoader(
    ...         SimpleDataset(examples),
    ...         batch_size=batch_size,
    ...         shuffle=shuffle),
    ...     num_epoch_repeats=num_epoch_repeats_2)
    >>> len(data_loader_2)
    1
    >>> list(data_loader_2)
    [['example-A', 'example-B']]
    '''

    # interface = Interface(Acceleratable)

    def __init__(self, data_loader, num_epoch_repeats):
        self.data_loader = data_loader
        self.num_epoch_repeats = num_epoch_repeats
        self.iterator = None

    @property
    def batch_size(self):
        return self.data_loader.batch_size

    def __len__(self):
        return round(self.num_epoch_repeats * len(self.data_loader))

    def __iter__(self):
        for iteration_idx in range(len(self)):
            if not self.iterator:
                self.iterator = iterate(self.data_loader)
            yield next(self.iterator)

    @implement
    def decompose(self):
        yield self.data_loader

    @implement
    def compose(self, data_loader):
        return type(self)(data_loader, self.num_epoch_repeats)


class VariableSizedBatchSampler(Sampler[List[int]]):
    r"""
    Sampler that uses a function that computes the size of an examle to yield a mini-batch of indices.

    Example:
    >>> data_source = [10, 30, 50, 70, 90, 0, 20, 40, 60, 80]
    >>> sampler = VariableSizedBatchSampler(data_source, lambda x: x, 100)
    >>> index_batches = tuple(sampler)
    >>> index_batches
    ((0, 1, 2), (3,), (4, 5), (6, 7), (8,), (9,))
    >>> batches = tuple(list(data_source[idx] for idx in index_batch) for index_batch in sampler)
    >>> batches
    ([10, 30, 50], [70], [90, 0], [20, 40], [60], [80])
    """

    def __init__(self, data_source: Sized, size_fn, max_size) -> None:
        self.data_source = data_source
        self.size_fn = size_fn
        self.max_size = max_size
        self._index_batches = []
        self._pre_computed = False

    def __iter__(self) -> Iterator[List[int]]:
        self._pre_compute()
        return iter(self._index_batches)

    def __len__(self) -> int:
        self._pre_compute()
        return len(self._index_batches)

    def _add_batch(self, batch):
        assert not self._pre_computed
        self._index_batches.append(batch)
        return batch

    def _pre_compute(self):
        if not self._pre_computed:
            for index_batch in self._get_iter():
                pass

    def _get_iter(self):
        for batch_slice in slice_by_max_size(
                self.data_source,
                size_fn=self.size_fn,
                max_size=self.max_size,
        ):
            yield self._add_batch(tuple(range(batch_slice.start, batch_slice.stop)))

        self._pre_computed = True
