
import multiprocessing
from queue import Empty
from itertools import chain
from abc import ABCMeta, abstractmethod

from .iteration import distinct_pairs

# import multiprocessing as mp
# from multiprocessing import Pool


def map_with_apply_async(fn, iterable, *, num_processes, mp=multiprocessing):
    pool = mp.Pool(num_processes)

    async_results = []
    for elem in iterable:
        async_results.append(pool.apply_async(fn, args=(elem,)))

    results = tuple(async_result.get() for async_result in async_results)
    
    pool.close()
    pool.join()

    return results


class ArgGroup:
    def __init__(self, *args, **kwargs):
        self._arg_group = dict(
            args=args,
            kwargs=kwargs,
        )

    def __repr__(self):
        return repr(self._arg_group)

    def __getitem__(self, key):
        return self._arg_group[key]

    @property
    def args(self):
        return self._arg_group['args']

    @property
    def kwargs(self):
        return self._arg_group['kwargs']

    def update(self, *args, **kwargs):
        args = self.args + args
        kwargs = dict(chain(self.kwargs.items(), kwargs.items()))

        return ArgGroup(*args, **kwargs)

    def augment(self, *args, **kwargs):
        args = self.args + args
        kwargs = dict(distinct_pairs(chain(self.kwargs.items(), kwargs.items())))

        return ArgGroup(*args, **kwargs)


class Processor(metaclass=ABCMeta):
    @classmethod
    def map(cls, coll, arg_groups, mp=multiprocessing):
        input_q = mp.Queue()
        for elem_idx, elem in enumerate(coll):
            input_q.put([elem_idx, elem])
        coll_length = elem_idx + 1

        output_q = mp.Queue()

        processes = []
        for processor_id, arg_group in enumerate(arg_groups):
            process = mp.Process(
                target=cls,
                args=tuple(chain([processor_id, input_q, output_q], arg_group['args'])),
                kwargs=arg_group['kwargs'])
            process.start()
            processes.append(process)

        temp_dict = dict()
        output_idx = 0
        while output_idx < coll_length:
            elem_idx, elem = output_q.get()

            if elem_idx == output_idx:
                yield elem
                output_idx += 1
            else:
                temp_dict[elem_idx] = elem

            while output_idx in temp_dict:
                yield temp_dict[output_idx]
                del temp_dict[output_idx]
                output_idx += 1

        assert output_q.empty()

        for process in processes:
            process.join()

        # output = []
        # while not output_q.empty():
        #     output.append(output_q.get())

        # return output

    def __init__(self, processor_id, input_q, output_q, *init_args, **init_kwargs):
        self.id = processor_id
        self.initialize(*init_args, **init_kwargs)

        # while not input_q.empty():
        while True:
            try:
                elem_idx, elem = input_q.get_nowait()
                output_q.put([elem_idx, self.process(elem)])
            except Empty:
                break

    @abstractmethod
    def initialize(self, *init_args, **init_kwargs):
        pass

    @abstractmethod
    def process(self, elem):
        pass


class ExampleProcessor(Processor):
    """
    >>> num_processes = 4
    >>> for output in ExampleProcessor.map(
    ...         tuple(range(20)),
    ...         arg_groups=[ArgGroup(fn=lambda x: x * x)] * num_processes
    ... ):
    ...     print(output, end=', ')
    0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 
    """

    def initialize(self, fn, sleep_time=None):
        self.fn = fn
        if sleep_time is None:
            import random
            sleep_time = random.random()
        self.sleep_time = sleep_time

    def process(self, elem):
        import time
        time.sleep(self.sleep_time)
        return self.fn(elem)
