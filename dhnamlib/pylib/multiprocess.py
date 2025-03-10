
import multiprocessing
from queue import Empty
from itertools import chain, cycle
from abc import ABCMeta, abstractmethod

from .iteration import distinct_pairs
from .klass import subclass, implement
# from .klass import abstractfunction
from .decoration import deprecated

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


@deprecated
class Mapper(metaclass=ABCMeta):
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


@deprecated
class ExampleMapper(Mapper):
    """
    >>> num_processes = 4
    >>> for output in ExampleMapper.map(
    ...         tuple(range(20)),
    ...         arg_groups=[ArgGroup(fn=lambda x: x * x)] * num_processes
    ... ):
    ...     print(output, end=',')
    0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,
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


class ProcessorStopException(Exception):
    pass


class Processor(metaclass=ABCMeta):
    def __init__(self, processor_id, input_q, output_q, *init_args, **init_kwargs):
        self.id = processor_id
        self.input_q = input_q
        self.output_q = output_q
        self.initialize(*init_args, **init_kwargs)
        self.run()

    def run(self):
        while True:
            input_datum = self.input_q.get()
            if isinstance(input_datum, ProcessorStopException):
                break
            else:
                elem_idx, elem = input_datum
                self.output_q.put([elem_idx, self.process(elem)])

    @abstractmethod
    def initialize(self, *init_args, **init_kwargs):
        pass

    @abstractmethod
    def process(self, elem):
        pass


class Distributor(metaclass=ABCMeta):
    def __init__(
            self,
            processor_cls,
            arg_groups,
            fixed_split=False,
            mp=multiprocessing,
    ):
        self.processor_cls = processor_cls
        self.fixed_split = fixed_split
        self.mp = mp

        if self.fixed_split:
            def get_input_q(processor_id):
                return mp.Queue()
        else:
            shared_input_q = mp.Queue()

            def get_input_q(processor_id):
                return shared_input_q

        input_q_list = []
        output_q = mp.Queue()

        processes = []
        for processor_id, arg_group in enumerate(arg_groups):
            input_q = get_input_q(processor_id)
            process = mp.Process(
                target=processor_cls,
                args=tuple(chain(
                    [processor_id, input_q, output_q],
                    arg_group['args'])),
                kwargs=arg_group['kwargs'])
            # process.start()
            input_q_list.append(input_q)
            processes.append(process)

        self.input_q_list = input_q_list
        self.output_q = output_q
        self.processes = processes

        for process in processes:
            process.start()

    def __del__(self):
        exception = ProcessorStopException()
        for input_q in self.input_q_list:
            input_q.put(exception)

        for process in self.processes:
            process.join()

        # super().__del__()

    def compute(self, coll, ordered=True):
        coll_length = self._distribute(coll)
        if ordered:
            return self._generate_ordered(coll_length)
        else:
            return self._generate_unordered(coll_length)

    def _distribute(self, coll):
        for (elem_idx, elem), input_q in zip(enumerate(coll), cycle(self.input_q_list)):
            input_q.put((elem_idx, elem))

        coll_length = elem_idx + 1
        return coll_length

    def _generate_unordered(self, coll_length):
        output_count = 0
        while output_count < coll_length:
            output_count += 1
            elem_idx, elem = self.output_q.get()
            yield elem

        assert self.output_q.empty()

    def _generate_ordered(self, coll_length):
        temp_dict = dict()
        output_idx = 0
        while output_idx < coll_length:
            elem_idx, elem = self.output_q.get()

            if elem_idx == output_idx:
                yield elem
                output_idx += 1
            else:
                temp_dict[elem_idx] = elem

            while output_idx in temp_dict:
                yield temp_dict[output_idx]
                del temp_dict[output_idx]
                output_idx += 1

        assert self.output_q.empty()


@subclass
class ExampleProcessor(Processor):
    """
    >>> num_processes = 4
    >>> distributor = Distributor(
    ...     processor_cls=ExampleProcessor,
    ...     arg_groups=[ArgGroup(fn=lambda x: x * x)] * num_processes,
    ...     fixed_split=False)
    >>> generator = distributor.compute(range(20), ordered=True)
    >>> for output in generator:
    ...     print(output, end=',')
    0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,

    >>> del distributor
    """

    @implement
    def initialize(self, fn, sleep_time=None):
        self.fn = fn
        if sleep_time is None:
            import random
            sleep_time = random.random()
        self.sleep_time = sleep_time

    @implement
    def process(self, elem):
        import time
        time.sleep(self.sleep_time)
        return self.fn(elem)


class Serializable(metaclass=ABCMeta):
    @abstractmethod
    def serialize(self):
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, serialized):
        pass


class ClassEncodable(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def clsdump(cls):
        pass

    @classmethod
    @abstractmethod
    def clsload(cls, dumped):
        pass


class Serialization(metaclass=ABCMeta):
    def __init__(self, cls_name, serialized):
        self.cls_name = cls_name
        self.serialized = serialized


def serialize(serializable: Serializable):
    return Serialization(type(serializable).__name__, serializable.serialize())
