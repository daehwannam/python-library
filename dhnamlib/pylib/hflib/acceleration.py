
import types
import itertools
from abc import ABCMeta, abstractmethod
import functools

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import gather_object

# from ..decoration import construct
# from ..decoration import to_variables
from ..iteration import split_by_lengths, partition, iterate
from ..time import get_time_seed as _get_time_seed_without_sync

# from ..torchlib.dnn import EpochRepeatingDataLoader


class Acceleratable(metaclass=ABCMeta):
    @abstractmethod
    def decompose(self):
        pass

    @abstractmethod
    def compose(self, *args):
        pass

    def safe_decompose(self):
        decomposed_objs = self.decompose()
        assert isinstance(decomposed_objs, (tuple, types.GeneratorType))

        for decomposed_obj in decomposed_objs:
            assert isinstance(decomposed_obj, _default_types), f'An object of {type(decomposed_obj)} is not allowed'
            yield decomposed_obj


_default_types = (
    torch.utils.data.DataLoader,
    torch.nn.Module,
    torch.optim.Optimizer,
    (getattr(torch.optim.lr_scheduler, 'LRScheduler', None) or getattr(torch.optim.lr_scheduler, '_LRScheduler', None))
)

_all_types = _default_types + (Acceleratable,)


# @construct(tuple)
# @to_variables
def xprepare(accelerator: Accelerator, objs, device_placement=None):
    """
    Extension of Accelerator.prepare.

    >>> from dhnamlib.pylib.torchlib.dnn import EpochRepeatingDataLoader, SimpleDataset

    >>> examples = ['example-A', 'example-B', 'example-C', 'example-D']
    >>> batch_size = 2
    >>> shuffle = False

    >>> num_epoch_repeats_1 = 3
    >>> data_loader_1 = EpochRepeatingDataLoader(
    ...    torch.utils.data.DataLoader(
    ...        SimpleDataset(examples),
    ...        batch_size=batch_size,
    ...        shuffle=shuffle),
    ...    num_epoch_repeats=num_epoch_repeats_1)

    >>> data_loader_2 = torch.utils.data.DataLoader(
    ...        SimpleDataset(examples),
    ...        batch_size=batch_size,
    ...        shuffle=shuffle)

    >>> accelerator = Accelerator()
    >>> new_data_loader_1, new_data_loader_2 = xprepare(accelerator, [data_loader_1, data_loader_2])
    """

    for obj in objs:
        assert isinstance(obj, _all_types), f'An object of {type(obj)} is not allowed'

    decomposed_obj_groups = []
    if device_placement is not None:
        assert len(objs) == len(device_placement)
        decomposed_dev_place_groups = []
    group_sizes = []

    for idx, obj in enumerate(objs):
        if isinstance(obj, Acceleratable):
            decomposed_obj_group = tuple(obj.safe_decompose())
        else:
            decomposed_obj_group = [obj]
        decomposed_obj_groups.append(decomposed_obj_group)
        if device_placement is not None:
            dev_place = device_placement[idx]
            decomposed_dev_place_groups.append([dev_place] * len(decomposed_obj_group))
        group_sizes.append(len(decomposed_obj_group))

    all_decomposed_objs = tuple(itertools.chain(*decomposed_obj_groups))
    all_accelerated_decomposed_objs = accelerator.prepare(
        *all_decomposed_objs,
        device_placement=(
            list(itertools.chain(*decomposed_dev_place_groups))
            if device_placement is not None else
            None
        )
    )
    if len(all_decomposed_objs) == 1:
        all_accelerated_decomposed_objs = (all_accelerated_decomposed_objs,)

    accelerated_decomposed_obj_groups = split_by_lengths(all_accelerated_decomposed_objs, group_sizes)

    output_objs = []

    for obj, accelerated_decomposed_obj_group in zip(objs, accelerated_decomposed_obj_groups):
        if isinstance(obj, Acceleratable):
            output_obj = obj.compose(*accelerated_decomposed_obj_group)
        else:
            [output_obj] = accelerated_decomposed_obj_group

        output_objs.append(output_obj)

    if len(output_objs) == 1:
        return output_objs[0]
    else:
        return output_objs


class XAccelerator(Accelerator):
    def prepare(self, *args, device_placement=None):
        return xprepare(super(), args, device_placement=device_placement)

    def save_pretrained_model(self, model, save_directory):
        # if self.is_local_main_process:
        unwrapped_model = self.unwrap_model(model)

        # https://huggingface.co/docs/accelerate/v0.25.0/en/package_reference/accelerator#-transformers-models
        # https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
        unwrapped_model.save_pretrained(
            save_directory,
            is_main_process=self.is_main_process,
            save_function=self.save,
            # state_dict=self.get_state_dict(model),
        )

    def prepare_val_data_loader(self, data_loader):
        _even_batches = self.even_batches
        self.even_batches = False
        prepared_data_loader = self.prepare(data_loader)
        self.even_batches = _even_batches

        return prepared_data_loader

    def within_local_main_process(self, func):
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            if self.is_local_main_process:
                return func(*args, **kwargs)

        return decorated_func

    @property
    def accelerating(self):
        return self.distributed_type != DistributedType.NO


class NoAccelerator:
    def __init__(self, device):
        self._device = device

    def prepare(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            return args

    def save_pretrained_model(self, model, save_directory):
        model.save_pretrained(
            save_directory,
            # state_dict=self.get_state_dict(model),
        )

    @property
    def device(self):
        return self._device

    # def to_device(self, obj):
    #     return obj.to(self.device)

    def backward(self, loss):
        loss.backward()

    @property
    def is_local_main_process(self):
        return True

    @property
    def is_main_process(self):
        return True

    @property
    def sync_gradients(self):
        return True

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    def prepare_val_data_loader(self, data_loader):
        return data_loader

    def wait_for_everyone(self):
        pass

    def within_local_main_process(self, func):
        return func

    @property
    def accelerating(self):
        return False

    def unwrap_model(self, model):
        return model


def get_time_seed():
    '''
    Compute a seed by using time for accelerate.utils.set_seed
    '''
    [seed] = broadcast_object_list([_get_time_seed_without_sync(exponent=6)], from_process=0)
    return seed


def set_seed_randomly():
    set_seed(get_time_seed())


def alternate_object(obj, batch_size):
    local_part_seq = tuple(partition(obj, batch_size, strict=False))
    part_seqs = gather_object([local_part_seq])
    num_processes = len(part_seqs)
    part_iterators = tuple(map(iterate, part_seqs))
    alternated_part_seq = []

    process_idx = 0
    while part_iterators[process_idx]:
        alternated_part_seq.extend(next(part_iterators[process_idx]))
        process_idx = (process_idx + 1) % num_processes

    for process_idx in range(num_processes):
        assert not part_iterators[process_idx]

    return alternated_part_seq
