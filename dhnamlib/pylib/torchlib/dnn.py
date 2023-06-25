
import os
import numpy as np
# from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rnnlib
# from ..decoration import deprecated
from ..iteration import nest, get_elem


class MyModule(nn.Module):
    # @classmethod
    # def set_gpu_usage(cls, gpu_usage):
    #     cls.gpu_usage = gpu_usage

    def __init__(self):
        super().__init__()

        if len(tuple(self.parameters())) == 0:
            # because enable_cuda requires at least one parameter
            self.scalar_parameter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def enable_cuda(self, arg):
        "it should be carefully used because of performance issue."
        return enable_cuda(self, arg)

    def load_state_dict_partially(self, prev_state_dict):
        # https://github.com/pytorch/vision/issues/173
        model_dict = self.state_dict()

        overlap = {k: v for k, v in prev_state_dict.items()
                   if k in model_dict}

        new_state_dict = self.state_dict()
        new_state_dict.update(overlap)
        self.load_state_dict(new_state_dict)

        missing = {k: v for k, v in model_dict.items()
                   if k not in prev_state_dict}
        unexpected = {k: v for k, v in prev_state_dict.items()
                      if k not in model_dict}

        return missing, unexpected

    def get_module_device(self):
        return get_module_device(self)

    def parameters_requiring_grad(self):
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter


def enable_cuda(model, arg):
    "it should be carefully used because of performance issue."
    if is_cuda_enabled(model):
        arg = arg.cuda()
    else:
        arg = arg.cpu()
    return arg


def get_any_parameter(model):
    try:
        return next(model.parameters())
    except StopIteration:
        raise Exception("The model '{}' has no parameter.".format(model.__name__))


def is_cuda_enabled(model):
    return get_any_parameter(model).is_cuda


def get_module_device(model):
    return get_any_parameter(model).device


def make_cuda_on(condition):
    def cuda_on(arg):
        if condition:
            arg = arg.cuda()
        else:
            arg = arg.cpu()
        return arg
    return cuda_on


def create_lstm(*args, **kargs):
    # return rnnlib.LayerNormLSTM(*args, **kargs)  # debug
    if kargs['r_dropout'] > 0 or kargs['layer_norm_enabled']:
        return rnnlib.LayerNormLSTM(*args, **kargs)
    else:
        del kargs['r_dropout'], kargs['layer_norm_enabled']
        if ((len(args) > 2 and args[2] == 1) or kargs['num_layers'] == 1) and kargs['dropout'] > 0:
            kargs['dropout'] = 0
        return nn.LSTM(*args, **kargs)


def create_lstm_cell(*args, **kargs):
    if kargs['dropout'] > 0 or kargs['layer_norm_enabled']:
        return rnnlib.LayerNormLSTMCell(*args, **kargs)
    else:
        del kargs['dropout'], kargs['layer_norm_enabled']
        return nn.LSTMCell(*args, **kargs)


def normalize(tensor):
    return tensor / tensor.sum()


class WeightedTanh(nn.Module):
    def __init__(self, scale_weight=None):
        super().__init__()
        if scale_weight is None:
            self.scale_weight = nn.Parameter(torch.ones(1))
        else:
            self.scale_weight = scale_weight

    def forward(self, tensor):
        return self.scale_weight * torch.tanh(tensor)


class LinearOnly(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)


class BilinearOnly(nn.Bilinear):
    def __init__(self, in1_features, in2_features, out_features, bias=False):
        super().__init__(in1_features, in2_features, out_features, bias)


class BilinearFlat(nn.Bilinear):
    def __init__(self, in1_features, in2_features, bias=True):
        super().__init__(in1_features, in2_features, out_features=1, bias=bias)

    def forward(self, input1, input2):
        """
        two input tensors should match their intermediate sizes from k to n-1
        e.g. input1.size() --> (a, b, c, d, e, in1_features)
             input2.size() --> (d, e, in2_features)

             or
             input1.size() --> (b, c, d, e, in1_features)
             input2.size() --> (a, b, c, d, e, in2_features)

             then
             output.size() --> (a, b, c, d, e)

        :param input1: (N,*,H1) or (*,H1)
        :param input2: (N,*,H2) or (*,H2)
        :output: (N,*)
        """
        input1_size = input1.size()
        input2_size = input2.size()

        if len(input1_size) == len(input2_size):
            super_output = super().forward(input1, input2)
        elif len(input1_size) < len(input2_size):
            new_input1 = self.expand_smaller_tensor(input1, input2)
            super_output = super().forward(new_input1, input2)
        else:
            assert len(input1_size) > len(input2_size)
            new_input2 = self.expand_smaller_tensor(input2, input1)
            super_output = super().forward(input1, new_input2)

        return super_output.squeeze(-1)

    def expand_smaller_tensor(self, smaller_tensor, bigger_tensor):
        smaller_size = smaller_tensor.size()
        bigger_size = bigger_tensor.size()

        gap_size = self.compare_sizes(smaller_size, bigger_size)
        # multiplier = 1
        # for dim in gap_size:
        #     multiplier *= dim

        # return smaller_tensor.view(-1).repeat(multiplier).view(gap_size + smaller_size)
        return smaller_tensor.view(-1).expand(gap_size + smaller_size)

    def compare_sizes(self, size1, size2):
        assert len(size1) < len(size2)
        for dim1, dim2 in zip(reversed(size1[:-1]), reversed(size2[:-1])):
            assert dim1 == dim2, "Tensors' intermediate dimensions should be different"
        gap_size = size2[: len(size2) - len(size1)]
        return gap_size


class BilinearFlatOnly(BilinearFlat):
    def __init__(self, in1_features, in2_features, bias=False):
        super().__init__(in1_features, in2_features, bias=bias)


class LinearLayerNorm(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, tensor):
        return self.ln(self.linear(tensor))


def residual(func, x):
    return func(x) + x


def max_over_index(tensor, index):
    max_values, max_indices = tensor.max(index)
    return max_values


def _masked_softmax(softmax_fn, input, mask=None, *args, **kwargs):
    if mask is None:
        # _mask = torch.ones(input.size(), device=input.device)
        masked_input = input
    elif isinstance(mask, (list, tuple)):
        _mask = torch.tensor(mask, device=input.device)
        masked_input = input.masked_fill((1 - _mask.int()).bool(), float('-inf'))

    return softmax_fn(masked_input, *args, **kwargs)


def masked_softmax(input, mask=None, *args, **kwargs):
    return _masked_softmax(softmax_fn=F.softmax, input=input, mask=mask, *args, **kwargs)


def masked_log_softmax(input, mask=None, *args, **kwargs):
    '''
    >>> tensor = torch.tensor([1,2,3,4], dtype=torch.float)
    >>> log_probs = masked_log_softmax(tensor, [0, 1, 0, 1], dim=-1)
    >>> log_probs                                             # doctest: +SKIP
    tensor([   -inf, -2.1269,    -inf, -0.1269])              # doctest: +SKIP
    >>> log_probs.exp()                                       # doctest: +SKIP
    tensor([0.0000, 0.1192, 0.0000, 0.8808])                  # doctest: +SKIP
    '''

    return _masked_softmax(softmax_fn=F.log_softmax, input=input, mask=mask, *args, **kwargs)


def nll_without_reduction(input, target, *args, **kwargs):
    '''
    Example:

    >>> logits = torch.randn(3, 5, requires_grad=True)  # logits is of size N x C = 3 x 5
    >>> labels = torch.tensor([1, 0, 4])               # each element in labels has to have 0 <= value < C
    >>> log_probs = F.log_softmax(logits, dim=1)
    >>> nll = nll_without_reduction(log_probs, labels)
    >>> print(nll)                                                 # doctest: +SKIP
    tensor([1.4162, 2.9010, 2.6152], grad_fn=<ViewBackward0>)      # doctest: +SKIP
    '''
    input_size = input.size()
    target_size = target.size()
    assert input_size[:-1] == target_size

    num_classes = input_size[-1]

    reshaped_input = input.view(-1, num_classes)
    reshaped_target = target.view(-1)

    reshaped_nll = F.nll_loss(reshaped_input, reshaped_target, reduction='none')
    nll = reshaped_nll.view(target_size)

    return nll


def lengths_to_mask(lengths, max_length=None):
    """
    Example:

    >>> lengths_to_mask([3, 4, 6, 2])
    tensor([[1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0]])
    >>> lengths_to_mask([3, 4, 6, 2], max_length=8)
    tensor([[1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0]])
    """
    # This code is almost same with `rnnlib.get_indicator`
    # except the ouput's dimension and dtype

    if isinstance(lengths, (list, tuple)):
        lengths = torch.tensor(lengths, dtype=torch.int64)

    lengths_size = lengths.size()
    flat_lengths = lengths.view(-1, 1)

    if not max_length:
        max_length = lengths.max()
    unit_range = torch.arange(max_length)
    flat_range = unit_range.expand(flat_lengths.size()[0:1] + unit_range.size())
    flat_indicator = flat_range < flat_lengths

    return flat_indicator.view(lengths_size + (-1,)).long()


def id_tensor_to_mask(id_tensor, pad_token_id):
    '''
    >>> id_tensor_to_mask(torch.tensor([1, 2, 3, 4, 10, 10, 10, 10]), 10)
    tensor([1, 1, 1, 1, 0, 0, 0, 0])
    >>> id_tensor_to_mask(torch.tensor([1, 2, 3, 4, 10, 10, 20, 20]), [10, 20])
    tensor([1, 1, 1, 1, 0, 0, 0, 0])
    >>> id_tensor_to_mask(torch.tensor([[1, 2, 3, 4, 10, 10, 20, 20], [5, 6, 10, 10, 20, 20, 20, 20]]), [10, 20])
    tensor([[1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0]])
    '''
    if isinstance(pad_token_id, int):
        pad_token_ids = [pad_token_id]
    else:
        pad_token_ids = pad_token_id
        assert len(pad_token_ids) > 0
    del pad_token_id

    pad_token_id_iter = iter(pad_token_ids)
    accumulation = (id_tensor != next(pad_token_id_iter))

    for pad_token_id in pad_token_id_iter:
        accumulation = accumulation.logical_and(id_tensor != pad_token_id)

    return accumulation.long()


def _get_coll_dim(coll):
    depth = 0
    _coll = coll
    while isinstance(_coll, (list, tuple)):
        depth += 1
        _coll = _coll[0]
    return depth


def get_dim(coll):
    if isinstance(coll, torch.Tensor):
        return coll.dim()
    else:
        return _get_coll_dim(coll)


def _get_coll_size(coll):
    _coll = coll
    size = []
    while isinstance(_coll, (list, tuple)):
        size.append(len(_coll))
        _coll = _coll[0]
    return torch.Size(size)


def get_size(coll):
    if isinstance(coll, torch.Tensor):
        return coll.size()
    else:
        return _get_coll_size(coll)


def get_size_but_last(coll):
    return get_size(coll)[:-1]


def candidate_ids_to_mask(candidate_ids, vocab_size):
    '''
    >>> candidate_ids_to_mask([[0, 2, 4], [2, 3, 4]], vocab_size=6)
    tensor([[1., 0., 1., 0., 1., 0.],
            [0., 0., 1., 1., 1., 0.]])
    '''
    def to_tuple(*args):
        return args

    size_but_last = get_size_but_last(candidate_ids)
    mask = torch.zeros(*size_but_last, vocab_size)
    for indices in nest(*map(range, size_but_last)):
        candidates = get_elem(candidate_ids, indices)
        mask[to_tuple(*indices, candidates)] = 1

    return mask


def pad_sequence(sequence, pad, max_length=None):
    '''
    Pad nested sequence. `sequence` can be multi-dimensional lists, such as 3D or 4D arrays.

    >>> pad_sequence([[[1, 2, 3], [4, 5]],[[6, 7, 8, 9]]], pad=float('inf'))
    [[[1, 2, 3, inf], [4, 5, inf, inf]], [[6, 7, 8, 9]]]
    >>> pad_sequence([[[1, 2, 3], [4, 5]],[[6, 7, 8, 9]]], pad=float('inf'), max_length=6)
    [[[1, 2, 3, inf, inf, inf], [4, 5, inf, inf, inf, inf]], [[6, 7, 8, 9, inf, inf]]]
    '''

    from .. import iteration

    assert isinstance(sequence, (tuple, list))

    dim = 0
    seq = sequence
    while isinstance(seq, (tuple, list)):
        dim += 1
        if len(seq) == 0:
            break
        seq = seq[0]
    assert dim >= 2

    if max_length is None:
        def find_max_length(seq, depth):
            if depth < dim:
                return max(find_max_length(elem, depth + 1) for elem in seq)
            else:
                return len(seq)

        max_length = find_max_length(sequence, 1)

    padded_sequence = iteration.apply_recursively(sequence, coll_fn=list)

    def pad_recursively(seq, depth):
        if depth < dim:
            for elem in seq:
                pad_recursively(elem, depth + 1)
        else:
            while len(seq) < max_length:
                seq.append(pad)

    pad_recursively(padded_sequence, 1)

    return padded_sequence


def _has_param_grad(param):
    grad = param.grad
    return grad is not None and bool((grad != 0).any())


def has_grad(obj):
    if isinstance(obj, nn.Module):
        module = obj
        return any(map(_has_param_grad, module.parameters()))
    else:
        return _has_param_grad(obj)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def seed_everything(seed):
    # This code is copied from https://github.com/shijx12/KQAPro_Baselines/blob/7cea2738fd095a2c17594d492923ee80a212ac0f/utils/misc.py
    '''
    Set the seed of the entire development environment
    :param seed:
    :return:
    '''
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def disable_benchmark():
    # `torch.backends.cudnn.benchmark` enhances speed when input size is identical,
    # but the speed is reduced for dataset with various input sizes.
    #
    # The value if False by default
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/8

    torch.backends.cudnn.benchmark = False


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
