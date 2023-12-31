
from abc import ABCMeta, abstractmethod

from ..iteration import distinct_pairs, not_none_valued_pairs
from ..decoration import construct
from ..function import compose
from ..klass import abstractfunction


def get_measure(name, higher_better: bool):
    return dict(name=name, higher_better=higher_better)


def get_performance(*args, **kwargs):
    '''Pairs of measure names and values'''
    return dict(distinct_pairs(*args, **kwargs))


@construct(compose(dict, distinct_pairs))
def get_init_performance(measures):
    for measure in measures:
        name = measure['name']
        higher_better = measure['higher_better']
        init_score = float('-inf') if higher_better else float('inf')
        yield [name, init_score]


def is_better_value(value1, value2, higher_better):
    if higher_better:
        return value1 > value2
    else:
        return value1 < value2


def is_better_performance(performance1, performance2, measures):
    assert len(performance1) == len(performance2) == len(measures)
    for measure in measures:
        score1 = performance1[measure['name']]
        score2 = performance2[measure['name']]
        higher_better = measure['higher_better']
        if is_better_value(score1, score2, higher_better):
            return True
        elif is_better_value(score2, score1, higher_better):
            return False
    return False


def get_init_status(measures, update_unit=None):
    return dict(not_none_valued_pairs(
        measures=measures,
        update_unit=update_unit,
        last_update_num=0,
        best_update_num=0,
        last_performance=get_init_performance(measures),
        best_performance=get_init_performance(measures),
        history=[]))


def update_status(status, performance, update_num=None):
    if update_num is None:
        _update_num = status['last_update_num'] + 1
    else:
        assert status['last_update_num'] < update_num
        _update_num = update_num

    status.update(
        last_update_num=_update_num,
        last_performance=performance)

    status['history'].append(
        dict(last_update_num=_update_num,
             performance=performance))

    updating_best = is_better_performance(performance, status['best_performance'], status['measures'])

    if updating_best:
        status.update(
            best_update_num=_update_num,
            best_performance=performance)

    return updating_best
