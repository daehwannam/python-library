
import random
import math

from . import iteration as iter_util
from .structure import AttrDict
from collections import namedtuple


def quickselect(items, item_index, key=lambda x: x):
    """ quickselect algorithm
    the list is sorted, so the items from index 0 to (item_index - 1) are smaller than
    those from item_index to (len(items) - 1)

    * code reference: https://www.koderdojo.com/blog/quickselect-algorithm-in-python

    :param items: a list
    :param item_index: a 
    :returns: 
    :rtype: 

    """
    def select(lst, l, r, index):
        # base case
        if r == l:
            return lst[l]

        # choose random pivot
        pivot_index = random.randint(l, r)

        # move pivot to beginning of list
        lst[l], lst[pivot_index] = lst[pivot_index], lst[l]

        # partition
        i = l
        for j in range(l + 1, r + 1):
            if key(lst[j]) < key(lst[l]):
                i += 1
                lst[i], lst[j] = lst[j], lst[i]

        # move pivot to correct location
        lst[i], lst[l] = lst[l], lst[i]

        # recursively partition one side only
        if index == i:
            return lst[i]
        elif index < i:
            return select(lst, l, i - 1, index)
        else:
            return select(lst, i + 1, r, index)

    if items is None or len(items) < 1:
        return None

    if item_index < 0 or item_index > len(items) - 1:
        raise IndexError()

    return select(items, 0, len(items) - 1, item_index)


# Sparse vector functions
def vec_increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.
    """
    for f, v in list(d2.items()):
        d1[f] = d1.get(f, 0) + v * scale


def vec_sum(vectors):
    total = {}
    for vec in vectors:
        vec_increment(total, 1., vec)
    return total


def vec_div(vec, scalar):
    return {feature: value / scalar for feature, value in vec.items()}


def vec_dot(vec1, vec2):
    if len(vec2) < len(vec1):
        vec1, vec2 = vec2, vec1
    return sum(
        v * vec2.get(k, 0)
        for k, v in vec1.items())


def vec_cosine(vec1, vec2):
    denominator = math.sqrt(vec_dot(vec1, vec1) * vec_dot(vec2, vec2))
    if denominator == 0:
        return 0
    return vec_dot(vec1, vec2) / denominator


def sparse_kmeans(examples, K, max_num_iters=float('inf')):
    '''
    examples: list of examples, each example is a feature-to-number dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    max_num_iters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    centers = random.sample(examples, K)
    assignments = None

    # PRE-COMPUTING NORMS =============================================================
    def get_squared_norms():
        return [sum(value ** 2 for value in center.values()) for center in centers]
    center_squared_norms = get_squared_norms()

    def get_l2_loss(cluster_idx, example):
        l2_loss = center_squared_norms[cluster_idx]
        center = centers[cluster_idx]
        for key, value in example.items():
            center_value = center.get(key, 0)
            l2_loss += value ** 2 - 2 * value * center_value
        return l2_loss
    # ============================================================= PRE-COMPUTING NORMS

    for iter_cnt in iter_util.exrange(max_num_iters):
        prev_assignments = assignments
        assignments = [None] * len(examples)

        # update assignments
        for example_idx, example in enumerate(examples):
            assignments[example_idx] = min(range(K),
                                           key=lambda x: get_l2_loss(x, example))
        # terminate when converging
        if prev_assignments == assignments:
            # print('k-means converged')
            break

        # update centers
        example_clusters = [[] for _ in range(K)]
        for example_idx, example in enumerate(examples):
            cluster_idx = assignments[example_idx]
            example_clusters[cluster_idx].append(example)

        centers = [vec_div(vec_sum(example_cluster), len(example_cluster))
                   for example_cluster in example_clusters]

        # PRE-COMPUTING NORMS =============================================================
        center_squared_norms = get_squared_norms()
        # ============================================================= PRE-COMPUTING NORMS

    else:
        print('max iteration')

    # compute clusters
    clusters = tuple(set() for _ in range(K))
    for example_idx in range(len(examples)):
        clusters[assignments[example_idx]].add(example_idx)

    # compute loss
    loss = 0
    for example_idx, example in enumerate(examples):
        loss += get_l2_loss(assignments[example_idx], example)

    # result
    result = AttrDict(centers=centers,
                      assignments=assignments,
                      clusters=clusters,
                      loss=loss)
    return result


def test_sparse_kmeans():
    random.seed(42)
    x1 = dict(f1=0, f2=0)         # x1 => (0, 0)
    x2 = dict(f1=0, f2=1)         # x2 => (0, 1)
    x3 = dict(f1=0, f2=2)         # x3 => (0, 2)
    x4 = dict(f1=0, f2=3)         # x4 => (0, 3)
    x5 = dict(f1=0, f2=4)         # x5 => (0, 4)
    x6 = dict(f1=0, f2=5)         # x6 => (0, 5)
    examples = [x1, x2, x3, x4, x5, x6]
    result = sparse_kmeans(examples, 2, max_num_iters=10)
    print(result.centers, result.assignments, result.loss, sep='\n')

