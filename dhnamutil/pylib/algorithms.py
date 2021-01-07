
import random


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
        for j in range(l+1, r+1):
            if key(lst[j]) < key(lst[l]):
                i += 1
                lst[i], lst[j] = lst[j], lst[i]

        # move pivot to correct location
        lst[i], lst[l] = lst[l], lst[i]

        # recursively partition one side only
        if index == i:
            return lst[i]
        elif index < i:
            return select(lst, l, i-1, index)
        else:
            return select(lst, i+1, r, index)

    if items is None or len(items) < 1:
        return None

    if item_index < 0 or item_index > len(items) - 1:
        raise IndexError()

    return select(items, 0, len(items) - 1, item_index)
