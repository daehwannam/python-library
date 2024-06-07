
"""
Data processing
"""

def _get_ratio(ratio, percent):
    assert (ratio is not None and percent is None) or (ratio is None and percent is not None)
    if ratio is None:
        ratio = percent / 100
    else:
        assert percent is None
    return ratio

def split_dataset(dataset, *, ratio=None, percent=None, round_fn=round):
    _ratio = _get_ratio(ratio, percent)
    train_set_size = round_fn(len(dataset) * _ratio)
    train_set = dataset[:train_set_size]
    val_set = dataset[train_set_size:]
    return train_set, val_set


def extract_portion(dataset, *, ratio=None, percent=None, round_fn=round):
    _ratio = _get_ratio(ratio, percent)
    dataset_size = round_fn(len(dataset) * _ratio)
    portion = dataset[:dataset_size]
    return portion
