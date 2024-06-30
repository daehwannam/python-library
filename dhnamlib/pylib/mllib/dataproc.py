
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

def split_train_val(
        dataset, *, ratio=None, percent=None, round_fn=round,
        train_set_size=None, val_set_size=None
):
    if train_set_size is None:
        if val_set_size is None:
            _ratio = _get_ratio(ratio, percent)
            train_set_size = round_fn(len(dataset) * _ratio)
        else:
            train_set_size = len(dataset) - val_set_size
    else:
        assert (val_set_size is None) or (train_set_size + val_set_size == len(dataset))

    train_set = dataset[:train_set_size]
    val_set = dataset[train_set_size:]
    return train_set, val_set


def extract_portion(dataset, *, ratio=None, percent=None, round_fn=round):
    _ratio = _get_ratio(ratio, percent)
    dataset_size = round_fn(len(dataset) * _ratio)
    portion = dataset[:dataset_size]
    return portion
