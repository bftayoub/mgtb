from __future__ import annotations


def crop_hf_cache(cache, max_length: int):
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    if isinstance(cache, tuple):
        return tuple(_crop_layer(layer, max_length) for layer in cache)
    return cache


def _crop_layer(layer, max_length: int):
    if isinstance(layer, tuple):
        return tuple(_crop_tensor(tensor, max_length) for tensor in layer)
    return _crop_tensor(layer, max_length)


def _crop_tensor(tensor, max_length: int):
    if not hasattr(tensor, "shape") or len(tensor.shape) < 3:
        return tensor
    seq_dim = -2
    slices = [slice(None)] * len(tensor.shape)
    slices[seq_dim] = slice(0, max_length)
    return tensor[tuple(slices)]
