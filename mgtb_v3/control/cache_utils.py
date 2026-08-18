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


def replay_last_logits(model, tokens, cache, device=None):
    """Replay the last retained token on a cache cropped to prefix[:-1]."""
    import torch

    if not tokens:
        raise ValueError("cannot replay an empty prefix")
    device = device or getattr(model, "device", None)
    input_ids = torch.tensor([[int(tokens[-1])]], device=device)
    with torch.no_grad():
        output = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    return output.logits[:, -1, :], getattr(output, "past_key_values", None)


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
