import torch


def recursive_to_device(input, device):
    """NOTE: If input is dict/list/tuple, it is changed in place."""
    if isinstance(input, torch.Tensor):
        # print('=> IS torch.Tensor')
        # print('=> input.device before to_device: {}'.format(input.device))
        input = input.to(device)
        # print('=> input.device after to_device: {}'.format(input.device))
    elif isinstance(input, dict):
        for k, v in input.items():
            input[k] = recursive_to_device(v, device)
    # Optimizer: only the key is 'params' that will be put on gpu(s).
    elif isinstance(input, (list, tuple)):
        input = [recursive_to_device(v, device) for v in input]
    return input
