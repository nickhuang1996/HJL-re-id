import torch


def get_default_device():
    """Get default device for `*.to(device)`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


if __name__ == '__main__':
    device = get_default_device()
    print(device)
