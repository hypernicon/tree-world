import torch

class TorchBlocker:
    def __init__(self, module: torch.nn.Module):
        self.module = module
