import torch

class TorchBlocker:
    def __init__(self, module: torch.nn.Module):
        self.module = module


def map_index_to_space(indices, R, gamma):
    return gamma * (indices - R)


def map_space_to_preindex(x, R, gamma):
    return x / gamma + R


def gamma_from_real_magnitude(real_magnitude, R):
    """
       For a location spanning -R to R, the gamma is the distance between each point on the grid,
       and the real magnitude is the distance between the origin and the edge of the location along an axis.
    """
    return real_magnitude / R

def real_magnitude_from_gamma(gamma, R):
    """
       For a location spanning -R to R, the gamma is the distance between each point on the grid,
       and the real magnitude is the distance between the origin and the edge of the location along an axis.
    """
    return gamma * R
