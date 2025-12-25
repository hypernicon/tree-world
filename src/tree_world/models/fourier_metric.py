import torch
import torch.distributions as D
import math
from typing import Optional, Union, Callable, Tuple

from ..fourier import make_alphas, make_lattice_basis, solve_for_deltas
from .mixture import IndexedMixture


def check_valid_location(location: torch.Tensor, batch_lengths: Optional[torch.Tensor]=None, __idx: Optional[torch.Tensor]=None):
    if torch.isnan(location).any() or torch.isinf(location).any():
        print(f"location BAD: nan={torch.isnan(location).any().item()} inf={torch.isinf(location).any().item()}")
        print(f"location stats: min={location.nan_to_num().min().item()} max={location.nan_to_num().max().item()}")
        raise ValueError(f"location is nan/inf")
    
    reshaped_location = location.reshape(-1, 2)
    location_norms = torch.norm(reshaped_location, dim=-1)
    if batch_lengths is not None:
        batch_mask = torch.arange(location.shape[1], device=location.device)[None, :] >= batch_lengths[:, None]
        if location.ndim == 4:
            # B, T, S, L -- two time dims!
            batch_mask = batch_mask[..., None].logical_or(batch_mask[..., None, :])
            if __idx is not None:
                batch_mask = batch_mask.gather(dim=-1, index=__idx)

        while batch_mask.ndim < location.ndim:
            batch_mask = batch_mask[..., None]
        batch_mask = batch_mask.expand_as(location).reshape(-1, 2)
        batch_mask = batch_mask.max(dim=-1).values
        location_norms = torch.masked_fill(location_norms, batch_mask, 1.0)

    if not torch.allclose(location_norms, torch.ones_like(location_norms), atol=1e-1):
        print(f"location_norms: {location_norms.shape} -- {location_norms[:100].detach().cpu().float().numpy().tolist()}")
        print(f"location is not on the unit sphere: {reshaped_location[:100].detach().cpu().float().numpy().tolist()}")
        print(f"max norm: {location_norms.max().item()}, min norm: {location_norms.min().item()}")
        if batch_lengths is not None:
            print(f"batch_lengths: {batch_lengths.shape} -- {batch_lengths.detach().cpu().numpy().tolist()}")
        else:
            print("NO BATCH LENGTHS PROVIDED")
        first_index = location_norms.argmin()
        position = []
        current_index = first_index.item()
        for i in range(location.ndim - 2):
            position.append(current_index % location.shape[i])
            print(f"current_index: {current_index}, location.shape[i]: {location.shape[i]}, position: {position[-1]}")
            current_index = current_index // location.shape[i]
        position.append(current_index)
        print(f"first_index: {first_index}, position: {position}")
        raise ValueError(f"location_norms is not on the unit sphere")


class FourierMetric(torch.nn.Module):
    def __init__(self, location_dim: int, dim: int, scale: float=1.0, ratio: float=math.sqrt(2.0), alphas_trainable: bool=False):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.ratio = ratio
        self.location_dim = location_dim
        self.J = location_dim // (2 * (dim + 1))
        assert location_dim % (2 * (dim + 1)) == 0, f"location_dim={location_dim} must be divisible by 2*(dim+1)={2*(dim+1)}"

        alphas = make_alphas(location_dim, dim, scale, ratio)
        lattice_basis, K, K_dagger = make_lattice_basis(alphas, dim)

        if alphas_trainable:
            self.alphas = torch.nn.Parameter(alphas)
        else:
            self.register_buffer('alphas', alphas)

        self.register_buffer('lattice_basis', lattice_basis)
        self.register_buffer('K', K)
        self.register_buffer('K_dagger', K_dagger)

    def reshape_to_components(self, location: torch.Tensor):
        shape = tuple(list(location.shape[:-1]) + [self.J, self.dim + 1, 2])

        return location.view(*shape)

    def compute_displacements(self, location1: torch.Tensor, location2: torch.Tensor):
        # distance is the estimated displacement plus errors
        location1 = self.reshape_to_components(location1)  # Now has shape (..., J, d+1, 2)
        location2 = self.reshape_to_components(location2)  # Now has shape (..., J, d+1, 2)

        s2c1 = location2[..., 1] * location1[..., 0].float()
        c2s1 = location2[..., 0] * location1[..., 1].float()
        c2c1 = location2[..., 0] * location1[..., 0].float()
        s2s1 = location2[..., 1] * location1[..., 1].float()
        delta_thetas = torch.atan2(s2c1 - c2s1, c2c1 + s2s1 + 1e-6)

        # estimated displacements from thetas, made as small as possible solving across alphas -- but may not agree!
        # deltas has shape (..., J, d)
        deltas = solve_for_deltas(
            delta_thetas.view(delta_thetas.shape[:-2] + (-1,)), 
            self.K_dagger.to(delta_thetas.dtype), 
            self.lattice_basis.to(delta_thetas.dtype), 
            self.alphas.to(delta_thetas.dtype)
        )
        mean_deltas = deltas.mean(dim=-2)

        return deltas.to(location1.dtype), mean_deltas.to(location1.dtype)
    
    def pseudo_distance(self, location1: torch.Tensor, location2: torch.Tensor, squared: bool=False):
        deltas, mean_deltas = self.compute_displacements(location1, location2)
        J = deltas.shape[-2]
        assert J > 1, "J must be greater than 1"
        
        dev_deltas = deltas - mean_deltas[..., None, :]
        variances = dev_deltas.square().sum(dim=-1).mean(dim=-1) * (J / (J - 1))  # (...)

        squared_distances = mean_deltas.square().sum(dim=-1)

        final_squared_distances = squared_distances + variances

        if squared:
            return final_squared_distances
        else:
            return (final_squared_distances + 1e-12).sqrt()
    
    def cross_distance(self, location1: torch.Tensor, location2: torch.Tensor, squared: bool=False):
        return self.pseudo_distance(location1[:, :, None, ...], location2[:, None, :, ...], squared=squared)

    def block_rotate(self, location: torch.Tensor, thetas: torch.Tensor):
        D = location.shape[-1] // 2

        cos_thetas = torch.cos(thetas).view(-1, D)
        sin_thetas = torch.sin(thetas).view(-1, D)

        location_blocks = location.view(-1, D, 2)
        location_cos_thetas = location_blocks * cos_thetas[..., None]
        location_sin_thetas = location_blocks * sin_thetas[..., None]

        next_location = torch.stack([
            location_cos_thetas[..., 0] - location_sin_thetas[..., 1],
            location_sin_thetas[..., 0] + location_cos_thetas[..., 1],
        ], dim=-1)

        return next_location.reshape_as(location)
    
    def apply_displacement(self, displacement: torch.Tensor, location: torch.Tensor):
        # displacement has shape (..., d)

        K = self.K
        while K.ndim < displacement.ndim + 1:
            K = K[None, ...]
        delta_thetas = (K @ displacement[..., None]).squeeze(-1) # (..., d+1)

        delta_thetas = delta_thetas[..., None, :]

        alphas = self.alphas[..., None]
        while alphas.ndim < delta_thetas.ndim:
            alphas = alphas[None, ...]
        delta_thetas = delta_thetas * alphas

        return self.block_rotate(location, delta_thetas)

    def sample(self, shape: Tuple[int, ...] = torch.Size(), device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        thetas = torch.empty(shape + (self.location_dim // 2 // (self.dim + 1),), device=device, dtype=dtype).uniform_(-math.pi, math.pi)
        thetas = thetas[..., None].expand(thetas.shape + (self.dim + 1,))  # <-- within the same module, phase is shared
        return torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=-1).view(shape + (self.location_dim,))


class FourierCodeDistribution(D.Distribution):
    support = D.constraints.real
    has_rsample = True

    def __init__(self, metric: FourierMetric, reference_location: torch.Tensor, scale: torch.Tensor, 
                 batch_lengths: Optional[torch.Tensor]=None, __idx: Optional[torch.Tensor]=None, validate_args=None):
        self.metric = metric
        self.reference_location = reference_location
        self.batch_lengths = batch_lengths
        # check_valid_location(reference_location, batch_lengths, __idx)

        self.dtype = reference_location.dtype
        self.device = reference_location.device
        self.scale = scale

        batch_shape = reference_location.shape[:-1]

        assert scale.shape == () or scale.shape == batch_shape
        if scale.shape == ():
            self.scale = self.scale.expand(batch_shape)

        # jac_weights will have shape (..., d+1)
        reference_location = self.metric.reshape_to_components(reference_location)  # Now has shape (..., J, d+1, 2)
        alphas = self.metric.alphas
        while alphas.ndim < reference_location.ndim - 1:
            alphas = alphas[None, ...]
        alphas = alphas.transpose(-2, -1) # (..., J, d+1)
        jac_weights = (alphas.square() * reference_location.square().sum(dim=-1)).sum(dim=-2)

        # K is (d+1, d); need (wK)^T K = (..., d+1, d+1)
        K = self.metric.K
        while K.ndim < jac_weights.ndim + 1:
            K = K[None, ...]
        wK = (jac_weights[..., None] * K)
        jac_squared = K.transpose(-2, -1) @ wK

        eps = 1e-6
        I = torch.eye(self.metric.dim, device=jac_squared.device, dtype=torch.float32)
        while I.ndim < jac_squared.ndim:
            I = I[None, ...]
        jac_squared = jac_squared.float() + eps * I
        logdet = torch.logdet(jac_squared)
        self.logdet_jac = - 0.5 * logdet.to(self.dtype)

        super().__init__(batch_shape=batch_shape, event_shape=(self.metric.location_dim,), validate_args=validate_args)

    def sample(self, sample_shape: Tuple[int, ...] = torch.Size()):
        # sample Delta x ~ N(0, I_M)
        u = torch.randn(sample_shape + self.batch_shape + (self.metric.dim,), device=self.device, dtype=self.dtype)
        scale = self.scale[..., None]
        while scale.ndim < u.ndim:
            scale = scale[None, ...]
        deltas = scale * u
        locations = self.metric.apply_displacement(deltas, self.reference_location)

        # check_valid_location(locations, self.batch_lengths)
        return locations, deltas
    
    rsample = sample

    def log_prob(self, locations: torch.Tensor, displacements: Optional[torch.Tensor]=None):
        if displacements is None:
            _, displacements = self.metric.compute_displacements(self.reference_location, locations)

        scale = self.scale[..., None]
        while scale.ndim < displacements.ndim:
            scale = scale[None, ...]

        displacements = displacements / scale

        gaussian_log_prob = (
            - 0.5 * displacements.square().sum(dim=-1)
            - self.metric.dim * (0.5 * math.log(2.0 * math.pi) + torch.log(self.scale))
        )

        return gaussian_log_prob + self.logdet_jac


class IndexedFourierMixture(IndexedMixture):
    def __init__(self, logits: torch.Tensor, metric: FourierMetric, reference_location: torch.Tensor, scale: torch.Tensor, 
                 batch_lengths: Optional[torch.Tensor]=None):
        # check_valid_location(reference_location, batch_lengths)
        self.batch_lengths = batch_lengths
        super().__init__(logits, self.distribution_builder, metric, reference_location, scale, batch_lengths)
    
    def distribution_builder(self, metric: FourierMetric, reference_location: torch.Tensor, scale: torch.Tensor, 
                                   batch_lengths: Optional[torch.Tensor]=None, __idx: Optional[torch.Tensor]=None) -> D.Distribution:
        return FourierCodeDistribution(metric, reference_location, scale, batch_lengths, __idx)

    def sample(self, sample_shape: Tuple[int, ...] = torch.Size()):
        locations, displacements = super().sample(sample_shape)
        # check_valid_location(locations, self.batch_lengths)
        return locations, displacements
    
    def rsample(self, sample_shape: Tuple[int, ...] = torch.Size()):
        locations, displacements = super().rsample(sample_shape)
        # check_valid_location(locations, self.batch_lengths)
        return locations, displacements
