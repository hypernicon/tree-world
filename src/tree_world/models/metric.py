import torch
import math
from typing import Optional, Union

from ..fourier import make_alphas, make_lattice_basis


class PseudoMetric(torch.nn.Module):
    def __init__(self, vector_dim: int, dim: int=2, scale: float=10.0, ratio: float=math.sqrt(2.0), alphas_trainable: bool=False,
                 metric_rank: Optional[int]=None):
        super().__init__()
        self.vector_dim = vector_dim

        # alphas = make_alphas(vector_dim, dim, scale, ratio)
        # lattice_basis, K, K_dagger = make_lattice_basis(alphas, dim)

        # if alphas_trainable:
        #     self.alphas = torch.nn.Parameter(alphas)
        # else:
        #     self.alphas = torch.nn.Buffer(alphas)

        # self.lattice_basis = torch.nn.Buffer(lattice_basis)
        # self.K = torch.nn.Buffer(K)
        # self.K_dagger = torch.nn.Buffer(K_dagger)
    
        if metric_rank is not None:
            self.metric_rank = min(vector_dim, metric_rank)
        else:
            self.metric_rank = vector_dim

        self.metric = torch.nn.Linear(self.vector_dim, self.metric_rank, bias=False)

        self.scale_factor = self.metric_rank ** -0.5
        
    def affinity_2d(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        if prepared_k:
            return (self.metric(location1) * location2).sum(dim=-1)
        else:
            return (self.metric(location1) * self.metric(location2)).sum(dim=-1)
    
    def affinity_nd(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        if prepared_k:
            return torch.bmm(self.metric(location1), location2.transpose(-2, -1)).sum(dim=-1)
        else:
            return torch.bmm(self.metric(location1), self.metric(location2).transpose(-2, -1)).sum(dim=-1)
    
    def log_affinity(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False, scale_by_operator_norm: bool=True):
        if location2.ndim == 2:
            base = self.low_rank_affinity_2d(location1, location2, prepared_k)
        else:
            base = self.low_rank_affinity_nd(location1, location2, prepared_k)

        if scale_by_operator_norm:
            base = base / (self.metric_operator_norm() + 1e-8)

        return self.scale_factor * base

    def psuedo_distance(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        if prepared_k:
            diff = self.metric(location1) - location2
        else:
            diff = self.metric(location1 - location2)
        return torch.norm(diff, dim=-1)
        
    def prepare_k(self, location: torch.Tensor):
        return self.metric(location)
    
    def prepare_q(self, location: torch.Tensor):
        return self.metric(location)

    def cross_affinity(self, vector1: torch.Tensor, vector2: torch.Tensor, projected: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        if not projected:
            vector1 = self.metric(vector1)
            vector2 = self.metric(vector2)

            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    while scale.ndim < vector1.ndim:
                        scale = scale[..., None]
                    
            vector1 = vector1 * scale
            vector2 = vector2 * scale

        affinity = (vector1[..., None, :] * vector2[..., None, :, :]).sum(dim=-1)

        return affinity
    
    def cross_distance(self, vector1: torch.Tensor, vector2: torch.Tensor, squared: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        vector1 = self.metric(vector1)
        vector2 = self.metric(vector2)

        if scale is not None:
            if isinstance(scale, torch.Tensor):
                while scale.ndim < vector1.ndim:
                    scale = scale[..., None]
                    
            vector1 = vector1 * scale
            vector2 = vector2 * scale

        affinity = self.cross_affinity(vector1, vector2, projected=True, scale=None)
        
        square1 = vector1.pow(2).sum(dim=-1)
        square2 = vector2.pow(2).sum(dim=-1)
        dist = (square1[..., None] + square2[..., None, :] - 2 * affinity)

        if squared:
            return dist
        else:
            return dist.pow(0.5)
    
    def metric_operator_norm(self):
        # Q = U R
        _, R = torch.linalg.qr(self.metric.weight.float(), mode="reduced")      # Rq: (..., r, r)
        
        core = R @ R.mH    # (..., r, r)  (use .mH so this works for real and complex)
        return torch.linalg.matrix_norm(core, ord=2).to(self.metric.weight.dtype)     # (...,)
    