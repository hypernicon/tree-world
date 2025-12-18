import torch
import math
from typing import Optional

from ..fourier import make_alphas, make_lattice_basis


class LocationMetric(torch.nn.Module):
    def __init__(self, location_dim: int, dim: int=2, scale: float=10.0, ratio: float=math.sqrt(2.0), alphas_trainable: bool=False,
                 metric_rank: Optional[int]=None):
        super().__init__()
        self.location_dim = location_dim

        alphas = make_alphas(location_dim, dim, scale, ratio)
        lattice_basis, K, K_dagger = make_lattice_basis(alphas, dim)

        if alphas_trainable:
            self.alphas = torch.nn.Parameter(alphas)
        else:
            self.alphas = torch.nn.Buffer(alphas)

        self.lattice_basis = torch.nn.Buffer(lattice_basis)
        self.K = torch.nn.Buffer(K)
        self.K_dagger = torch.nn.Buffer(K_dagger)
    
        if metric_rank is not None:
            self.metric_rank = metric_rank
        else:
            self.metric_rank = location_dim

        self.square = self.location_dim == self.metric_rank
        if self.square:   
            # TODO: set initial values to reflect alphas and K
            self.metric = torch.nn.Linear(location_dim, location_dim, bias=False)
        
        else:
            self.metric_q = torch.nn.Linear(location_dim, metric_rank, bias=False)
            self.metric_k = torch.nn.Linear(metric_rank, location_dim, bias=False)

        self.scale_factor = self.metric_rank ** -0.5
        
    def square_affinity_2d(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        assert self.square, "square_affinity is only supported for square metrics"
        if prepared_k:
            return (location1 * location2).sum(dim=-1)
        else:
            return (location1 * self.metric(location2)).sum(dim=-1)

    def low_rank_affinity_2d(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        assert not self.square, "low_rank_affinity is only supported for low-rank metrics"
        if prepared_k:
            return (self.metric_q(location1) * location2).sum(dim=-1)
        else:
            return (self.metric_q(location1) * self.metric_k(location2)).sum(dim=-1)
    
    def square_affinity_nd(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        assert self.square, "square_affinity is only supported for square metrics"
        if prepared_k:
            return torch.bmm(location1, location2.transpose(-2, -1)).sum(dim=-1)
        else:
            return torch.bmm(location1, self.metric(location2).transpose(-2, -1)).sum(dim=-1)
    
    def low_rank_affinity_nd(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False):
        assert not self.square, "low_rank_affinity is only supported for low-rank metrics"
        if prepared_k:
            return torch.bmm(self.metric_q(location1), location2.transpose(-2, -1)).sum(dim=-1)
        else:
            return torch.bmm(self.metric_q(location1), self.metric_k(location2).transpose(-2, -1)).sum(dim=-1)
    
    def log_affinity(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False, scale_by_operator_norm: bool=True):
        if self.square:
            if location2.ndim == 2:
                base = self.square_affinity_2d(location1, location2, prepared_k)
            else:
                base = self.square_affinity_nd(location1, location2, prepared_k)
        else:
            if location2.ndim == 2:
                base = self.low_rank_affinity_2d(location1, location2, prepared_k)
            else:
                base = self.low_rank_affinity_nd(location1, location2, prepared_k)

        if scale_by_operator_norm:
            base = base / (self.metric_operator_norm() + 1e-8)

        return self.scale_factor * base
        
    def prepare_k(self, location: torch.Tensor):
        if self.square:
            return self.metric(location)
        else:
            return self.metric_k(location)
    
    def prepare_q(self, location: torch.Tensor):
        if self.square:
            return self.metric(location)
        else:
            return self.metric_q(location)
    
    def metric_operator_norm(self):
        if self.square:
            return torch.linalg.matrix_norm(self.metric.weight, ord=2)
        else:
            # Q = U Rq
            _, Rq = torch.linalg.qr(self.metric_q.weight, mode="reduced")      # Rq: (..., r, r)

            # K = V Rk  
            _, Rk = torch.linalg.qr(self.metric_k.weight, mode="reduced")   # Rk: (..., r, r)

            core = Rq @ Rk.mH    # (..., r, r)  (use .mH so this works for real and complex)
            return torch.linalg.matrix_norm(core, ord=2)    # (...,)