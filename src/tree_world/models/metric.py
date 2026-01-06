import torch
import torch.distributions as D
import math
from typing import Optional, Union, Tuple

from .mixture import IndexedMixture


LOG2PI = math.log(2.0 * math.pi)


def stable_left_inverse(E: torch.Tensor, tau: float = 1e-3, suppress_warnings: bool = False) -> torch.Tensor:
    """
    E: (M,N)  (M << N)
    returns X: (N,M) ≈ right-inverse
    """
    Ef = E.float()
    M, N = Ef.shape
    I = torch.eye(N, device=E.device, dtype=torch.float32)
    K = Ef.T @ Ef + tau * I                     # (M,M)
    # Solve K X = E.T  -> X = K^{-1} E.T
    X = torch.linalg.solve(K, Ef.T)               # (N, M)

    # Attempt some error correction to account for tau
    error_correction = (X @ Ef).diagonal().abs().mean()

    # try to rebalance the diagonal to one, but only if the diagonal is big enough
    if error_correction > 1e-2:
        X = X / error_correction
    elif not suppress_warnings:
        print(f"WARNING: diagonal average {error_correction} is too small; left inverse may be unstable")

    return X.to(E.dtype)


class EmbeddedLowRankGaussian(D.Distribution):
    """
    Intrinsic (manifold) distribution:
        u ~ N(0, I_M)
        z = loc + scale ⊙ (E u)
    where:
        E: (M, N), full row-rank (ideally) embedding matrix, M << N
        E_dagger: (N, M) left inverse so that E_dagger @ E ≈ I_N
    """
    support = D.constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,          # batch+(E,)
        E: torch.Tensor,            # (M,E)
        scale: torch.Tensor,        # batch+(E,)  (can be scalar-broadcasted)
        E_dagger: Optional[torch.Tensor] = None,     # (E,M) optional; if None compute
        validate_args=None,
        tau: float = 1e-3,
    ):
        self.loc = loc
        self.E = E
        self.scale = scale
        self.tau = float(tau)

        M, N = E.shape
        self.beta = ((math.sqrt(2) / 2) * (M / N) ** 2) # scaling to make average distance = 1

        assert loc.shape[-1] == N, f"loc last dim {loc.shape[-1]} != E second dim {N}"
        assert isinstance(scale, float) or scale.shape == loc.shape[:-1] or scale.shape == (1,) or scale.shape == (), \
            f"scale shape {scale.shape} must broadcast to loc {loc.shape[:-1]}"

        if E_dagger is None:
            E_dagger = stable_left_inverse(E, tau=self.tau)

        assert E_dagger.shape == (N, M)
        
        self.E_dagger = E_dagger
        self.logdet_E = self._log_det_E()

        super().__init__(batch_shape=loc.shape[:-1], event_shape=(E,), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        dtype = self.loc.dtype
        device = self.loc.device
        M, N = self.E.shape

        scale = self.scale
        if isinstance(scale, torch.Tensor):
            while scale.ndim < len(sample_shape) + len(self.batch_shape):
                scale = scale[None, ...]

            scale = scale[..., None]

        # sample u ~ N(0, I_M)
        u = torch.randn(sample_shape + self.batch_shape + (M,), device=device, dtype=dtype)
        u_scaled = (self.beta * scale) * u

        u_flat = u_scaled.reshape(-1, M)

        delta_z = (u_flat @ self.E_dagger.T).reshape(sample_shape + self.batch_shape + (N,))
        return self.loc + delta_z, u
    
    sample = rsample

    def _log_det_E(self):
        M, N = self.E.shape
        beta = (math.sqrt(2) / 2) * ((M / N) ** 2)
        beta2 = beta ** 2
        Ef = self.E.float()
        ETE = Ef.T @ Ef
        K = ETE / beta2 + self.tau * torch.eye(N, device=self.E.device, dtype=torch.float32)
        return 0.5 * torch.logdet(K).to(self.E.dtype)
    
    def log_prob(self, value: torch.Tensor, u: Optional[torch.Tensor]=None) -> torch.Tensor:
        scale = self.scale
        if isinstance(scale, torch.Tensor):
            while scale.ndim < value.ndim - 1:
                scale = scale[None, ...]

        else:
            scale = torch.tensor(scale, dtype=value.dtype)

        if u is None:
            M, N = self.E.shape
            factor = (N / M) * math.sqrt(2)
            E = self.E
            while E.ndim < value.ndim + 1:
                E = E[None, ...]

            u = (E @ (value - self.loc)[..., None]).squeeze(-1) 
            s = scale
            if s.ndim > 0:
                s = s[..., None]
            u = u * (factor / s)
            
        else:
            M = u.shape[-1]
        
        logp_u = -0.5 * u.square().sum(dim=-1) - (M/2) * math.log(2*math.pi)
        # assert (logp_u <= 0.0).all(), f"logp_u: {logp_u.min().item()}, {logp_u.mean().item()}, {logp_u.max().item()}"
        # assert (scale > 0.0).all(), f"scale: {scale.min().item()}, {scale.mean().item()}, {scale.max().item()}"
        logp = logp_u - self.logdet_E + 2 * torch.log(scale + 1e-8)
        # assert (logp <= 0.0).all(), f"logp: {logp.min().item()}, {logp.mean().item()}, {logp.max().item()}"
        return logp


class PseudoMetric(torch.nn.Module):
    def __init__(self, vector_dim: int, metric_rank: Optional[int]=None):
        super().__init__()
        self.vector_dim = vector_dim
    
        if metric_rank is not None:
            self.metric_rank = min(vector_dim, metric_rank)
        else:
            self.metric_rank = vector_dim

        self.metric = torch.nn.Linear(self.vector_dim, self.metric_rank, bias=False)

        self.scale_factor = self.metric_rank ** -0.5

        self.A = None
        self.W_norm = None
    
    def reset(self):
        self.A = None
        self.W_norm = None
        
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

    def pseudo_distance(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False, squared: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                while scale.ndim < location1.ndim:
                    scale = scale[..., None]
                    
            location1 = location1 * scale
            location2 = location2 * scale
        
        original_dtype = self.metric.weight.dtype
        dtype = location1.dtype
        self.metric.to(dtype)

        if prepared_k:
            diff = self.metric(location1) - location2
        else:
            diff = self.metric(location1 - location2)

        self.metric.to(original_dtype)

        if squared:
            return diff.pow(2).sum(dim=-1)
        else:
            return torch.norm(diff, dim=-1)
        
    def prepare_k(self, location: torch.Tensor):
        return self.metric(location)
    
    def prepare_q(self, location: torch.Tensor):
        return self.metric(location)

    def cross_affinity(self, vector1: torch.Tensor, vector2: torch.Tensor, projected: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        if not projected:
            if scale is not None:
                if isinstance(scale, torch.Tensor):
                    while scale.ndim < vector1.ndim:
                        scale = scale[..., None]
                    
                    scale1 = scale[:, -vector1.shape[-1]:]
                    scale2 = scale[:, -vector2.shape[-1]:]
                else:
                    scale1 = scale2 = scale

                vector1 = vector1 * scale1
                vector2 = vector2 * scale2

            vector1 = self.metric(vector1)
            vector2 = self.metric(vector2)

        affinity = (vector1[..., None, :] * vector2[..., None, :, :]).sum(dim=-1)

        return affinity
    
    def cross_distance(self, vector1: torch.Tensor, vector2: torch.Tensor, squared: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                while scale.ndim < vector1.ndim:
                    scale = scale[..., None]
                
                scale1 = scale[:, -vector1.shape[1]:]
                scale2 = scale[:, -vector2.shape[1]:]
            else:
                scale1 = scale2 = scale

            vector1 = vector1 * scale1
            vector2 = vector2 * scale2
        
        vector1 = self.metric(vector1.to(self.metric.weight.dtype)).to(vector1.dtype)
        vector2 = self.metric(vector2.to(self.metric.weight.dtype)).to(vector2.dtype)

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

    def make_pseudoinverse(self):
        E = self.metric.weight
        self.A = stable_left_inverse(E.float()).to(E.dtype)  # (E,M)

    def build_distribution_from_center(
        self,
        center: torch.Tensor,        # (B,S,E) or (B,E)
        scale: torch.Tensor,          # broadcastable to center, e.g. (B,S,E) or scalar
        tau: float = 1e-3,
    ):
        # compute right-inverse once per call
        if self.A is None:
            self.make_pseudoinverse()

        E = self.metric.weight

        base = EmbeddedLowRankGaussian(
            loc=center,
            E=E,
            scale=scale,
            E_dagger=self.A,
            tau=tau,
        )
        
        return base
    
    def sample(self, shape: Tuple[int, ...] = torch.Size(), device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        dist = self.build_distribution_from_center(torch.zeros(shape + (self.vector_dim,), device=device, dtype=dtype), 1.0, tau=self.tau)
        return dist.sample(shape)


class IndexedLowRankGaussianMixture(IndexedMixture):
    def __init__(self, logits: torch.Tensor, metric: PseudoMetric, center: torch.Tensor, scale: torch.Tensor, batch_lengths: Optional[torch.Tensor]=None):

            super().__init__(logits, self.distribution_builder, metric, center, scale, batch_lengths=batch_lengths)
        
    def distribution_builder(self, metric: PseudoMetric, center: torch.Tensor, scale: torch.Tensor, 
                             batch_lengths: Optional[torch.Tensor]=None, idx: Optional[torch.Tensor]=None) -> D.Distribution:
        return metric.build_distribution_from_center(center, scale)

