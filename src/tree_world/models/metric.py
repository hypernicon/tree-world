import torch
import torch.distributions as D
import math
from typing import Optional, Union

from ..fourier import make_alphas, make_lattice_basis


def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _build_A_and_cholesky(W, cs2, lam):
    """
    W:   (M,E)
    cs2: (Bflat,E)
    lam: scalar tensor
    returns:
      L: (Bflat,M,M)  Cholesky of A = I + (1/lam) W diag(cs2) W^T
      logdetA: (Bflat,)
    """
    M, E = W.shape
    Bflat = cs2.shape[0]
    dtype = W.dtype
    device = W.device

    # WWt = W diag(cs2[b]) W^T  -> (Bflat,M,M)
    # compute in fp32 for stability, but keep memory bounded (no Ww temp)
    WWt = torch.einsum('me,be,ne->bmn', W.float(), cs2.float(), W.float())  # fp32

    I = torch.eye(M, device=device, dtype=torch.float32).unsqueeze(0).expand(Bflat, M, M)
    A = I + (1.0 / lam.float()) * WWt  # fp32

    L = torch.linalg.cholesky(A)  # fp32
    logdetA = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # fp32

    return L.to(dtype), logdetA.to(dtype)


class LowRankPlusDiagGaussian(D.Distribution):
    """
    x ~ N(loc, s^2 * (lam I + (W diag(col_scale))^T (W diag(col_scale)))^-1)

    W:         (M, E) shared
    col_scale: batch+(E,)
    loc:       batch+(E,)
    """
    support = D.constraints.real
    has_rsample = True
    arg_constraints = {
        'loc': D.constraints.real, 
        'W': D.constraints.real, 
        'col_scale': D.constraints.real, 
    }

    def __init__(self, loc, W, col_scale, s=1.0, lam=1e-3, validate_args=None):
        self.loc = loc
        self.W = W
        self.col_scale = col_scale
        self.s = float(s)
        self.lam = float(lam)

        super().__init__(
            batch_shape=loc.shape[:-1],
            event_shape=loc.shape[-1:],
            validate_args=validate_args,
        )

        M, E = W.shape
        dtype = W.dtype
        assert loc.shape[-1] == E, f"loc last dim {loc.shape[-1]} != W second dim {E}"
        assert col_scale.shape == loc.shape, f"col_scale {col_scale.shape} != loc {loc.shape}"

        self._M = M
        self._E = E

    def rsample(self, sample_shape=torch.Size()):
        device = self.loc.device
        dtype = self.loc.dtype
        M, E = self._M, self._E
        lam, s = self.lam, self.s

        # Flatten batch for cached chol(A)
        loc = self.loc.reshape(-1, E)             # (Bflat, E)
        cs = self.col_scale.reshape(-1, E)        # (Bflat, E)
        lam_t = torch.as_tensor(self.lam, device=self.loc.device, dtype=self.loc.dtype)
        L, _ = _build_A_and_cholesky(self.W, cs.square(), lam_t)
        Bflat = loc.shape[0]

        # z ~ N(0, (s^2/lam) I_E)
        z = (s / math.sqrt(lam)) * torch.randn(sample_shape + (Bflat, E), device=device, dtype=dtype)
        # eps ~ N(0, s^2 I_M)
        eps = s * torch.randn(sample_shape + (Bflat, M), device=device, dtype=dtype)

        # y = W (cs ⊙ z) + eps
        cz = cs.unsqueeze(0) * z                  # sample+(Bflat,E)
        y = cz @ self.W.t() + eps                 # sample+(Bflat,M)

        # alpha = A^{-1} (y/lam)
        alpha = torch.cholesky_solve((y / lam).unsqueeze(-1).float(), L.unsqueeze(0).float()).squeeze(-1).to(dtype)  # sample+(Bflat,M)

        # delta = z - cs ⊙ (W^T alpha)
        Wt_alpha = alpha @ self.W                 # sample+(Bflat,E)
        delta = z - cs.unsqueeze(0) * Wt_alpha    # sample+(Bflat,E)

        out = loc.unsqueeze(0) + delta            # sample+(Bflat,E)
        return out.reshape(sample_shape + self.batch_shape + (E,))

    def _log_prob_core(self, value, loc, W, col_scale, lam_t, s_t):
        """
        value, loc: (Bflat,E)
        W:          (M,E)
        col_scale:  (Bflat,E)
        lam_t, s_t: scalar tensors (on same device)
        returns:    (Bflat,)
        """
        M, E = W.shape
        x = value - loc                       # (Bflat,E)
        cs2 = col_scale.square()              # (Bflat,E)

        L, logdetA = _build_A_and_cholesky(W, cs2, lam_t)

        # quad = (lam||x||^2 + ||W (c⊙x)||^2) / s^2
        cx = col_scale * x                    # (Bflat,E)
        Wx = cx @ W.t()                       # (Bflat,M)
        quad = (lam_t * (x * x).sum(dim=-1) + (Wx * Wx).sum(dim=-1)) / (s_t * s_t)

        # logdet precision = [E log lam + logdetA] - 2E log s
        logdet_prec = (E * torch.log(lam_t) + logdetA) - 2.0 * E * torch.log(s_t)

        lp = 0.5 * (logdet_prec - quad - E * math.log(2.0 * math.pi))
        return lp
    
    def log_prob(self, value):
        E = self._E

        value_f = value.reshape(-1, E)
        loc_f   = self.loc.reshape(-1, E)
        cs_f    = self.col_scale.reshape(-1, E)

        lam_t = torch.as_tensor(self.lam, device=value.device, dtype=value.dtype)
        s_t   = torch.as_tensor(self.s,   device=value.device, dtype=value.dtype)

        lp_f = torch.utils.checkpoint.checkpoint(self._log_prob_core, value_f, loc_f, self.W, cs_f, lam_t, s_t, use_reentrant=False)
        return lp_f.reshape(self.batch_shape)


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

    def psuedo_distance(self, location1: torch.Tensor, location2: torch.Tensor, prepared_k: bool=False, squared: bool=False, scale: Optional[Union[float, torch.Tensor]]=None):
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                while scale.ndim < location1.ndim:
                    scale = scale[..., None]
                    
            location1 = location1 * scale
            location2 = location2 * scale

        if prepared_k:
            diff = self.metric(location1) - location2
        else:
            diff = self.metric(location1 - location2)

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
        
        vector1 = self.metric(vector1)
        vector2 = self.metric(vector2)

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

    def build_distribution_with_center(
        self,
        center: torch.Tensor,           # (B,S,E)
        scale: torch.Tensor,         # (B,S,E)  learnable unconstrained
        lam: float = 1e-2,
        eps: float = 1e-2,
        bounded: bool = False,
    ):
        # learned positive step
        scale = scale.clamp_min(eps)  # (B,S,E)

        W = self.metric.weight  # (M,E)

        if bounded:
            center_c = center.clamp(-1 + eps, 1 - eps)
            y0 = atanh(center_c)  # (B,S,E)

            # Jacobian diag at center: d tanh / dy = 1 - tanh(y)^2 = 1 - center^2
            col_scale = (1.0 - center_c.square()).clamp_min(eps)

            # delta distribution in y-space, zero mean
            base_delta = LowRankPlusDiagGaussian(
                loc=torch.zeros_like(y0),
                W=W,
                col_scale=col_scale,
                s=1.0,
                lam=lam,
            )

            # y = y0 + step ⊙ delta
            y_dist = D.TransformedDistribution(base_delta, D.AffineTransform(loc=y0, scale=scale))

            # l = tanh(y)
            return D.TransformedDistribution(y_dist, D.TanhTransform(cache_size=1))

        else:
            # unbounded: center is already in ℝ^E
            col_scale = torch.ones_like(center)

            base_delta = LowRankPlusDiagGaussian(
                loc=torch.zeros_like(center),
                W=W,
                col_scale=col_scale,
                s=1.0,
                lam=lam,
            )

            # x = center + step ⊙ delta
            return D.TransformedDistribution(base_delta, D.AffineTransform(loc=center, scale=scale))