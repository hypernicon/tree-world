import torch
import torch.distributions as D
import math
from typing import Optional, Union

from .mixture import IndexedMixture


LOG2PI = math.log(2.0 * math.pi)


def stable_right_inverse(W: torch.Tensor, tau: float = 1e-3) -> torch.Tensor:
    """
    W: (M,E)  (M << E)
    returns A: (E,M) ≈ right-inverse
    """
    Wf = W.float()
    M = Wf.shape[0]
    I = torch.eye(M, device=W.device, dtype=torch.float32)
    K = Wf @ Wf.T + tau * I                     # (M,M)
    # Solve K X = W  -> X = K^{-1} W
    X = torch.linalg.solve(K, Wf)               # (M,E)
    A = X.T                                     # (E,M)  = W^T K^{-1}
    return A.to(W.dtype)


class EmbeddedLowRankGaussian(D.Distribution):
    """
    Intrinsic (manifold) distribution:
        u ~ N(0, I_M)
        y = loc + scale ⊙ (A u)
    where:
        W: (M, E), full row-rank (ideally)
        A: (E, M) right-inverse, e.g. pinv(W) so that W @ A ≈ I_M

    log_prob(y) is the *intrinsic* log density induced by u (not a full R^E Lebesgue density).
    """
    support = D.constraints.real
    has_rsample = True

    def __init__(
        self,
        loc: torch.Tensor,          # batch+(E,)
        W: torch.Tensor,            # (M,E)
        scale: torch.Tensor,        # batch+(E,)  (can be scalar-broadcasted)
        A: Optional[torch.Tensor] = None,     # (E,M) optional; if None compute pinv(W)
        validate_args=None,
        eps_scale: float = 1e-3,
        chol_jitter: float = 1e-3,
    ):
        self.loc = loc
        self.W = W
        self.scale = scale
        self.eps_scale = float(eps_scale)
        self.chol_jitter = float(chol_jitter)

        M, E = W.shape
        assert loc.shape[-1] == E, f"loc last dim {loc.shape[-1]} != W second dim {E}"
        assert scale.shape == loc.shape or scale.shape == (1,) or scale.shape == (), \
            f"scale shape {scale.shape} must broadcast to loc {loc.shape}"

        if A is None:
            # NOTE: pinv is done in fp32 for stability
            A = torch.linalg.pinv(W.float()).to(W.dtype)  # (E,M)

        assert A.shape == (E, M)

        self.A = A
        self._M = M
        self._E = E

        super().__init__(batch_shape=loc.shape[:-1], event_shape=(E,), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        dtype = self.loc.dtype
        device = self.loc.device
        M, E = self._M, self._E

        # sample u ~ N(0, I_M)
        u = torch.randn(sample_shape + self.batch_shape + (M,), device=device, dtype=dtype)

        # y = loc + scale ⊙ (A u)
        # (..,M) @ (M,E) -> (..,E)
        Au = u @ self.A.T
        y = self.loc + self.scale * Au
        return y

    def _flatten_batch(self, x: torch.Tensor):
        # x: sample_shape + batch_shape + (E,)
        E = x.shape[-1]
        batch_ndim = len(self.batch_shape)
        sample_shape = x.shape[:-(batch_ndim + 1)]
        x_flat = x.reshape(sample_shape + (-1, E))  # sample + (Bflat, E)
        return x_flat, sample_shape

    def _intrinsic_logdet_metric(self, diag_vec, chunk=256):
        """
        diag_vec: (N, E) float/half
        returns: (N,) logdet(A^T diag(diag_vec) A) with jitter
        """
        A = self.A.float()          # (E, M)
        E, M = A.shape
        diag_vec = diag_vec.float().clamp_min(self.eps_scale)
        out = []

        for i in range(0, diag_vec.shape[0], chunk):
            d = diag_vec[i:i+chunk]   # (c, E)

            # Form G = A^T diag(d) A without (c,M,E) tensor:
            # Compute DA = d[..., :, None] * A[None, :, :]  -> (c, E, M)
            DA = d.unsqueeze(-1) * A.unsqueeze(0)                # (c, E, M)  <-- this is the big temp, but only 'chunk'
            G = torch.matmul(DA.transpose(1, 2), A.unsqueeze(0))  # (c, M, M)

            # jitter + cholesky
            G.diagonal(dim1=-2, dim2=-1).add_(self.chol_jitter)
            L = torch.linalg.cholesky(G)
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (c,)
            out.append(logdet)

            # free ASAP
            del DA, G, L

        return torch.cat(out, dim=0).to(self.loc.dtype)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        value: sample_shape + broadcastable_batch + (E,)
        returns: sample_shape + broadcasted_batch
        """
        dtype = value.dtype
        device = value.device
        M, E = self._M, self._E

        # ---- 1) Broadcast everything to the distribution's batch shape ----
        # target shape is: sample_shape + batch_shape + (E,)
        # We'll infer sample_shape from 'value' by aligning its trailing event dim.
        if value.shape[-1] != E:
            raise ValueError(f"value last dim {value.shape[-1]} != E {E}")

        # Broadcast value to have the distribution batch dims (self.batch_shape).
        # We allow value to omit the S dim (e.g. (B,T,E)) or have singleton S (B,T,1,E).
        # The canonical batch tensor is loc: batch+(E,)
        loc = self.loc.to(dtype).to(device)
        scale = self.scale.to(dtype).to(device)
        W = self.W.to(dtype).to(device)

        # Bring value up to loc's batch rank (possibly by inserting singleton dims before event)
        # We want value.ndim == loc.ndim or loc.ndim+? with sample dims in front.
        batch_ndim = len(self.batch_shape)
        # value has: sample_ndim + batch' + 1(event)
        # loc has:   batch_ndim + 1(event)
        sample_ndim = value.ndim - (batch_ndim + 1)
        if sample_ndim < 0:
            raise ValueError(f"value has too few dims: {value.shape} for batch_shape {self.batch_shape}")

        # Now expand value/loc/scale to a common shape
        target_shape = value.shape[:sample_ndim] + loc.shape  # sample + batch + (E,)
        v = value.expand(target_shape)
        loc_e = loc.expand(target_shape)
        scale_e = scale.expand(target_shape)

        # ---- 2) Flatten sample+batch to (N, E) consistently ----
        v_flat = v.reshape(-1, E)
        loc_flat = loc_e.reshape(-1, E)
        scale_flat = scale_e.reshape(-1, E)

        # ---- 3) Compute intrinsic pullback u = W * ((y-loc)/scale) ----
        scale_flat = scale_flat.float().clamp_min(self.eps_scale).to(dtype)
        delta = (v_flat - loc_flat) / scale_flat                  # (N,E)
        u = delta @ W.T                                      # (N,M)

        # log N(u;0,I)
        logp_u = -0.5 * (u.float().pow(2).sum(dim=-1) + M * math.log(2.0 * math.pi))  # (N,)

        # ---- 4) Jacobian constant term for unbounded case (if you cache it) ----
        # If you are still computing logdet per-call, do it on (N,E) diag weights.
        # For unbounded, diag is scale^2 and does NOT depend on sample, so you should cache per batch element.
        # For now, keep it general:
        diag_vec = scale_flat.pow(2)  # (N,E)
        logdetG = self._intrinsic_logdet_metric(diag_vec)  # (N,)

        lp = logp_u - 0.5 * logdetG                        # (N,)
        return lp.reshape(target_shape[:-1]).to(dtype)


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
        W = self.metric.weight
        self.W_norm = W / W.norm(dim=0, keepdim=True)
        # self.A = torch.linalg.pinv(W.float()).to(W.dtype)  # (E,M)
        self.A = stable_right_inverse(W.float()).to(W.dtype)  # (E,M)

    def build_distribution_from_center(
        self,
        center: torch.Tensor,        # (B,S,E) or (B,E)
        scale: torch.Tensor,          # broadcastable to center, e.g. (B,S,E) or scalar
        eps_scale: float = 1e-6,
        chol_jitter: float = 1e-6,
    ):
        # compute right-inverse once per call
        if self.A is None:
            self.make_pseudoinverse()

        W = self.W_norm

        base = EmbeddedLowRankGaussian(
            loc=center,
            W=W,
            scale=scale,
            A=self.A,
            eps_scale=eps_scale,
            chol_jitter=chol_jitter,
        )
        
        return base


class IndexedLowRankGaussianMixture(IndexedMixture):
    def __init__(self, logits: torch.Tensor, metric: PseudoMetric, center: torch.Tensor, scale: torch.Tensor, batch_lengths: Optional[torch.Tensor]=None):

        super().__init__(logits, self.distribution_builder, metric, center, scale, batch_lengths)
    
    def distribution_builder(self, metric: PseudoMetric, center: torch.Tensor, scale: torch.Tensor, batch_lengths: Optional[torch.Tensor]=None, __idx: Optional[torch.Tensor]=None) -> D.Distribution:
        return metric.build_distribution_from_center(center, scale)

