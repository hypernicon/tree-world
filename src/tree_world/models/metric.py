import torch
import torch.distributions as D
import math
from typing import Optional, Union, Callable, Tuple

from ..fourier import make_alphas, make_lattice_basis


LOG2PI = math.log(2.0 * math.pi)


def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


import math
import torch
import torch.distributions as D

LOG2PI = math.log(2.0 * math.pi)

def atanh(x: torch.Tensor) -> torch.Tensor:
    # stable atanh for |x|<1
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


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
        eps_scale: float = 1e-6,
        chol_jitter: float = 1e-6,
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

    def _intrinsic_logdet_metric(self, diag_vec: torch.Tensor) -> torch.Tensor:
        """
        diag_vec: (Bflat, E) nonnegative weights defining G = A^T diag(diag_vec) A (M×M)
        returns: (Bflat,) logdet(G)
        """
        # Compute G in fp32 for stability:
        A = self.A.float()             # (E,M)
        At = A.T.contiguous()          # (M,E)
        diag_f = diag_vec.float()      # (Bflat,E)

        Bflat, E = diag_f.shape
        M = At.shape[0]

        # Build G = At * diag_f @ A  without forming diag:
        # (Bflat,M,E) = (1,M,E) * (Bflat,1,E)
        Atw = At.unsqueeze(0) * diag_f.unsqueeze(1)
        G = Atw @ A                    # (Bflat,M,M)

        # Add a tiny jitter to diagonal to avoid singularities when diag_vec has zeros
        G.diagonal(dim1=-2, dim2=-1).add_(self.chol_jitter)

        L = torch.linalg.cholesky(G)   # (Bflat,M,M) fp32
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (Bflat,)
        return logdet.to(self.loc.dtype)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        value: sample_shape + broadcastable_batch + (E,)
        returns: sample_shape + broadcasted_batch
        """
        dtype = self.loc.dtype
        device = self.loc.device
        M, E = self._M, self._E

        # ---- 1) Broadcast everything to the distribution's batch shape ----
        # target shape is: sample_shape + batch_shape + (E,)
        # We'll infer sample_shape from 'value' by aligning its trailing event dim.
        if value.shape[-1] != E:
            raise ValueError(f"value last dim {value.shape[-1]} != E {E}")

        # Broadcast value to have the distribution batch dims (self.batch_shape).
        # We allow value to omit the S dim (e.g. (B,T,E)) or have singleton S (B,T,1,E).
        # The canonical batch tensor is loc: batch+(E,)
        loc = self.loc
        scale = self.scale

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
        u = delta @ self.W.T                                      # (N,M)

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


class EmbeddedLowRankTanhGaussian(D.Distribution):
    """
    l = tanh(y), where y ~ EmbeddedLowRankGaussian(...)
    Intrinsic density in l-space (again, on the pushed manifold in (-1,1)^E).
    """
    support = D.constraints.interval(-1.0, 1.0)
    has_rsample = True

    def __init__(self, base: EmbeddedLowRankGaussian, clamp_eps: float = 1e-2, validate_args=None):
        self.base = base
        self.clamp_eps = float(clamp_eps)
        super().__init__(batch_shape=base.batch_shape, event_shape=base.event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        y = self.base.rsample(sample_shape)
        l = torch.tanh(y)
        # IMPORTANT for bf16: clamp in fp32 with eps big enough (you found 1e-2 works)
        l = l.float().clamp(-1.0 + self.clamp_eps, 1.0 - self.clamp_eps).to(y.dtype)
        return l

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # clamp away from ±1 in fp32
        l = value.float().clamp(-1.0 + self.clamp_eps, 1.0 - self.clamp_eps)

        # IMPORTANT: broadcast l to base.loc shape (batch+(E,))
        # base.loc is the authoritative batch shape, e.g. (B,T,S,E)
        l = l.expand_as(self.base.loc).to(self.base.loc.dtype)

        # invert: y = atanh(l)
        y = atanh(l)

        # base intrinsic log_prob in y-space (this will now match shapes)
        lp_y = self.base.log_prob(y)

        # compute d = 1 - l^2
        d = (1.0 - l.pow(2)).clamp_min(0.0)   # (B,T,S,E)

        # scale broadcast to same shape
        scale = self.base.scale.expand_as(self.base.loc)
        scale = scale.float().clamp_min(self.base.eps_scale).to(self.base.loc.dtype)

        # diag vectors for intrinsic logdet metric
        diag_y = scale.pow(2)                 # (B,T,S,E)
        diag_l = (scale * d).pow(2)           # (B,T,S,E)

        E = self.base._E
        logdetGy = self.base._intrinsic_logdet_metric(diag_y.reshape(-1, E)).reshape(lp_y.shape)
        logdetGl = self.base._intrinsic_logdet_metric(diag_l.reshape(-1, E)).reshape(lp_y.shape)

        lp_l = lp_y + 0.5 * (logdetGy - logdetGl)
        return lp_l


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

        self.A = None
        
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

    def make_pseudoinverse(self):
        W = self.metric.weight.T
        self.A = torch.linalg.pinv(W.float()).to(W.dtype)  # (E,M)

    def build_distribution_from_center(
        self,
        center: torch.Tensor,        # (B,S,E) or (B,E)
        scale: torch.Tensor,          # broadcastable to center, e.g. (B,S,E) or scalar
        bounded: bool,
        clamp_eps: float = 1e-2,
        eps_scale: float = 1e-6,
        chol_jitter: float = 1e-6,
    ):
        # compute right-inverse once per call
        W = self.metric.weight.T
        if self.A is None:
            self.make_pseudoinverse()

        base = EmbeddedLowRankGaussian(
            loc=center,
            W=W,
            scale=scale,
            A=self.A,
            eps_scale=eps_scale,
            chol_jitter=chol_jitter,
        )

        if bounded:
            return EmbeddedLowRankTanhGaussian(base, clamp_eps=clamp_eps)
        else:
            return base


def sample_indexed_mixture(
    logits: torch.Tensor,  # (B,T,S) or (...,S)
) -> torch.Tensor:
    """
    Samples mixture component indices without constructing a Distribution.
    Returns indices of shape logits.shape[:-1], values in [0,S).
    """
    # Always do categorical sampling in fp32 for stability in bf16
    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
    # multinomial expects 2D; flatten leading dims
    flat = probs.reshape(-1, probs.shape[-1])
    idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return idx.reshape(logits.shape[:-1])


def gather_component(
    x: torch.Tensor,        # (..., S, E) or (..., S)
    idx: torch.Tensor,      # (...,)
    comp_dim: int = -2      # dimension of S in x
) -> torch.Tensor:
    """
    Gathers x at indices idx along comp_dim.
    If x is (..., S, E) and idx is (...), returns (..., E).
    If x is (..., S) and idx is (...), returns (...).
    """
    # Move comp_dim to -1 or -2 handling
    if x.ndim == idx.ndim + 1:
        # x: (..., S)
        gather_idx = idx.unsqueeze(-1)
        out = x.gather(dim=comp_dim, index=gather_idx).squeeze(-1)
        return out
    elif x.ndim == idx.ndim + 2:
        # x: (..., S, E)
        E = x.shape[-1]
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, E)
        out = x.gather(dim=comp_dim, index=gather_idx).squeeze(comp_dim)
        return out
    else:
        raise ValueError(f"Unsupported shapes: x {x.shape}, idx {idx.shape}")


class IndexedMixture:
    """
    Mixture that samples only the chosen component.

    logits: (..., S)
    params: any tensors with an S dimension aligned to logits
    make_comp: function that takes gathered params and returns a torch.distributions.Distribution
    """
    def __init__(self, logits: torch.Tensor, metric: PseudoMetric, center: torch.Tensor, scale: torch.Tensor, bounded: bool,
                 clamp_eps: float = 1e-2, eps_scale: float = 1e-6, chol_jitter: float = 1e-6):
        self.logits = logits
        self.metric = metric
        self.metric.make_pseudoinverse()

        self.params = {"center": center, "scale": scale}
        self.kwargs = {"bounded": bounded,"clamp_eps": clamp_eps, "eps_scale": eps_scale, "chol_jitter": chol_jitter}

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # sample component indices
        idx = sample_indexed_mixture(self.logits)  # (...,)
        # gather params for chosen component
        gathered = {k: gather_component(v, idx, comp_dim=-2) for k, v in self.params.items()}
        # sample from that component
        comp = self.metric.build_distribution_from_center(**gathered, **self.kwargs)
        return comp.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        idx = sample_indexed_mixture(self.logits)
        gathered = {k: gather_component(v, idx, comp_dim=-2) for k, v in self.params.items()}
        comp = self.metric.build_distribution_from_center(**gathered, **self.kwargs)
        return comp.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor, top_k: int=None) -> torch.Tensor:
        """
        Exact mixture log_prob computed as logsumexp over components.

        comp_log_prob(value, **params) must return log_prob_x of shape (..., S)
        """
        # log mix probs in fp32
        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(self.logits.float(), dim=-1, k=top_k)
            log_mix = torch.log_softmax(top_k_logits.float(), dim=-1)  # (..., top_k)
            center = gather_component(self.params["center"], top_k_indices, comp_dim=-2)
            scale = gather_component(self.params["scale"], top_k_indices, comp_dim=-2)
        else:
            log_mix = torch.log_softmax(self.logits.float(), dim=-1)  # (..., S)
            center = self.params["center"]
            scale = self.params["scale"]

        # component log probs: (..., S)
        v = value.unsqueeze(-2).float()
        comp = self.metric.build_distribution_from_center(center=center, scale=scale, **self.kwargs)
        log_px = comp.log_prob(v)              # compute in fp32 internally if needed

        z = log_mix + log_px
        return torch.logsumexp(z, dim=-1).to(log_px.dtype)
