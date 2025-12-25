import torch
import torch.distributions as D
from typing import Callable, Optional


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
) -> torch.Tensor:
    """
    Gathers x at indices idx along comp_dim.
    If x is (..., S, E) and idx is (...), returns (..., E).
    If x is (..., S) and idx is (...), returns (...).
    """
    if idx is None:
        return x

    if x.ndim == idx.ndim:
        B, Tx, S = x.shape
        B, T, K = idx.shape
        if Tx != T:
            x = x.expand(B, T, S)
        out = x.gather(dim=-1, index=idx).squeeze(-1)
        return out
    elif x.ndim == idx.ndim + 1:
        if x.ndim == 3:
            B, Tx, S = x.shape
            B, T = idx.shape
            if Tx != T:
                x = x.expand(B, T, S)
            gather_idx = idx.unsqueeze(-1).expand(B, T, 1)
            out = x.gather(dim=-1, index=gather_idx).squeeze(-1)
            return out
        else:  # x.ndim == 4
            B, Tx, S, E = x.shape
            B, T, K = idx.shape
            if Tx != T:
                x = x.expand(B, T, S, E)
            gather_idx = idx.unsqueeze(-1).expand(B, T, K, E)
            out = x.gather(dim=-2, index=gather_idx).squeeze(-1)
            return out
    elif x.ndim == idx.ndim + 2:
        # x: (..., S, E)
        B, Tx, S, E = x.shape
        B, T = idx.shape
        if Tx != T:
            x = x.expand(B, T, S, E)
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, E)
        out = x.gather(dim=-2, index=gather_idx).squeeze(-2)
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
    def __init__(self, logits: torch.Tensor, distribution_builder: Callable, *params, **param_kwargs):
        self.logits = logits
        self.distribution_builder = distribution_builder
        self.params = params
        self.param_kwargs = param_kwargs
    
    def _build_distribution(self, idx: Optional[torch.Tensor]=None) -> D.Distribution:
        if idx is None:
            params = self.params
            param_kwargs = self.param_kwargs
        else:
            params = [gather_component(p, idx) if isinstance(p, torch.Tensor) else p for p in self.params]
            param_kwargs = {k: gather_component(v, idx) if isinstance(v, torch.Tensor) else v for k, v in self.param_kwargs.items()}

        return self.distribution_builder(*params, **param_kwargs)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # sample component indices
        idx = sample_indexed_mixture(self.logits)  # (...,)
        comp = self._build_distribution(idx)
        return comp.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        idx = sample_indexed_mixture(self.logits)
        comp = self._build_distribution(idx)
        return comp.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor, aux: Optional[torch.Tensor]=None, top_k: int=None) -> torch.Tensor:
        """
        Exact mixture log_prob computed as logsumexp over components.

        comp_log_prob(value, **params) must return log_prob_x of shape (..., S)
        """
        # log mix probs in fp32
        idx = None
        if top_k is not None and top_k < self.logits.shape[-1]:
            top_k_logits, idx = torch.topk(self.logits.float(), dim=-1, k=top_k)
            log_mix = torch.log_softmax(top_k_logits.float(), dim=-1)  # (..., top_k)
        else:
            log_mix = torch.log_softmax(self.logits.float(), dim=-1)  # (..., S)

        # component log probs: (..., S)
        v = value.unsqueeze(-2).float()
        comp = self._build_distribution(idx)
        if aux is not None:
            B, T = v.shape[:-2]
            target_shape = (B, T) + aux.shape[1:]
            aux = gather_component(aux[:, None, ...].expand(target_shape), idx)
            log_px = comp.log_prob(v, aux)
        else:
            log_px = comp.log_prob(v)

        z = log_mix + log_px
        return torch.logsumexp(z, dim=-1).to(log_px.dtype)


class IndexedGaussianMixture(IndexedMixture):
    def __init__(self, logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):

        super().__init__(logits, self.distribution_builder, loc, scale)
    
    def distribution_builder(self, loc: torch.Tensor, scale: torch.Tensor) -> D.Distribution:
        return D.Independent(D.Normal(loc, scale), 1)

