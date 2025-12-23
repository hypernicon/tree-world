import math
import torch
import torch.distributions as D

from typing import Optional, Union
from ..fourier import make_alphas, make_lattice_basis, solve_for_deltas
from .metric import PseudoMetric, sample_indexed_mixture, gather_component


def check_nan_inf(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"{name} BAD: nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()}")
        print(f"{name} stats: min={t.nan_to_num().min().item()} max={t.nan_to_num().max().item()}")
        raise ValueError(f"{name} is nan/inf")


def loss_for_deltas(delta_thetas: torch.Tensor, K_dagger: torch.Tensor, lattice_basis: torch.Tensor, alphas: torch.Tensor):
    deltas = solve_for_deltas(delta_thetas, K_dagger, lattice_basis, alphas)

    # deltas has shape (batch_size, time_steps, J, d)
    return deltas.var(dim=-2).mean()


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, num_heads: int=1):
    # Convert our inputs, which are (batch_size, time_steps, embed_dim) to (batch_size, num_heads, time_steps, head_dim)
    B, T, _ = query.shape
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).transpose(1, 2)
    result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    result = result.transpose(1, 2).reshape(B, T, -1)
    return result


class IndexedGaussianMixture:
    """
    Mixture that samples only the chosen component.

    logits: (..., S)
    params: any tensors with an S dimension aligned to logits
    make_comp: function that takes gathered params and returns a torch.distributions.Distribution
    """
    def __init__(self, logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, bounded: bool, bounded_eps: float = 1e-2):
        self.logits = logits
        self.bounded = bounded
        self.bounded_eps = bounded_eps

        self.params = {"loc": loc, "scale": scale}
    
    def _build_distribution(self, loc: torch.Tensor, scale: torch.Tensor) -> D.Distribution:
        if self.bounded:
            loc = loc.clamp(-1.0 + self.bounded_eps, 1.0 - self.bounded_eps)
        comp = D.Independent(D.Normal(loc, scale), 1)
        if self.bounded:
            comp = D.TransformedDistribution(comp, D.TanhTransform(cache_size=1))
        return comp

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # sample component indices
        idx = sample_indexed_mixture(self.logits)  # (...,)
        # gather params for chosen component
        gathered = {k: gather_component(v, idx, comp_dim=-2) for k, v in self.params.items()}
        if self.bounded:
            gathered["loc"] = gathered["loc"].clamp(-1.0 + self.bounded_eps, 1.0 - self.bounded_eps)

        comp = self._build_distribution(gathered["loc"], gathered["scale"])
        return comp.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        idx = sample_indexed_mixture(self.logits)
        gathered = {k: gather_component(v, idx, comp_dim=-2) for k, v in self.params.items()}
        comp = self._build_distribution(gathered["loc"], gathered["scale"])
        return comp.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor, top_k: int=None) -> torch.Tensor:
        """
        Exact mixture log_prob computed as logsumexp over components.

        comp_log_prob(value, **params) must return log_prob_x of shape (..., S)
        """
        # log mix probs in fp32
        if top_k is not None and top_k < self.logits.shape[-1]:
            top_k_logits, top_k_indices = torch.topk(self.logits.float(), dim=-1, k=top_k)
            log_mix = torch.log_softmax(top_k_logits.float(), dim=-1)  # (..., top_k)
            loc = gather_component(self.params["loc"], top_k_indices, comp_dim=-2)
            scale = gather_component(self.params["scale"], top_k_indices, comp_dim=-2)
        else:
            log_mix = torch.log_softmax(self.logits.float(), dim=-1)  # (..., S)
            loc = self.params["loc"]
            scale = self.params["scale"]

        # component log probs: (..., S)
        v = value.unsqueeze(-2).float()
        comp = self._build_distribution(loc, scale)
        log_px = comp.log_prob(v)              # compute in fp32 internally if needed

        z = log_mix + log_px
        return torch.logsumexp(z, dim=-1).to(log_px.dtype)


class ErrorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(self.mlp(x))


class TemTransformerFeedForward(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.fc1_dropout = torch.nn.Dropout(dropout)
        
        self.fc_silu = torch.nn.Linear(embed_dim, hidden_dim)

        self.fc2 = torch.nn.Linear(hidden_dim, embed_dim)
        self.fc2_dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        y = self.fc1(x)
        z = self.fc_silu(x)
        x = torch.nn.functional.silu(y) * z
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        x = self.fc2_dropout(x)
        return x


class TemTransformerLayer(torch.nn.Module):
    def __init__(self, key_dim: int, value_dim: int, embed_dim: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.q_proj = torch.nn.Linear(key_dim, embed_dim, bias=False)
        self.k_proj = torch.nn.Linear(key_dim, embed_dim, bias=False)
        self.v_proj = torch.nn.Linear(value_dim, embed_dim, bias=False)
        self.v_out = torch.nn.Linear(embed_dim, value_dim, bias=False)

        self.attention_norm = torch.nn.LayerNorm(embed_dim)

        self.feed_forward = TemTransformerFeedForward(value_dim, 4*value_dim, dropout)
        self.feed_forward_norm = torch.nn.LayerNorm(value_dim)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        key_prefix: Optional[torch.Tensor]=None,
        value_prefix: Optional[torch.Tensor]=None,
        add_residual: bool=True, 
        allow_self_attention: bool=True,
        mask: Optional[torch.Tensor]=None,
        causal: bool=True
    ):
        if key_prefix is not None:
            key = torch.cat([key_prefix, key], dim=1)

        orig_value = value
        if value_prefix is not None:
            value = torch.cat([value_prefix, value], dim=1)

        query = self.q_proj(query)
        key = self.k_proj(key)
        value_p = self.v_proj(value)

        value_p = self.attention_norm(value_p)

        if mask is None:
            if causal:
                if allow_self_attention:
                    diagonal = 1
                else:
                    diagonal = 0

                base = torch.full((query.shape[1], key.shape[1]), float('-inf'), dtype=query.dtype, device=query.device)
                mask = torch.triu(base, diagonal=diagonal)

                if not allow_self_attention:
                    # CRITICAL & NONOBVIOUS: prevent NaN in the gradient of the first query; we'll patch the output later
                    # problem is that the first time step has nothing to pay attention to in causal mode, so we get NaN
                    mask[0, :] = 0.0

            elif not allow_self_attention:
                I = torch.eye(max(query.shape[1], key.shape[1]), dtype=torch.bool, device=query.device)[:query.shape[1], :key.shape[1]]
                mask = torch.zeros((query.shape[1], key.shape[1]), dtype=query.dtype, device=query.device)
                mask = mask.masked_fill(I, float('-inf'))

        attn_output = scaled_dot_product_attention(query, key, value_p, attn_mask=mask, num_heads=self.num_heads)

        if causal and not allow_self_attention: # non-causal no self-attention doesn't have the NaN problem
            attn_output[:, 0, :] = torch.zeros_like(attn_output[:, 0, :])
            if attn_output.shape[1] > 1:
                attn_output[:, 1, :] = value_p[:, 0, :]

        attn_output = self.v_out(attn_output)

        if add_residual:
            x = orig_value + self.feed_forward_norm(attn_output)
        else:
            x = self.feed_forward_norm(attn_output)

        y = x + self.feed_forward(x)

        return y


class MetricSampler(torch.nn.Module):
    def __init__(self, qk_metric: PseudoMetric, v_metric: PseudoMetric, v_error_mlp: ErrorMLP, qk_dim: int, bounded: bool=False):
        super().__init__()
        self.qk_metric = qk_metric
        self.v_metric = v_metric
        self.v_error_mlp = v_error_mlp
        self.qk_dim = qk_dim

        self.scale_factor = qk_dim ** -0.5
        self.bounded = bounded
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, value_std: torch.Tensor,
                close_to: Optional[torch.Tensor]=None, close_to_factor: float=1.0):
        B, T, _ = query.shape
        B, S, _ = key.shape
        B, S, E = value.shape

        # op_norm = self.qk_metric.metric_operator_norm()
        # scale = self.scale_factor ** 0.5 / (op_norm ** 0.5 + 1e-8)
        scale = self.scale_factor ** 0.5

        qk_distances = self.qk_metric.cross_distance(query, key, squared=True, scale=scale)
        check_nan_inf("qk_distances", qk_distances)

        mask = torch.tril(torch.ones((max(S, T), max(S, T)), dtype=torch.bool, device=query.device), diagonal=-1)
        qk_distances = qk_distances.masked_fill(mask[None, :, :], float('inf'))
        assert not torch.isnan(qk_distances).any()

        if close_to is not None:
            # close_to has shape (B, S, D) --> distances (B, S)
            v_scale = (E ** -0.25) / (self.v_error_mlp(close_to) + 1e-8)
            close_to_distances = self.v_metric.psuedo_distance(value, close_to, squared=True, scale=v_scale).clamp(max=1e4)
            assert not torch.isnan(close_to_distances).any()
            qk_distances = qk_distances + close_to_factor * close_to_distances[:, None, :]

        invalid_mask = (qk_distances >= float('inf')).all(dim=-1, keepdim=True)  # (B, T, 1)
        qk_distances = qk_distances.masked_fill(invalid_mask, 1.0)

        dist = IndexedGaussianMixture(
            logits=-0.5 * qk_distances, 
            loc=value[:, None, :, :].expand(-1, T, -1, -1), 
            scale=value_std[:, None, :, :].expand(-1, T, -1, -1), 
            bounded=self.bounded,
            bounded_eps=1e-2
        )
        return dist, invalid_mask.squeeze(-1)


class GeometricActionDecoder(torch.nn.Module):
    def __init__(self, location_dim: int, action_dim: int, hidden_dim: int, dropout: float=0.25, 
                 physical_dim: int=2, physical_scale: float=10.0, physical_ratio: float=math.sqrt(2.0)):
        super().__init__()
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        assert location_dim % 2 == 0

        self.action_mlp = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, location_dim // 2),
        )

        self.physical_dim = physical_dim
        self.physical_scale = physical_scale
        self.physical_ratio = physical_ratio
        self.alphas = torch.nn.Buffer(make_alphas(location_dim, physical_dim, physical_scale, physical_ratio))
        lattice_basis, K, K_dagger = make_lattice_basis(self.alphas, physical_dim)
        self.lattice_basis = torch.nn.Buffer(lattice_basis)
        self.K_dagger = torch.nn.Buffer(K_dagger)
        self.K = torch.nn.Buffer(K)

        self.error_mlp = ErrorMLP(location_dim, location_dim)

        assert self.location_dim % (2 * self.physical_dim) == 0

    def forward(self, location: torch.Tensor, action: torch.Tensor, eps: float=1e-6, 
                allow_extension: bool=True, regularize: bool=True):
        B, T, D = location.shape
        assert D == self.location_dim
        assert (B, T, self.action_dim) == action.shape

        # use a block diagonal matrix to rotate the location
        thetas = self.action_mlp(action)

        # if we wanted to be explicit (since our actions are actually displacements in physical space), we could do this:
        # thetas = self.alphas[None, None, :, None] * ((self.K[None, None, ...] @ action[..., None]).view(B, T, -1, self.physical_dim + 1)) 
        # thetas = thetas.view(B, T, -1)
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        location_blocks = location.reshape(B, T, -1, 2)
        location_cos_thetas = location_blocks * cos_thetas[..., None]
        location_sin_thetas = location_blocks * sin_thetas[..., None]

        next_location = torch.stack([
            location_cos_thetas[..., 0] - location_sin_thetas[..., 1],
            location_sin_thetas[..., 0] + location_cos_thetas[..., 1],
        ], dim=-1).reshape(B, T, D)

        # shift the location one step forward to align with the past; output is one step longer than the input
        next_location = torch.cat([location[:, :1], next_location], dim=1)

        if regularize:
            displacement_loss = loss_for_deltas(thetas, self.K_dagger, self.lattice_basis, self.alphas)

        if not allow_extension:
            next_location = next_location[:, :-1]
        if regularize:
            return next_location, displacement_loss
        else:
            return next_location
    
    def logprobs(self, location: torch.Tensor, mean_location: torch.Tensor):
        std_location = self.error_mlp(mean_location)
        return (
            - 0.5 * math.log(2 * math.pi) 
            - torch.log(std_location + 1e-8).sum(dim=-1) 
            - 0.5 * ((location - mean_location) / std_location).pow(2).sum(dim=-1)
        )


class TemLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_heads: int=4, 
                       action_hidden_dim: int=128, dropout: float=0.1, compute_window=1024, physical_dim: int=2, 
                       physical_scale: float=10.0, physical_ratio: float=math.sqrt(2.0)):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.location_metric = PseudoMetric(location_dim, dim=physical_dim, scale=physical_scale, ratio=physical_ratio, metric_rank=embed_dim)
        self.sensory_metric = PseudoMetric(sensory_dim, dim=physical_dim, scale=physical_scale, ratio=physical_ratio, metric_rank=embed_dim)
        self.sensory_metric_with_location = PseudoMetric(sensory_dim, dim=physical_dim, scale=physical_scale, ratio=physical_ratio, metric_rank=embed_dim)

        self.location_error_mlp = ErrorMLP(location_dim, location_dim)
        self.sensory_error_mlp = ErrorMLP(location_dim, sensory_dim)

        self.location_refiner = MetricSampler(
            self.sensory_metric_with_location, self.location_metric, self.location_error_mlp, self.sensory_dim, bounded=True
        )
        self.sensory_predictor = MetricSampler(
            self.location_metric, self.sensory_metric, self.sensory_error_mlp, self.location_dim, bounded=False
        )

        self.geometric_action_decoder = GeometricActionDecoder(
            location_dim, action_dim, action_hidden_dim, dropout, physical_dim, physical_scale, physical_ratio
        )

        self.position_encoder = torch.nn.Linear(location_dim, sensory_dim, bias=False)

    def remove_prefix(self, tensor: torch.Tensor, prefix_length: torch.Tensor, batch_lengths: torch.Tensor):
        """
        Extract a new tensor that starts at the given prefix length and runs to the end of the tensor
        """
        B = tensor.shape[0]
        actual_lengths = batch_lengths - prefix_length
        S = actual_lengths.max()
        indices = torch.arange(S, device=tensor.device)[None, :] + prefix_length[:, None]
        while indices.ndim < tensor.ndim:
            indices = indices.unsqueeze(-1)
        indices = indices.expand(B, S, *tensor.shape[2:])
        return tensor.gather(dim=1, index=indices), indices
    
    def restore_prefix(self, original: torch.Tensor, extracted: torch.Tensor, indices: torch.Tensor):
        """
        Restore the prefix to the original tensor
        """
        return original.scatter(dim=1, index=indices, src=extracted)

    def forward(self, sensory: torch.Tensor, prior_location: Optional[torch.Tensor]=None, action: Optional[torch.Tensor]=None, 
                max_steps: int=2, threshold: float=0.05, refine_alpha: float=0.1, eps: float=1e-6, prefix_length: Union[int, torch.Tensor]=0,
                batch_lengths: Optional[torch.Tensor]=None):
        B, T, S = sensory.shape
        assert max_steps > 0
        if isinstance(prefix_length, int):
            prefix_length = torch.tensor([prefix_length]*B, dtype=torch.long, device=sensory.device)

        if batch_lengths is not None:
            assert batch_lengths.shape == (B,)
            assert (prefix_length < batch_lengths).all()
        else:
            batch_lengths = T * torch.ones((B,), dtype=torch.long, device=sensory.device)

        if prior_location is None:
            initial_location = torch.empty((B, 1, self.location_dim), dtype=sensory.dtype, device=sensory.device).uniform_(-1, 1)
            return initial_location, initial_location, torch.zeros_like(sensory), 0.0, 0.0, 0.0, 0.0
        
        # These next two ifs let us just supply the previous output location and action sequence, and extend them to the new length
        if prior_location.shape[1] < T:
            prior_location = torch.cat([
                prior_location, 
                torch.zeros((B, T - prior_location.shape[1], self.location_dim), 
                dtype=prior_location.dtype, 
                device=prior_location.device
            )], dim=1)
        
        if action.shape[1] < T - prefix_length.min():
            action = torch.cat([
                action, 
                torch.zeros((B, T - prefix_length.min() - action.shape[1], self.action_dim), 
                dtype=action.dtype, 
                device=action.device
            )], dim=1)

        sensory_location = prior_location
        check_nan_inf("prior_location", prior_location)

        prior_location_minus_prefix, prior_location_indices = self.remove_prefix(prior_location, prefix_length, batch_lengths)
        geometric_location_minus_prefix, displacement_loss = self.geometric_action_decoder(
            prior_location_minus_prefix, action, allow_extension=False, regularize=True
        ) # <-- we've already extended the action sequence
        geometric_location = self.restore_prefix(prior_location, geometric_location_minus_prefix, prior_location_indices)
        sensory_plus_geometric = self.make_sensory_keys(geometric_location.detach(), sensory) # <-- stop_gradient 
        check_nan_inf("sensory_plus_geometric", sensory_plus_geometric)
        
        # if sensory_key_prefix is not None and location_prefix is not None:
        #     sensory_plus_geometric_with_prefix = torch.cat([sensory_key_prefix, sensory_plus_geometric], dim=1)
        #     sensory_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
        # else:
        #     sensory_plus_geometric_with_prefix = sensory_plus_geometric
        #    sensory_location_with_prefix = sensory_location

        for k in range(max_steps):
            location_std = self.location_error_mlp(sensory_location)
            location_distribution, location_invalid_mask = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location, location_std,
                close_to=geometric_location.detach(), close_to_factor=1.0
            )

            with torch.no_grad():
                sensory_location = location_distribution.sample().clamp(min=-1+1e-4, max=1-1e-4)

            location_disagreement = self.location_metric.psuedo_distance(geometric_location, sensory_location)

            if (location_disagreement < threshold).all():
                break

        # VAE requires that we sample the encoder, not the decoder, so we use the sensory location as the next location
        next_location = location_distribution.sample().clamp(min=-1+1e-2, max=1-1e-2)
        if next_location.shape[0] > 1:
            print(f"next_location: min: {next_location.min().item()}, mean: {next_location.mean().item()}, max: {next_location.max().item()}")
        check_nan_inf("next_location", next_location) 

        next_location_minus_prefix, _ = self.remove_prefix(next_location, prefix_length, batch_lengths)
        geometric_logprobs = self.geometric_action_decoder.logprobs(
            next_location_minus_prefix, geometric_location_minus_prefix
        )
        sensory_location_logprobs = location_distribution.log_prob(next_location, top_k=32)
        sensory_location_logprobs_minus_prefix, _ = self.remove_prefix(
            sensory_location_logprobs, prefix_length, batch_lengths
        )

        kl_divergence = sensory_location_logprobs_minus_prefix - geometric_logprobs
        location_invalid_mask_minus_prefix, location_invalid_mask_indices = self.remove_prefix(
            location_invalid_mask, prefix_length, batch_lengths
        )
        mask = torch.isnan(kl_divergence) | torch.isinf(kl_divergence) | location_invalid_mask_minus_prefix
        mask = mask | (location_invalid_mask_indices >= batch_lengths[:, None])
        kl_divergence = kl_divergence.masked_fill(mask, 0.0)
        kl_divergence = kl_divergence.sum(dim=-1) / ((~mask).to(kl_divergence.dtype).sum(dim=-1) + 1e-6)
        check_nan_inf("kl_divergence", kl_divergence)

        sensory_std = self.sensory_error_mlp(next_location)
        sensory_distribution, sensory_invalid_mask = self.sensory_predictor(
            next_location, next_location, sensory, sensory_std
        )

        with torch.no_grad():
            sensory_predicted = sensory_distribution.sample()
        
        if sensory.shape[0] > 1:
            print(f"sensory_predicted: min: {sensory_predicted.min().item()}, mean: {sensory_predicted.mean().item()}, max: {sensory_predicted.max().item()}")

        sensory_logprobs = sensory_distribution.log_prob(sensory_predicted, top_k=32)
        mask = torch.isnan(sensory_logprobs) | torch.isinf(sensory_logprobs) | sensory_invalid_mask
        mask = mask | (torch.arange(T, device=sensory.device)[None, :] >= batch_lengths[:, None])
        sensory_logprobs = sensory_logprobs.masked_fill(mask, 0.0)
        sensory_logprobs = sensory_logprobs.sum(dim=-1) / ((~mask).to(sensory_logprobs.dtype).sum(dim=-1) + 1e-6)
        sensory_logprobs = sensory_logprobs.mean()
        check_nan_inf("sensory_logprobs", sensory_logprobs)

        sensory_error = (sensory - sensory_predicted).pow(2).sum(dim=-1)
        sensory_error = sensory_error.masked_fill(sensory_invalid_mask, 0.0)
        sensory_error = sensory_error.sum() / ((~sensory_invalid_mask).to(sensory_error.dtype).sum() + 1e-6)

        if not (kl_divergence >= 0.0).all():
            print(f"kl_divergence is negative: {kl_divergence[kl_divergence < 0.0].shape}")
            print(f"kl_divergence: {kl_divergence.detach().cpu().float().numpy().tolist()}")
        elbo = sensory_logprobs - kl_divergence.mean()

        return (
            next_location.detach(), 
            geometric_location.detach(), 
            sensory_predicted.detach(), 
            elbo, 
            sensory_error.mean().detach(), 
            location_disagreement.mean().detach(), 
            displacement_loss
        )
    
    def make_sensory_keys(self, location: torch.Tensor, sensory: torch.Tensor):
        return sensory + self.position_encoder(location)

    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        return cls(
            location_dim=config.location_dim,
            sensory_dim=config.sensory_embedding_dim,
            action_dim=config.dim,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            action_hidden_dim=config.action_hidden_dim,
            dropout=config.dropout,
            physical_dim=config.dim,
            physical_scale=config.grid_extent,
        )
