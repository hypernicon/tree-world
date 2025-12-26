import math
import torch
import torch.distributions as D

from typing import Optional, Union
from ..fourier import make_alphas, make_lattice_basis, solve_for_deltas
from .metric import PseudoMetric, IndexedLowRankGaussianMixture
from .fourier_metric import IndexedFourierMixture, FourierCodeDistribution, FourierMetric, check_valid_location
from .mixture import IndexedGaussianMixture


def check_nan_inf(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"{name} BAD: nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()}")
        print(f"{name} stats: min={t.nan_to_num().min().item()} max={t.nan_to_num().max().item()}")
        raise ValueError(f"{name} is nan/inf")


def loss_for_deltas(delta_thetas: torch.Tensor, K_dagger: torch.Tensor, lattice_basis: torch.Tensor, alphas: torch.Tensor):
    deltas = solve_for_deltas(delta_thetas, K_dagger, lattice_basis, alphas)  # (B, T, J, d)
    mean_deltas = deltas.mean(dim=-2)  # (B, T, d)
    J = deltas.shape[-2]

    dev_deltas = deltas - mean_deltas[..., None, :]  # (B, T, J, d)
    variances = dev_deltas.square().sum(dim=-1).mean(dim=-1) * (J / (J - 1))  # (B, T)

    return variances.mean()


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, num_heads: int=1):
    # Convert our inputs, which are (batch_size, time_steps, embed_dim) to (batch_size, num_heads, time_steps, head_dim)
    B, T, _ = query.shape
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).transpose(1, 2)
    result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    result = result.transpose(1, 2).reshape(B, T, -1)
    return result


class ErrorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=128, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.scale = scale
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.scale * torch.nn.functional.softplus(self.mlp(x))


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
    def __init__(self, qk_metric: PseudoMetric, v_metric: PseudoMetric, qk_dim: int, location: bool=False):
        super().__init__()
        self.qk_metric = qk_metric
        self.v_metric = v_metric
        self.qk_dim = qk_dim

        self.scale_factor = qk_dim ** -0.5
        self.location = location
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, value_std: torch.Tensor,
                close_to: Optional[torch.Tensor]=None, close_to_factor: float=1.0, batch_lengths: Optional[torch.Tensor]=None):
        B, T, _ = query.shape
        B, S, _ = key.shape
        B, S, E = value.shape

        # op_norm = self.qk_metric.metric_operator_norm()
        # scale = self.scale_factor ** 0.5 / (op_norm ** 0.5 + 1e-8)

        qk_distances = self.scale_factor * self.qk_metric.cross_distance(query.float(), key.float(), squared=True)
        check_nan_inf("qk_distances", qk_distances)

        mask = torch.tril(torch.ones((max(S, T), max(S, T)), dtype=torch.bool, device=query.device), diagonal=-1)
        qk_distances = qk_distances.masked_fill(mask[None, :, :], float('inf'))
        assert not torch.isnan(qk_distances).any()

        if close_to is not None:
            # close_to has shape (B, S, D) --> distances (B, S)
            v_scale = E ** -0.5
            close_to_distances = (v_scale * self.v_metric.pseudo_distance(value.float(), close_to.float(), squared=True))
            assert not torch.isnan(close_to_distances).any()
            qk_distances = qk_distances + close_to_factor * close_to_distances[:, None, :]
        
        qk_distances = qk_distances.to(value.dtype)

        if batch_lengths is not None:
            batch_mask = torch.arange(S, device=query.device)[None, :] >= batch_lengths[:, None]
            while batch_mask.ndim < qk_distances.ndim:
                batch_mask = batch_mask[..., None]
            qk_distances = qk_distances.masked_fill(batch_mask, float('inf'))

        invalid_mask = (qk_distances >= float('inf')).all(dim=-1, keepdim=True)  # (B, T, 1)
        if batch_lengths is not None:
            length_protected_ones = (torch.arange(S, device=query.device)[None, :] < batch_lengths[:, None]).to(qk_distances.dtype)[:, None, :]
            qk_distances = torch.where(invalid_mask, length_protected_ones.to(qk_distances.dtype), qk_distances)
        else:
            qk_distances = qk_distances.masked_fill(invalid_mask, 1.0)

        mixture_class = IndexedLowRankGaussianMixture if not self.location else IndexedFourierMixture
        if self.location:
            v_std = value_std[:, None, :].expand(B, T, S)
            # v = value.view(-1, 2)
            # v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
            # value = v.view_as(value)
            check_valid_location(value, batch_lengths)
        else:
            v_std = value_std[:, None, :, :].expand(B, T, S, E)

        dist = mixture_class(
            -0.5 * qk_distances,                              #  logits
            self.v_metric,
            value[:, None, :, :].expand(-1, T, -1, -1),       #  center
            v_std,   #  scale
            batch_lengths=batch_lengths,
        )
        return dist, invalid_mask.squeeze(-1)


class GeometricActionDecoder(torch.nn.Module):
    def __init__(self, metric: FourierMetric, location_dim: int, action_dim: int, hidden_dim: int, dropout: float=0.25):
        super().__init__()
        self.metric = metric
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

        self.scale = torch.nn.Parameter(torch.ones(1,1) * 5.0)

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
        next_location = self.metric.block_rotate(location, thetas).view(B, T, D)

        # shift the location one step forward to align with the past; output is one step longer than the input
        next_location = torch.cat([location[:, :1], next_location], dim=1)

        if regularize:
            displacement_loss = loss_for_deltas(thetas, self.metric.K_dagger, self.metric.lattice_basis, self.metric.alphas)

        if not allow_extension:
            next_location = next_location[:, :-1]

        if regularize:
            return next_location, displacement_loss
        else:
            return next_location
    
    def logprobs(self, location: torch.Tensor, mean_location: torch.Tensor):
        B, T, D = location.shape
        comp = FourierCodeDistribution(self.metric, mean_location, self.scale.expand(B, T))
        return comp.log_prob(location)


class TemLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_heads: int=4, 
                       action_hidden_dim: int=128, dropout: float=0.1, compute_window=1024, physical_dim: int=2, 
                       physical_scale: float=10.0, physical_ratio: float=math.sqrt(2.0), fourier: bool=True):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.fourier = fourier

        self.geometric_location_metric = FourierMetric(location_dim, physical_dim, physical_scale, physical_ratio)
        if self.fourier:
            self.location_metric = self.gemoetric_location_metric
        else:
            self.location_metric = PseudoMetric(location_dim, metric_rank=embed_dim)

        self.geometric_action_decoder = GeometricActionDecoder(
            self.geometric_location_metric, location_dim, action_dim, action_hidden_dim, dropout
        )

        self.sensory_metric = PseudoMetric(sensory_dim, metric_rank=embed_dim)
        self.sensory_metric_with_location = PseudoMetric(sensory_dim, metric_rank=embed_dim)

        if self.fourier:
            self.location_scale = torch.nn.Parameter(torch.ones(1, 1) * 5.0)
        else:
            self.location_scale = torch.nn.Parameter(torch.ones(1, 1, 1) * 1.0)

        self.sensory_error_mlp = ErrorMLP(location_dim, sensory_dim, scale=.1)

        self.location_refiner = MetricSampler(
            self.sensory_metric_with_location, self.location_metric, self.sensory_dim, location=self.fourier
        )
        self.sensory_predictor = MetricSampler(
            self.location_metric, self.sensory_metric, self.location_dim, location=False
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

    def sample_fourier_location(self, initial_location: torch.Tensor, location_distribution: D.Distribution, batch_lengths: Optional[torch.Tensor]=None):
        with torch.no_grad():
            location, deltas = location_distribution.sample()
        
        # hold initial location constant
        location[:, :1] = initial_location

        location = check_valid_location(location, batch_lengths)
        return location, deltas
    
    def sample_location(self, initial_location: torch.Tensor, location_distribution: D.Distribution, batch_lengths: Optional[torch.Tensor]=None):
        if self.fourier:
            return self.sample_fourier_location(initial_location, location_distribution, batch_lengths)
        else:
            return location_distribution.sample(), None

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
            initial_location = self.location_metric.sample((B, 1), device=sensory.device, dtype=sensory.dtype)
            return initial_location, initial_location, torch.zeros_like(sensory), 0.0, 0.0, 0.0, 0.0
        
        # These next two ifs let us just supply the previous output location and action sequence, and extend them to the new length
        if prior_location.shape[1] < T:
            prior_location = torch.cat([
                prior_location, 
                torch.stack([
                    torch.ones((B, T - prior_location.shape[1], self.location_dim//2), dtype=prior_location.dtype, device=prior_location.device),
                    torch.zeros((B, T - prior_location.shape[1], self.location_dim//2), dtype=prior_location.dtype, device=prior_location.device),
                ], dim=-1).view(B, -1, self.location_dim),
            ], dim=1)
        
        if action.shape[1] < T - prefix_length.min():
            action = torch.cat([
                action, 
                torch.zeros((B, T - prefix_length.min() - action.shape[1], self.action_dim,), dtype=action.dtype, device=action.device),
            ], dim=1)

        sensory_location = prior_location
        check_nan_inf("prior_location", prior_location)
        check_valid_location(prior_location, batch_lengths)

        initial_location = prior_location[:, :1]

        prior_location_minus_prefix, prior_location_indices = self.remove_prefix(prior_location, prefix_length, batch_lengths)
        geometric_location_minus_prefix, displacement_loss = self.geometric_action_decoder(
            prior_location_minus_prefix, action, allow_extension=False, regularize=True
        ) # <-- we've already extended the action sequence
        geometric_location = self.restore_prefix(prior_location, geometric_location_minus_prefix, prior_location_indices)
        check_valid_location(geometric_location, batch_lengths)
        sensory_plus_geometric = self.make_sensory_keys(geometric_location.detach(), sensory) # <-- stop_gradient 
        check_nan_inf("sensory_plus_geometric", sensory_plus_geometric)
        
        # if sensory_key_prefix is not None and location_prefix is not None:
        #     sensory_plus_geometric_with_prefix = torch.cat([sensory_key_prefix, sensory_plus_geometric], dim=1)
        #     sensory_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
        # else:
        #     sensory_plus_geometric_with_prefix = sensory_plus_geometric
        #    sensory_location_with_prefix = sensory_location

        for k in range(max_steps):
            location_distribution, location_invalid_mask = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location, self.location_scale,
                close_to=geometric_location.detach(), close_to_factor=1.0, batch_lengths=batch_lengths
            )

            with torch.no_grad():
                sensory_location, _ = self.sample_location(initial_location, location_distribution, batch_lengths)

            location_disagreement = self.location_metric.pseudo_distance(geometric_location, sensory_location)

            if (location_disagreement < threshold).all():
                break

        # VAE requires that we sample the encoder, not the decoder, so we use the sensory location as the next location
        with torch.no_grad():
            next_location, displacements = self.sample_location(initial_location, location_distribution, batch_lengths)

        if B > 1:
            print(f"next_location: min: {next_location.min().item()}, mean: {next_location.mean().item()}, max: {next_location.max().item()}")
        check_nan_inf("next_location", next_location) 

        next_location_minus_prefix, _ = self.remove_prefix(next_location, prefix_length, batch_lengths)
        geometric_logprobs = self.geometric_action_decoder.logprobs(
            next_location_minus_prefix, geometric_location_minus_prefix
        )

        if self.fourier:
            sensory_location_logprobs = location_distribution.log_prob(next_location, top_k=32)
        else:
            sensory_location_logprobs = location_distribution.log_prob(next_location, displacements, top_k=32)

        sensory_location_logprobs = location_distribution.log_prob(next_location, displacements, top_k=32)
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
        
        if B > 1:
            print(f"sensory_predicted: min: {sensory_predicted.min().item()}, mean: {sensory_predicted.mean().item()}, max: {sensory_predicted.max().item()}")

        sensory_logprobs = sensory_distribution.log_prob(sensory, top_k=32)
        mask = torch.isnan(sensory_logprobs) | torch.isinf(sensory_logprobs) | sensory_invalid_mask
        mask = mask | (torch.arange(T, device=sensory.device)[None, :] >= batch_lengths[:, None])
        sensory_logprobs = sensory_logprobs.masked_fill(mask, 0.0)
        sensory_logprobs = sensory_logprobs.sum(dim=-1) / ((~mask).to(sensory_logprobs.dtype).sum(dim=-1) + 1e-6)
        sensory_logprobs = sensory_logprobs.mean()
        check_nan_inf("sensory_logprobs", sensory_logprobs)

        sensory_error = (sensory - sensory_predicted).pow(2).sum(dim=-1)
        sensory_error = sensory_error.masked_fill(sensory_invalid_mask, 0.0)
        sensory_error = sensory_error.sum() / ((~sensory_invalid_mask).to(sensory_error.dtype).sum() + 1e-6)

        if not (kl_divergence >= 0.0).all() and B > 1:
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
            fourier=config.location_metric == "fourier"
        )
