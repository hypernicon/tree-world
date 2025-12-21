import math
import torch

from typing import Optional
from ..fourier import make_alphas, make_lattice_basis, solve_for_deltas
from .metric import PseudoMetric


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


class ErrorMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
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
    def __init__(self, qk_metric: PseudoMetric, v_metric: PseudoMetric, v_error_mlp: ErrorMLP, qk_dim: int):
        super().__init__()
        self.qk_metric = qk_metric
        self.v_metric = v_metric
        self.v_error_mlp = v_error_mlp
        self.qk_dim = qk_dim
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, qk_std: Optional[torch.Tensor]=None):
        B, T, D = query.shape
        B, S, D = key.shape
        B, T, E = value.shape
        assert D == self.qk_dim

        op_norm = self.qk_metric.metric_operator_norm()
        scale = self.scale_factor ** 0.5 / (op_norm ** 0.5 + 1e-8)

        if qk_std is not None:
            scale = scale / qk_std

        qk_distances = self.qk_metric.cross_distance(query, key, squared=True, scale=scale)
        mask = torch.tril(torch.ones((max(S, T), max(S, T)), dtype=torch.bool, device=query.device), diagonal=-1)
        qk_distances = qk_distances.masked_fill(mask[None, :, :], float('inf'))

        invalid_mask = (qk_distances >= float('inf')).all(dim=-1, keepdim=True)  # (B, T, 1)

        qk_weights = torch.softmax(-0.5 * qk_distances, dim=-1)

        # uniform sample for invalid rows... we'll fix this later
        qk_weights = qk_weights.masked_fill(invalid_mask, 1.0 / S)
        
        return qk_weights, invalid_mask.squeeze(-1)
    
    def sample(self, qk_weights: torch.Tensor, invalid_mask: torch.Tensor, value_mean: torch.Tensor):
        B, T, S = qk_weights.shape
        E = self.value_dim

        v_std = self.v_error_mlp(value_mean)

        sample_indices = torch.multinomial(qk_weights.view(-1, S), num_samples=1).view(B, T, 1, 1)
        sampled_value = value_mean[:, None, ...].repeat(1, T, 1, 1).gather(dim=-2, index=sample_indices.repeat(1, 1, 1, E)).squeeze(-2)

        sampled_value = sampled_value + torch.randn_like(sampled_value) * v_std
        sampled_value = sampled_value.masked_fill(invalid_mask[..., None], 0.0)

        return sampled_value, v_std

    def logprobs(self, qk_weights: torch.Tensor, value: torch.Tensor, value_mean: torch.Tensor, v_std: torch.Tensor):

        # compute the log probability of sampled_value
        # this is a mixture of gaussians, so unfortunately we can't use a simple log probability formula
        # how far is sampled_value from EVERY value? (B, T, S)
        sampled_value_distances = self.v_metric.cross_distance(value, value_mean, squared=True, scale=1./(v_std + 1e-8))
        sampled_value_probs = torch.exp(-0.5 * sampled_value_distances)  # B, T, S
        core_logprobs = torch.log((qk_weights * sampled_value_probs).sum(dim=-1) + 1e-8)  # B, T
        logprobs = core_logprobs - 0.5 * math.log(2 * math.pi) - torch.log(v_std + 1e-8).sum(dim=-1) # B, T
        return logprobs


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

        self.error_mlp = ErrorMLP(location_dim)

        assert self.location_dim % (2 * self.physical_dim) == 0

    def forward(self, location: torch.Tensor, action: torch.Tensor, eps: float=1e-6, allow_extension: bool=True, regularize: bool=True):
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
        self.sensory_metric_with_location = PseudoMetric(sensory_dim + location_dim, dim=physical_dim, scale=physical_scale, ratio=physical_ratio, metric_rank=embed_dim)

        self.location_error_mlp = ErrorMLP(location_dim)
        self.sensory_error_mlp = ErrorMLP(sensory_dim)

        self.location_refiner = MetricSampler(self.sensory_metric_with_location, self.location_metric, self.location_error_mlp, self.location_dim)
        self.sensory_predictor = MetricSampler(self.location_metric, self.sensory_metric, self.sensory_error_mlp, self.sensory_dim)

        self.geometric_action_decoder = GeometricActionDecoder(
            location_dim, action_dim, action_hidden_dim, dropout, physical_dim, physical_scale, physical_ratio
        )

        self.position_encoder = torch.nn.Linear(location_dim, sensory_dim, bias=False)

    def forward(self, sensory: torch.Tensor, prior_location: Optional[torch.Tensor]=None, action: Optional[torch.Tensor]=None, 
                sensory_prefix: Optional[torch.Tensor]=None, sensory_key_prefix: Optional[torch.Tensor]=None, 
                location_prefix: Optional[torch.Tensor]=None,
                max_steps: int=4, threshold: float=0.05, refine_alpha: float=0.1, eps: float=1e-6):
        assert max_steps > 0

        B, T, S = sensory.shape
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
        
        if action.shape[1] < T:
            action = torch.cat([
                action, 
                torch.zeros((B, T - action.shape[1], self.action_dim), 
                dtype=action.dtype, 
                device=action.device
            )], dim=1)

        sensory_location = prior_location
        geometric_location, displacement_loss = self.geometric_action_decoder(
            prior_location, action, allow_extension=False, regularize=True
        ) # <-- we've already extended the action sequence
        sensory_plus_geometric = self.make_sensory_keys(geometric_location.detach(), sensory) # <-- stop_gradient     
        
        # if sensory_key_prefix is not None and location_prefix is not None:
        #     sensory_plus_geometric_with_prefix = torch.cat([sensory_key_prefix, sensory_plus_geometric], dim=1)
        #     sensory_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
        # else:
        #     sensory_plus_geometric_with_prefix = sensory_plus_geometric
        #    sensory_location_with_prefix = sensory_location

        for k in range(max_steps):
            location_weights, location_invalid_mask = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location, None
            )

            sensory_location, location_std = self.location_refiner.sample(location_weights, location_invalid_mask, sensory_location)

            location_disagreement = self.location_metric.psuedo_distance(geometric_location, sensory_location)

            if (location_disagreement < threshold).all():
                break

            # sensory_location = (1 - refine_alpha) * sensory_location + refine_alpha * geometric_location.detach()

        # ... should we move towards the geometric location?
        next_location = 0.5 * (geometric_location + sensory_location).detach()

        geometric_logprobs = self.geometric_action_decoder.logprobs(next_location, geometric_location)
        sensory_location_logprobs = self.sensory_predictor.logprobs(location_weights, next_location, sensory_location, location_std)

        kl_divergence = sensory_location_logprobs - geometric_logprobs
        kl_divergence = kl_divergence.masked_fill(location_invalid_mask, 0.0)
        kl_divergence = kl_divergence.sum(dim=-1) / ((~location_invalid_mask).to(kl_divergence.dtype).sum(dim=-1) + 1e-8)

        # train the sensory predictor on the prefix too, if present
        # the prefix is all prior salient info, so this should be prioritized in training.
        next_location_with_prefix = next_location
        sensory_with_prefix = sensory
        if sensory_prefix is not None and location_prefix is not None:
            next_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
            sensory_with_prefix = torch.cat([sensory_prefix, sensory], dim=1)

        sensory_weights, sensory_invalid_mask = self.sensory_predictor(
            next_location_with_prefix, next_location_with_prefix, sensory_with_prefix, location_std, None
        )

        sensory_predicted, sensory_std = self.sensory_predictor.sample(sensory_weights, sensory_invalid_mask, sensory_with_prefix)

        sensory_logprobs = self.sensory_predictor.logprobs(sensory_weights, sensory_with_prefix, sensory_predicted, sensory_std)
        sensory_logprobs = sensory_logprobs.masked_fill(sensory_invalid_mask, 0.0)
        sensory_logprobs = sensory_logprobs.sum(dim=-1) / ((~sensory_invalid_mask).to(sensory_logprobs.dtype).sum(dim=-1) + 1e-8)
        sensory_logprobs = sensory_logprobs.mean()

        sensory_error = (sensory_with_prefix - sensory_predicted).pow(2).sum(dim=-1)
        sensory_error = sensory_error.masked_fill(sensory_invalid_mask, 0.0)
        sensory_error = sensory_error.sum() / ((~sensory_invalid_mask).to(sensory_error.dtype).sum() + 1e-8)

        elbo = sensory_logprobs - kl_divergence.mean()

        return next_location, sensory_location, sensory_predicted, elbo, sensory_error.mean(), location_disagreement.mean(), displacement_loss
    
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
