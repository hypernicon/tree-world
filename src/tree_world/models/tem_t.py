import math
import torch

from typing import Optional
from ..fourier import make_alphas, make_lattice_basis, solve_for_deltas
from .fourier_code import LocationMetric


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


class SensoryPredictor(torch.nn.Module):
    def __init__(self, metric: LocationMetric, location_dim: int):
        super().__init__()
        self.metric = metric
        self.location_dim = location_dim
        self.scale_factor = location_dim ** -0.5
    
    def forward(self, search_locations: torch.Tensor, memory_locations: torch.Tensor, sensory: torch.Tensor, max_distance: float=0.10):

        # locations have shape (batch_size, time_steps, location_dim)
        # sensory has shape (batch_size, time_steps, sensory_dim)
        B, S, D = memory_locations.shape
        B, T, D = search_locations.shape
        assert D == self.location_dim

        search_proj = self.metric.prepare_q(search_locations)
        memory_proj = self.metric.prepare_k(memory_locations)
        op_norm = self.metric.metric_operator_norm()
        scale = self.scale_factor / (op_norm + 1e-8)
        
        # location_affinity has shape (batch_size, time_steps, time_steps)
        location_affinity = torch.bmm(search_proj, memory_proj.transpose(-2, -1))
        # search_diagonal = search_proj.pow(2).sum(dim=-1)
        # memory_diagonal = memory_proj.pow(2).sum(dim=-1)

        # location_distances = (search_diagonal[..., None] - 2 * location_affinity + memory_diagonal[..., None, :]).pow(0.5) * scale

        location_affinity = location_affinity * scale
        
        # location_affinity = location_affinity.masked_fill(location_distances > max_distance, float('-inf'))
        
        mask = torch.eye(max(S, T), dtype=torch.bool, device=search_locations.device)[:S, :T]
        location_affinity = location_affinity.masked_fill(mask[None, :, :], float('-inf'))

        invalid_mask = (location_affinity <= float('-inf')).all(dim=-1, keepdim=True)

        location_weights = torch.softmax(location_affinity, dim=-1)
        entropy = - (location_weights * torch.log(location_weights + 1e-8)).sum(dim=-1)

        location_weights = location_weights.masked_fill(invalid_mask, 0.0)
        entropy = entropy.masked_fill(invalid_mask, 0.0)
        print(f"entropy: min {entropy.min()}, mean {entropy.mean()}, max {entropy.max()}")

        sensory_predicted = torch.bmm(location_weights, sensory)

        return sensory_predicted, invalid_mask.squeeze(-1)


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

        self.location_metric = LocationMetric(location_dim, dim=physical_dim, scale=physical_scale, ratio=physical_ratio, metric_rank=embed_dim)
        self.location_refiner = TemTransformerLayer(sensory_dim, location_dim, num_heads*location_dim, num_heads, dropout)
        # self.sensory_predictor = TemTransformerLayer(location_dim, sensory_dim, embed_dim, num_heads, dropout)
        self.sensory_predictor = SensoryPredictor(self.location_metric, location_dim)

        self.geometric_action_decoder = GeometricActionDecoder(
            location_dim, action_dim, action_hidden_dim, dropout, physical_dim, physical_scale, physical_ratio
        )

        self.sensory_error_mlp = ErrorMLP(sensory_dim, 1)
        self.location_error_mlp = ErrorMLP(location_dim, 1)

        self.position_encoder = torch.nn.Linear(location_dim, sensory_dim, bias=False)

    def forward(self, sensory: torch.Tensor, prior_location: Optional[torch.Tensor]=None, action: Optional[torch.Tensor]=None, 
                sensory_prefix: Optional[torch.Tensor]=None, sensory_key_prefix: Optional[torch.Tensor]=None, 
                location_prefix: Optional[torch.Tensor]=None,
                max_steps: int=4, threshold: float=0.05, refine_alpha: float=0.1, eps: float=1e-6):
        assert max_steps > 0

        B, T, S = sensory.shape
        if prior_location is None:
            initial_location = torch.empty((B, 1, self.location_dim), dtype=sensory.dtype, device=sensory.device).uniform_(-1, 1)
            return initial_location, initial_location, torch.zeros_like(sensory), 0.0, 0.0, 0.0
        
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
        geometric_location, displacement_loss = self.geometric_action_decoder(prior_location, action, allow_extension=False, regularize=True) # <-- we've already extended the action sequence
        sensory_plus_geometric = self.make_sensory_keys(geometric_location.detach(), sensory) # <-- stop_gradient     
        
        # if sensory_key_prefix is not None and location_prefix is not None:
        #     sensory_plus_geometric_with_prefix = torch.cat([sensory_key_prefix, sensory_plus_geometric], dim=1)
        #     sensory_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
        # else:
        #     sensory_plus_geometric_with_prefix = sensory_plus_geometric
        #    sensory_location_with_prefix = sensory_location

        for k in range(max_steps):
            sensory_location = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location,
                key_prefix=sensory_key_prefix, value_prefix=location_prefix,
                causal=False
            )
            
            # sensory_location = scaled_dot_product_attention(
            #    sensory_plus_geometric, sensory_plus_geometric_with_prefix, sensory_location_with_prefix, 
            #    attn_mask=None, num_heads=self.num_heads
            #)
            sensory_location = torch.tanh(sensory_location)

            location_disagreement = self.location_metric.psuedo_distance(geometric_location, sensory_location)

            if (location_disagreement < threshold).all():
                break

            # sensory_location = (1 - refine_alpha) * sensory_location + refine_alpha * geometric_location.detach()

        next_location = 0.5 * (geometric_location.detach() + sensory_location)

        # train the sensory predictor on the prefix too, if present
        # the prefix is all prior salient info, so this should be prioritized in training.
        next_location_with_prefix = next_location
        sensory_with_prefix = sensory
        if sensory_prefix is not None and location_prefix is not None:
            next_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
            sensory_with_prefix = torch.cat([sensory_prefix, sensory], dim=1)

        sensory_predicted, invalid_mask = self.sensory_predictor(next_location_with_prefix, next_location_with_prefix, sensory_with_prefix, max_distance=0.10)

        # next_location_with_prefix_k = self.location_metric.prepare_k(next_location_with_prefix)
        # next_location_with_prefix_q = self.location_metric.prepare_q(next_location_with_prefix)

        # sensory_predicted = self.sensory_predictor(
        #    sensory_location_with_prefix, sensory_location_with_prefix, sensory_with_prefix, 
        #    allow_self_attention=False, add_residual=False, causal=False
        #)
        # S = next_location_with_prefix.shape[1]
        # I = torch.eye(S, dtype=torch.bool, device=next_location_with_prefix.device)
        # mask = torch.zeros((S, S), dtype=sensory.dtype, device=sensory.device).masked_fill(I, float('-inf'))
        # TODO: we need to prevent locations from attending to faraway locations
        # sensory_predicted = scaled_dot_product_attention(
        #     next_location_with_prefix_q, next_location_with_prefix_k, sensory_with_prefix, attn_mask=mask, num_heads=1
        # )

        std_sensory = self.sensory_error_mlp(sensory_with_prefix)
        std_location = self.location_error_mlp(next_location)

        sensory_error = ((sensory_with_prefix - sensory_predicted)/std_sensory).pow(2).sum(dim=-1)
        sensory_error = sensory_error.masked_fill(invalid_mask, 0.0)
        sensory_error = sensory_error.sum() / ((~invalid_mask).to(sensory_error.dtype).sum() + 1e-8)

        location_error = location_disagreement / std_location.pow(2)

        print(f"std sensory min {std_sensory.min()}, mean {std_sensory.mean()}, max {std_sensory.max()}", end="\t")
        print(f"std location min {std_location.min()}, mean {std_location.mean()}, max {std_location.max()}")

        return next_location, sensory_location, sensory_predicted, sensory_error.mean(), location_error.mean(), displacement_loss
    
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
