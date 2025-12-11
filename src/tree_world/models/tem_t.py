import torch

from typing import Optional


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

        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=False, batch_first=True)
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
        mask: Optional[torch.Tensor]=None
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
            if allow_self_attention:
                diagonal = 1
            else:
                diagonal = 0

            base = torch.full((query.shape[1], key.shape[1]), float('-inf'), dtype=query.dtype, device=query.device)
            mask = torch.triu(base, diagonal=diagonal)

            if not allow_self_attention:
                # CRITICAL & NONOBVIOUS: prevent NaN in the gradient of the first query; we'll patch the output later
                mask[0, :] = 0.0

        attn_output, attn_output_weights = self.attention(query, key, value_p, attn_mask=mask)

        if not allow_self_attention:
            attn_output[:, 0, :] = torch.zeros_like(attn_output[:, 0, :])
            if attn_output.shape[1] > 1:
                attn_output[:, 1, :] = value_p[:, 0, :]

        attn_output = self.v_out(attn_output)

        if add_residual:
            x = orig_value + self.feed_forward_norm(attn_output)
        else:
            x = self.feed_forward_norm(attn_output)

        y = self.feed_forward(x)
        return y + x


class GeometricActionDecoder(torch.nn.Module):
    def __init__(self, location_dim: int, action_dim: int, hidden_dim: int, dropout: float=0.25):
        super().__init__()
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        assert location_dim % 2 == 0

        self.action_mlp = torch.nn.Sequential(
            torch.nn.Linear(action_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, location_dim // 2),
        )

    def forward(self, location: torch.Tensor, action: torch.Tensor, eps: float=1e-6, allow_extension: bool=True):
        B, T, D = location.shape
        assert D == self.location_dim
        assert (B, T, self.action_dim) == action.shape

        # use a block diagonal matrix to rotate the location
        thetas = self.action_mlp(action)
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

        if allow_extension:
            return next_location
        else:
            return next_location[:, :-1]


class TemLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_heads: int=8, 
                       action_hidden_dim: int=128, dropout: float=0.1, compute_window=1024):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.location_refiner = TemTransformerLayer(sensory_dim, location_dim, embed_dim, num_heads, dropout)
        self.sensory_predictor = TemTransformerLayer(location_dim, sensory_dim, embed_dim, num_heads, dropout)

        self.geometric_action_decoder = GeometricActionDecoder(location_dim, action_dim, action_hidden_dim, dropout)

        self.position_encoder = torch.nn.Linear(location_dim, sensory_dim, bias=False)

    def forward(self, sensory: torch.Tensor, prior_location: Optional[torch.Tensor]=None, action: Optional[torch.Tensor]=None, 
                sensory_prefix: Optional[torch.Tensor]=None, sensory_key_prefix: Optional[torch.Tensor]=None, 
                location_prefix: Optional[torch.Tensor]=None,
                max_steps: int=4, threshold: float=0.05, refine_alpha: float=0.1, eps: float=1e-6):
        assert max_steps > 0

        B, T, S = sensory.shape
        if prior_location is None:
            initial_location = torch.empty((B, 1, self.location_dim), dtype=sensory.dtype, device=sensory.device).uniform_(-1, 1)
            return initial_location, initial_location, torch.zeros_like(sensory), torch.zeros_like(sensory), 0.0, 0.0
        
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
        geometric_location = self.geometric_action_decoder(prior_location, action, allow_extension=False) # <-- we've already extended the action sequence
        sensory_plus_geometric = sensory + self.position_encoder(geometric_location.detach()) # <-- stop_gradient     
        for k in range(max_steps):
            sensory_location = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location,
                key_prefix=sensory_key_prefix, value_prefix=location_prefix
            )

            location_disagreement = (
                geometric_location - sensory_location
            ).pow(2).sum(dim=-1)

            if (location_disagreement < threshold).all():
                break

            sensory_location = (1 - refine_alpha) * sensory_location + refine_alpha * geometric_location.detach()

        sensory_predicted = self.sensory_predictor(
            sensory_location, sensory_location, sensory, allow_self_attention=False, add_residual=False,
            key_prefix=location_prefix, value_prefix=sensory_prefix
        )
        sensory_error = (sensory - sensory_predicted).pow(2).sum(dim=-1)  # <-- or, if we want to use a norm comparison

        return geometric_location, sensory_location, sensory_predicted, sensory_plus_geometric, sensory_error.mean(), location_disagreement.mean()

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
        )
