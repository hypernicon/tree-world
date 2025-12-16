import math
import torch

from typing import Optional


def make_lattice_basis(alphas, dim: int = 2):
    """
    Construct lattice basis matrices B_j for each grid module j in d dimensions.

    Args:
        alphas: 1D tensor of shape (J,) with spatial frequencies α_j.
        dim: spatial dimension d (>=1).

    Returns:
        B: tensor of shape (J, d, d), where B[j] is the lattice basis B_j.
        K: tensor of shape (d+1, d) with simplex directions as rows.
        K_dagger: tensor of shape (d, d+1) with pseudoinverse of K, returned for convenience.
    """
    if dim < 1:
        raise ValueError("dimension must be >= 1")

    # Ensure alphas is a tensor of shape (J,)
    dtype = alphas.dtype
    device = alphas.device
    J = alphas.shape[0]

    # 1. Build a basis U for the null subspace of the simplex K
    U = torch.cat([torch.eye(dim, dtype=dtype, device=device), -torch.ones(1, dim, dtype=dtype, device=device)], dim=0)

    # 2. Build a regular simplex in R^(d+1) and normalize
    Q, _ = torch.linalg.qr(U, mode="reduced")  # Q: (d+1, d)
    V = torch.eye(dim + 1, dtype=dtype, device=device) - (1.0 / (dim + 1))
    K = V @ Q  # (d+1, d)
    K = K / K.norm(dim=1, keepdim=True)  # each row is now unit length

    # Pseudoinverse of K: K^† ∈ R^{d×(d+1)}
    K_dagger = torch.linalg.pinv(K)

    # Base (unscaled) lattice generator: shape (d, d)
    base = K_dagger @ U

    # 3. Scale by 2π / α_j for each module
    scale = (2.0 * math.pi) / alphas.view(J, 1, 1)  # (J, 1, 1)
    B = scale * base[None, ...]                  # (J, d, d)

    return B, K, K_dagger


def make_alphas(location_dim: int, dim: int = 2, scale: float = 10.0,
                ratio: float = math.sqrt(2.0), dtype=torch.get_default_dtype(), device=torch.device("cpu")) -> torch.Tensor:
    """
    Choose module spatial frequencies alpha_j given a location code size and
    a target "safe" displacement scale.

    Args:
        location_dim: int, total number of phase channels = J * (d+1).
        dim: spatial dimension d.
        scale: radius such that for ||Δx|| < scale, the coarsest module
               is unambiguous (λ_0/2 ≈ scale).
        ratio: geometric ratio between successive periods (default √2).

    Returns:
        alphas: (J,) tensor of spatial frequencies α_j, with j=0 coarsest.
    """
    num_dirs = dim + 1
    assert location_dim % (2 * num_dirs) == 0,  f"location_dim={location_dim} must be divisible by 2*(dimension+1)={2*num_dirs}"
    J = location_dim // num_dirs // 2  # number of modules
    j_idx = torch.arange(J, dtype=dtype, device=device)
    return (math.pi / scale) * ratio ** j_idx


def solve_for_deltas(delta_thetas: torch.Tensor, K_dagger: torch.Tensor, lattice_basis: torch.Tensor, alphas: torch.Tensor):
    d, dplus = K_dagger.shape
    J, _, _ = lattice_basis.shape
    batch_size, time_steps, L = delta_thetas.shape
    delta_thetas = delta_thetas.view(batch_size, time_steps, J, dplus, 1)
    displacement_base = (K_dagger[None, None, None, ...] @ delta_thetas).view(batch_size, time_steps, J, d) / alphas[None, None, :, None]

    # displacement_base is of shape (batch_size, time_steps, J, d)
    reference_displacement = displacement_base[..., 0, :]
    errors = reference_displacement[..., None, :] - displacement_base

    lattice_basis = lattice_basis.view(1, 1, J, d, d)
    offsets = torch.linalg.solve(lattice_basis.float(), errors[..., None].float()).round().to(delta_thetas.dtype)
    deltas = displacement_base + (lattice_basis @ offsets).squeeze(-1)
    return deltas


def loss_for_deltas(delta_thetas: torch.Tensor, K_dagger: torch.Tensor, lattice_basis: torch.Tensor, alphas: torch.Tensor):
    deltas = solve_for_deltas(delta_thetas, K_dagger, lattice_basis, alphas)

    # deltas has shape (batch_size, time_steps, J, d)
    return deltas.var(dim=-2).mean()


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, num_heads: int=1):
    # Convert our inputs, which are (batch_size, time_steps, embed_dim) to (batch_size, num_heads, time_steps, head_dim)
    B, T, _ = query.shape
    print(f"SDPA query shape: {query.shape} key shape: {key.shape} value shape: {value.shape}")
    query = query.view(query.shape[0], query.shape[1], num_heads, -1).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], num_heads, -1).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], num_heads, -1).transpose(1, 2)
    result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
    print(f"SDPA initial result shape: {result.shape}")
    result = result.transpose(1, 2).view(B, T, -1)
    print(f"SDPA result shape: {result.shape}")
    return result


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
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, location_dim // 2),
        )

        self.physical_dim = physical_dim
        self.physical_scale = physical_scale
        self.physical_ratio = physical_ratio
        self.alphas = torch.nn.Buffer(make_alphas(location_dim, physical_dim, physical_scale, physical_ratio))
        lattice_basis, _, K_dagger = make_lattice_basis(self.alphas, physical_dim)
        self.lattice_basis = torch.nn.Buffer(lattice_basis)
        self.K_dagger = torch.nn.Buffer(K_dagger)

        assert self.location_dim % (2 * self.physical_dim) == 0

    def forward(self, location: torch.Tensor, action: torch.Tensor, eps: float=1e-6, allow_extension: bool=True, regularize: bool=True):
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

        if regularize:
            displacement_loss = loss_for_deltas(thetas, self.K_dagger, self.lattice_basis, self.alphas)

        if not allow_extension:
            next_location = next_location[:, :-1]

        if regularize:
            return next_location, displacement_loss
        else:
            return next_location



class TemLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_heads: int=8, 
                       action_hidden_dim: int=128, dropout: float=0.1, compute_window=1024, physical_dim: int=2, 
                       physical_scale: float=10.0, physical_ratio: float=math.sqrt(2.0)):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.location_refiner = TemTransformerLayer(sensory_dim, location_dim, embed_dim, num_heads, dropout)
        # self.sensory_predictor = TemTransformerLayer(location_dim, sensory_dim, embed_dim, num_heads, dropout, use_ffn=False)

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
        for k in range(max_steps):
            sensory_location = self.location_refiner(
                sensory_plus_geometric, sensory_plus_geometric, sensory_location,
                key_prefix=sensory_key_prefix, value_prefix=location_prefix,
                causal=False
            )
            sensory_location = torch.tanh(sensory_location)

            location_disagreement = (
                geometric_location - sensory_location
            ).pow(2).sum(dim=-1)

            if (location_disagreement < threshold).all():
                break

            sensory_location = (1 - refine_alpha) * sensory_location + refine_alpha * geometric_location.detach()

        # train the sensory predictor on the prefix too, if present
        # the prefix is all prior salient info, so this should be prioritized in training.
        sensory_location_with_prefix = sensory_location
        sensory_with_prefix = sensory
        if sensory_prefix is not None and location_prefix is not None:
            sensory_location_with_prefix = torch.cat([location_prefix, sensory_location], dim=1)
            sensory_with_prefix = torch.cat([sensory_prefix, sensory], dim=1)

        # sensory_predicted = self.sensory_predictor(
        #    sensory_location_with_prefix, sensory_location_with_prefix, sensory_with_prefix, 
        #    allow_self_attention=False, add_residual=False, causal=False
        #)
        I = torch.eye(T, dtype=torch.bool, device=sensory_location_with_prefix.device)
        mask = torch.zeros((T, T), dtype=sensory.dtype, device=sensory.device).masked_fill(I, float('-inf'))
        sensory_predicted = scaled_dot_product_attention(
            sensory_location_with_prefix, sensory_location_with_prefix, sensory_with_prefix, attn_mask=mask, num_heads=1
        )

        sensory_error = (sensory_with_prefix - sensory_predicted).pow(2).sum(dim=-1)

        return geometric_location, sensory_location, sensory_predicted, sensory_error.mean(), location_disagreement.mean(), displacement_loss
    
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
