import torch
import math


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
    shape = delta_thetas.shape[:-1] + (J, dplus, 1)
    delta_thetas = delta_thetas.view(shape)
    while K_dagger.ndim < delta_thetas.ndim:
        K_dagger = K_dagger[None, ...]
    displacement_base = (K_dagger @ delta_thetas).view(shape[:-1] + (d,)) / alphas[None, :, None]
    reference_displacement = displacement_base[..., 0, :]
    errors = reference_displacement[..., None, :] - displacement_base
    lattice_basis = lattice_basis.view(J, d, d)
    while lattice_basis.ndim < errors.ndim + 1:
        lattice_basis = lattice_basis[None, ...]
    offsets = torch.linalg.solve(lattice_basis.float(), errors[..., None].float()).round().to(delta_thetas.dtype)
    deltas = displacement_base + (lattice_basis @ offsets).squeeze(-1)
    return deltas.view(shape[:-1] + (d,))
