import torch
import torchvision
from typing import Optional


def map_index_to_space(indices, R, gamma):
    return gamma * (indices - R)


def map_space_to_preindex(x, R, gamma):
    return x / gamma + R


def gamma_from_real_magnitude(real_magnitude, R):
    """
       For a location spanning -R to R, the gamma is the distance between each point on the grid,
       and the real magnitude is the distance between the origin and the edge of the location along an axis.
    """
    return real_magnitude / R

def real_magnitude_from_gamma(gamma, R):
    """
       For a location spanning -R to R, the gamma is the distance between each point on the grid,
       and the real magnitude is the distance between the origin and the edge of the location along an axis.
    """
    return gamma * R


def create_gaussion_belief_map_grid(R, gamma, sd=None, num_points_per_axis=None):
    """
       Utility function to create a large grid of Gaussian belief maps.
    """
    real_magnitude = real_magnitude_from_gamma(gamma, R)
    if num_points_per_axis is None:
        num_points_per_axis = int(2 * real_magnitude + 1)

        # the
        if num_points_per_axis % 2 == 0:
            num_points_per_axis += 1

    if sd is None:
        # one s.d. is the distsance from one point to the next
        sd = num_points_per_axis / (2*real_magnitude)

    S = 2 * R + 1
    grid_points = torch.linspace(0, S, num_points_per_axis)
    grid = torch.cartesian_prod(grid_points, grid_points).view(num_points_per_axis, num_points_per_axis, 2)  # (101, 101, 2)
    grid_locations = map_index_to_space(grid, R, gamma)

    centroids = grid_locations.view(1, -1, 1, 2)

    location_beliefs = torch.exp(-((grid_locations.view(1, 1, -1, 2) - centroids) / sd).pow(2).sum(dim=-1))
    location_beliefs = location_beliefs / location_beliefs.sum(dim=-1, keepdim=True)

    return location_beliefs

def create_gaussian_belief_map_random(R, gamma, sd=None, num_points_per_axis=None):
    """
    Utility function to create a random Gaussian belief map centered on the grid.
    """
    real_magnitude = real_magnitude_from_gamma(gamma, R)
    if num_points_per_axis is None:
        num_points_per_axis = int(2 * real_magnitude + 1)

        # the number of points per axis must be odd
        if num_points_per_axis % 2 == 0:
            num_points_per_axis += 1

    if sd is None:
        # one s.d. is the distsance from one point to the next
        sd = num_points_per_axis / (2*real_magnitude)

    S = 2 * R + 1
    grid_points = torch.linspace(0, S, num_points_per_axis)
    grid = torch.cartesian_prod(grid_points, grid_points).view(num_points_per_axis, num_points_per_axis, 2)  # (101, 101, 2)
    grid_locations = map_index_to_space(grid, R, gamma)

    centroids = torch.randn(1, num_points_per_axis, 1, 2) * real_magnitude / 5.0
    location_beliefs = torch.exp(-((grid_locations.view(1, 1, -1, 2) - centroids) / sd).pow(2).sum(dim=-1))
    location_beliefs = location_beliefs / location_beliefs.sum(dim=-1, keepdim=True)

    return location_beliefs

def create_initial_gaussian_belief(R, gamma, sd=None):
    """
    Utility function to create a initial Gaussian belief map centered on the grid.
    """
    real_magnitude = real_magnitude_from_gamma(gamma, R)

    if sd is None:
        sd = 0.1 * real_magnitude

    S = 2 * R + 1
        
    grid_points = torch.arange(S)
    grid = torch.cartesian_prod(grid_points, grid_points).view(S, S, 2)  # (101, 101, 2)
    grid_locations = map_index_to_space(grid, R, gamma)

    centroid = torch.zeros(1, 1, 2)
    location_beliefs = torch.exp(-((grid_locations.view(1, -1, 2) - centroid) / sd).pow(2).sum(dim=-1))
    location_beliefs = location_beliefs / location_beliefs.sum(dim=-1, keepdim=True)

    return location_beliefs.view(1, S, S)


class LocationBeliefMemory(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, embed_dim: int, batch_size: int=1, max_memory_size: int=-1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        self.memory_locations = None
        self.memory_values = None

    def reset(self):
        self.memory_locations = None
        self.memory_values = None

    def break_training_graph(self):
        self.memory_locations = self.memory_locations.detach()
        self.memory_values = self.memory_values.detach()

    def write(
        self,
        location_beliefs: torch.Tensor,
        sensory_data: torch.Tensor,
    ) -> None:
        # memory_locations: (B, T, S, S)
        # memory_values: (B, T, D)
        # location_beliefs: (B, S, S) or (B, N, S, S)  # N is the number of memory locations
        # sensory_data: (B, D)
        # return (B, T, D) or (B, N, D)
        assert sensory_data.ndim + 1 == location_beliefs.ndim, "sensory_data and location_beliefs must have a compatible number of dimensions"
        
        if location_beliefs.ndim == 3:
            location_beliefs = location_beliefs.unsqueeze(1)
            sensory_data = sensory_data.unsqueeze(1)

        if self.memory_locations is None or self.memory_values is None:
            assert sensory_data.ndim == 3
            assert location_beliefs.ndim == 4
            self.memory_locations = location_beliefs.clone()
            self.memory_values = sensory_data.clone()
            return

        assert sensory_data.ndim == self.memory_values.ndim == 3  
        assert self.memory_locations.ndim == location_beliefs.ndim == 4
         
        self.memory_locations = torch.cat([self.memory_locations, location_beliefs], dim=1)
        self.memory_values = torch.cat([self.memory_values, sensory_data], dim=1)
        
    def read(
        self,
        location_beliefs: torch.Tensor,
        mask_diagonal: bool = False,
        zero_invalid: bool = True,
        match_threshold: float = -1,
        return_weights: bool = False,
    ) -> torch.Tensor:
        # memory_locations: (B, T, S, S)
        # memory_values: (B, T, D)
        # location_beliefs: (B, S, S) or (B, Q, S, S)  # Q is the number of query locations
        # return (B, D) or (B, Q, D)

        single_query = False
        if location_beliefs.ndim == 3:
            single_query = True
            location_beliefs = location_beliefs.unsqueeze(1)

        if self.memory_locations is None or self.memory_values is None:
            result = torch.zeros(
                location_beliefs.shape[0], location_beliefs.shape[1], self.sensory_dim, 
                device=location_beliefs.device, dtype=location_beliefs.dtype
            )
            if single_query:
                return result.squeeze(1)
            
            if return_weights:
                return result, None
            else:
                return result

        batch_size, time_steps, S, _ = self.memory_locations.shape

        if match_threshold < 0:
            match_threshold = 2 / (S**2)

        num_queries = location_beliefs.shape[1]
        memory_locations = self.memory_locations.view(batch_size, time_steps, -1)
        location_beliefs = location_beliefs.view(batch_size, num_queries, -1)

        # compute the location affinity as (B, Q, T)
        # we use log here because the location affinity is a probability distribution, so the affinity is between 0 and 1
        location_affinity = torch.bmm(location_beliefs, memory_locations.transpose(1, 2))

        if match_threshold is not None:
            assert 0.0 < match_threshold < 1.0, "match_threshold must be greater than 0 and less than 1"
            location_affinity = location_affinity.masked_fill(location_affinity < match_threshold, 0.0)

        if mask_diagonal:
            assert time_steps == num_queries, "mask_diagonal is only supported when the number of query locations is equal to the number of memory locations"
            diagonal_mask = torch.eye(time_steps, device=location_affinity.device, dtype=torch.bool).unsqueeze(0)
            location_affinity = location_affinity.masked_fill(diagonal_mask, 0.0)

        # compute the attention weights as (B, Q, T)
        # we use log here because the location affinity is a probability distribution, so the affinity is between 0 and 1
        attention_weights = torch.softmax(torch.log(location_affinity), dim=-1)

        if zero_invalid:
            invalid = location_affinity.sum(dim=-1, keepdim=True) <= 0
            attention_weights = attention_weights.masked_fill(invalid, 0.0)

        # compute the new memory values as (B, Q, D)
        new_memory_values = torch.bmm(attention_weights, self.memory_values)

        if single_query:
            return new_memory_values.squeeze(1)

        if return_weights:
            return new_memory_values, attention_weights
        else:
            return new_memory_values


    def sample(
        self,
        search_key: torch.Tensor, 
        reference_location: Optional[torch.Tensor]=None,
        num_samples: int=1, 
        temperature: Optional[float]=None, 
        baseline: Optional[float]=None, 
        sharpen: Optional[float]=None, 
        gaussian_blur: Optional[float]=None,
        reference_sharpening: Optional[float]=None,
    ) -> Optional[torch.Tensor]:
        # memory_locations has shape (N, T, S, S)
        # memory_values has shape (N, T, D)
        # search_key has shape (N, D)
        if self.memory_locations is None or self.memory_values is None:
            return None

        N, T, S, _ = self.memory_locations.shape

        if temperature is None:
            temperature = self.memory_values.shape[-1]**(0.5)

        memory_locations = self.memory_locations.view(N, T, S*S)
        if reference_location is not None:
            reference_location = reference_location.view(N, S*S, 1)

        # compute the alignment scores (N, T)
        s_t = torch.bmm(self.memory_values, search_key[..., None]).squeeze(-1)

        if reference_location is not None:
            location_affinity = torch.bmm(memory_locations, reference_location).squeeze(-1)
            location_max = location_affinity.max(dim=-1, keepdim=True).values
            invalid_mask = location_max < 1e-8
            location_affinity = location_affinity / location_max
            if reference_sharpening is not None:
                location_affinity = location_affinity.pow(reference_sharpening)
                
            s_t = torch.where(invalid_mask, s_t, s_t * location_affinity)

        w_t = torch.softmax(s_t / temperature, dim=-1)

        # sample from the mixture, result will be (N, num_samples)
        t = torch.multinomial(w_t, num_samples=num_samples, replacement=True)
        t = t.unsqueeze(-1).repeat(1, 1, S*S)    # (N, num_samples, S^2)

        renormalize = False
        loc = memory_locations.gather(dim=-2, index=t)

        if baseline is not None:
            loc = loc + baseline   # make every location possible; value should be significantly less than 1 / S^2
            renormalize = True

        if sharpen is not None:
            loc = loc.pow(sharpen)
            loc = loc / loc.sum(dim=-1, keepdim=True)
            renormalize = True

        if gaussian_blur is not None:
            img = loc.view(N, -1, S, S)
            img = torchvision.transforms.GaussianBlur(kernel_size=int(2*gaussian_blur + 1), sigma=gaussian_blur)(img)
            loc = img.view(N, -1, S*S)
            renormalize = True
        
        if renormalize:
            loc = loc / loc.sum(dim=-1, keepdim=True)

        return loc.view(N, -1, S, S)

    def generate_prune_candidates(
        self,
        error_leave_one_out: torch.Tensor, 
        dependencies_leave_one_out: torch.Tensor, 
        max_error_to_prune: float=0.05
    ) -> torch.Tensor:
        # remove candidates that are a dependency of another candidate with a lower error
        sorted_error, error_indices = torch.sort(error_leave_one_out, dim=-1)
        unsort_indices = torch.argsort(error_indices, dim=-1)

        dependencies = dependencies_leave_one_out.gather(
            dim=-2, index=error_indices[..., None].repeat(1, 1, dependencies_leave_one_out.shape[-1])
        ).gather(
            dim=-1, index=error_indices[..., None, :].repeat(1, dependencies_leave_one_out.shape[-1], 1)
        )

        # generate a list of all candidates, ignoring dependencies
        candidates = sorted_error < max_error_to_prune

        # remove candidates with zero dependencies
        num_dependencies = dependencies.long().sum(dim=-1)
        candidates = candidates & (num_dependencies > 0)

        # remove candidates that are a dependency of another candidate with a lower error
        dependencies_mask = torch.tril(dependencies, diagonal=-1).any(dim=-1)
        candidates = candidates & dependencies_mask

        return candidates.gather(dim=-1, index=unsort_indices) 


    def prune_one_step(
        self,
        max_error_to_prune: float=0.05, match_threshold: float=None
    ):
        N, T, S, _ = self.memory_locations.shape
        _, _, D = self.memory_values.shape

        if match_threshold is None:
            match_threshold = 2 / (S*S)

        sense, weights = self.read(
            self.memory_locations,
            match_threshold=match_threshold, 
            mask_diagonal=True,
            return_weights=True
        )

        memory_locations = self.memory_locations.view(N, T, S*S)

        error = torch.norm(sense - self.memory_values, dim=-1)
        dependencies = weights > (1 / T)

        prune_candidates = self.generate_prune_candidates(error, dependencies, max_error_to_prune)

        # decide what to prune
        mem_size = prune_candidates.shape[-1]
        prune_size = mem_size -prune_candidates.long().sum(dim=-1).max().item()
        scores = mem_size - prune_candidates.float() * torch.arange(prune_candidates.shape[-1], device=prune_candidates.device)[None, ...]

        _, pruned_indices = torch.topk(scores, k=prune_size, dim=-1)

        pruned_indices_loc = pruned_indices[..., None].repeat(1, 1, S*S)
        pruned_indices_sense = pruned_indices[..., None].repeat(1, 1, D)

        self.memory_locations = memory_locations.gather(dim=-2, index=pruned_indices_loc).view(N, -1, S, S)
        self.memory_values = self.memory_values.gather(dim=-2, index=pruned_indices_sense)

        num_pruned = prune_candidates.sum(dim=-1)

        return num_pruned

    def prune(
        self, 
        max_error_to_prune: float=0.05, 
        match_threshold: float=None, 
        max_prune_steps: int=10
    ):
        starting_size = self.memory_locations.shape[1]
        for _ in range(max_prune_steps):
            num_pruned = self.prune_one_step(max_error_to_prune, match_threshold)
            if num_pruned.max().item() == 0:
                break

        total_pruned = starting_size - self.memory_locations.shape[1]

        return total_pruned
        
    