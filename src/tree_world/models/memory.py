import math
import torch


def geodesic_distance(location1: torch.Tensor, location2: torch.Tensor):
    """
    Calculate the geodesic distance between two points on the unit sphere.
    """
    dot = (location1 * location2).sum(dim=-1)
    dot = dot.clamp(-1.0, 1.0)
    return torch.acos(dot) / torch.pi


def normalize_location(location: torch.Tensor, temperature: float=0.25):
    """ 
    Convert a point in free R^d to the simplex and thence to the positive orthant of the unit sphere.
    """
    first_part = torch.exp(location / temperature)
    second_part = 1 + first_part.sum(dim=-1, keepdim=True)
    return torch.sqrt(torch.cat([first_part / second_part, 1 / second_part], dim=-1))


def unnormalize_location(location: torch.Tensor, temperature: float=0.25):
    """ Convert a point on the unit sphere back to free R^d.
    """
    return temperature * (2 * (torch.log(location[..., :-1]) - torch.log(location[..., -1])[..., None]))


def make_key_grid(embed_dim: int, points_per_dim: int, epsilon: float=1e-6):
    if embed_dim > 8:
        raise ValueError("Embedding dimension too high for grid search. Use a lower dimension or a different key generation method.")

    # first, space out a line with the desired increment between points
    points = torch.linspace(-1 + epsilon, 1 - epsilon, points_per_dim)

    # unfold to infinity, with concentration at the origin (half the points will be between -1 and 1)
    points = points / (1 - points.abs())

    # then, create a grid of points by taking the cartesian product of the points
    grid = torch.cartesian_prod([points] * embed_dim)
    return grid, grid.shape[0]


class BidirectionalMemory(torch.nn.Module):
    """
    This is a memory cache to store pairs of (key, value) embeddings.

    :param key_dim: The dimension of the key embeddings.
    :param value_dim: The dimension of the value embeddings.
    :param memory_size: The maximum length of the memory cache.
    :param dropout: The dropout rate.

    """
    def __init__(self, location_dim: int, sensory_dim: int, embed_dim: int, batch_size: int=1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size

        self.sensory_proj = torch.nn.Linear(sensory_dim, embed_dim, bias=False)
        self.sensory_read_proj = torch.nn.Linear(embed_dim, sensory_dim, bias=False)

        self.memory_locations = None
        self.memory_senses = None
        self.memory_location_sds = None

        self.sensory_factor = sensory_dim ** -0.5
        self.location_factor = location_dim ** -0.5

    def reset(self):
        self.memory_locations = None
        self.memory_senses = None
        self.memory_location_sds = None

    def break_training_graph(self):
        if self.memory_locations is not None:
            self.memory_locations = self.memory_locations.detach()
        if self.memory_senses is not None:
            self.memory_senses = self.memory_senses.detach()
        if self.memory_location_sds is not None:
            self.memory_location_sds = self.memory_location_sds.detach()

    def score(self, query: torch.Tensor, keys: torch.Tensor, factor: float=1.0):
        """
        Score a single queries against the keys.

        :param query: The query to read from the memory cache. Has shape (batch_size, embed_dim) or (batch_size, num_queries, embed_dim).
        :param keys: The keys to read from the memory cache. Has shape (batch_size, num_keys, embed_dim).
        :return: The scores. Has shape (batch_size, num_keys) or (batch_size, num_queries, num_keys).
        """
        squeeze = False
        if query.ndim < keys.ndim:
            query = query[..., None, :]
            squeeze = True
        return factor * torch.bmm(query, keys.transpose(-2, -1))
        if squeeze:
            return result.squeeze(dim=-2)
        else:
            return result

    def get_location_affinity(self, location: torch.Tensor, location_sd: torch.Tensor, mask_diagonal: bool=False, raw_weights: bool=False):
        """
        Get the affinity of a location to the memory locations.

        :param location: The location to get the affinity for. Has shape (batch_size, location_dim).
        :param location_sd: The standard deviation of the location. Has shape (batch_size, location_dim).
        :return: The affinity. Has shape (batch_size, num_keys).
        """
        if location.ndim < self.memory_locations.ndim:
            squeeze = True
            location = location[..., None, :]

        if location_sd.ndim < self.memory_location_sds.ndim:
            location_sd = location_sd[..., None, :]


        # shape (batch_size, num_queries, num_keys, location_dim)
        location_delta = location[..., None, :] - self.memory_locations[..., None, :, :]
        location_delta_sd = location_sd[..., None, :] + self.memory_location_sds[..., None, :, :]

        log_location_affinity = (
            - 0.5 * (location_delta / (location_delta_sd + 1e-8)).pow(2).sum(dim=-1) 
            - 0.5 * math.log(2 * math.pi) * location_delta_sd.shape[-1]
            - torch.log(location_delta_sd).sum(dim=-1)
        )

        if mask_diagonal:
            # TODO: this will fail if num_queries != num_keys or if there is more than one batch dimension
            eye = torch.eye(log_location_affinity.size(-2), device=log_location_affinity.device).bool()
            log_location_affinity = log_location_affinity.masked_fill(eye.unsqueeze(0), float('-inf'))

        if raw_weights:
            return log_location_affinity

        # shape (batch_size, num_queries, num_keys)
        location_weights = torch.softmax(log_location_affinity, dim=-1)
        location_weight_mean = location_weights.mean(dim=-1, keepdim=True)
        location_weights = location_weights.masked_fill(location_weights < location_weight_mean, 0.0)
        location_weights = location_weights / (location_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return location_weights
    
    def get_location_and_sensory_affinity(self, location: torch.Tensor, location_sd: torch.Tensor, sensory: torch.Tensor):
        """
        Get the affinity of a location and sensory to the memory locations and senses.
        """
        location_affinity = self.get_location_affinity(location, location_sd, raw_weights=True)
        sensory_affinity = self.score(sensory, self.memory_senses, factor=self.sensory_factor)
        return location_affinity, sensory_affinity

    def read(self, location: torch.Tensor, location_sd: torch.Tensor):
        """
        Read from the memory cache by keys.

        :param location: The location to read from the memory cache. Has shape (batch_size, query_dim).
        """
        if self.memory_locations is None:
            return torch.zeros(location.shape[0], self.sensory_dim, device=location.device, dtype=location.dtype)

        location_weights = self.get_location_affinity(location, location_sd)

        # pre_sense has shape (batch_size, embed_dim)
        pre_sense = torch.bmm(location_weights, self.memory_senses).squeeze(dim=-2)
        sense = self.sensory_read_proj(pre_sense)

        return sense

    def read_location_and_sensory(self, location: torch.Tensor, location_sd: torch.Tensor, sensory: torch.Tensor):
        """
        Read from the memory cache by location and sensory.
        """
        sensory = self.sensory_proj(sensory)
        location_affinity, sensory_affinity = self.get_location_and_sensory_affinity(location, location_sd, sensory)
        affinity = location_affinity * sensory_affinity
        scores = torch.softmax(affinity, dim=-1)
        location_out = torch.bmm(scores, self.memory_locations).squeeze(dim=-2)
        location_sd_out = torch.bmm(scores, self.memory_location_sds).squeeze(dim=-2)
        sensory_out = torch.bmm(scores, self.memory_senses).squeeze(dim=-2)
        sensory_out = self.sensory_read_proj(sensory_out)
        return location_out, location_sd_out, sensory_out


    def search(self, senses: torch.Tensor, num_results: int=1, threshold: float=0.1, diversity_steps: int=5, detach_locations: bool=False):
        """
        Search the senses for the most similar senses to queries, returning the closest matching keys.

        :param senses: The senses to search for. Has shape (batch_size, num_senses, sensory_dim).
        :param num_results: The number of results to return.
        :param threshold: The threshold for the match scores. Default is 2.0.
        :param diversity_steps: The number of steps to take to diversify the results. Default is 5.
        :param detach_locations: Whether to detach the memory locationparameters from grad calculations before computing the output. Default is False.
        :return: The top locations, the top senses, a boolean mask of whether each query was found, and the number of found queries.
        """
        if self.memory_senses is None:
            return (
                torch.zeros(self.batch_size, num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                torch.ones(self.batch_size, num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                torch.zeros(self.batch_size, num_results, self.sensory_dim, device=senses.device, dtype=senses.dtype),
                torch.zeros(self.batch_size, num_results, device=senses.device, dtype=torch.bool),
                torch.zeros(self.batch_size, device=senses.device, dtype=torch.long)
            )

        senses = self.sensory_proj(senses)

        # scores has shape (batch_size, num_keys) or (batch_size, num_queries, num_keys)
        memory_senses = self.memory_senses
        scores = self.score(senses, memory_senses, factor=self.sensory_factor)

        # now do attention to institute score competition among nearby locations
        if diversity_steps > 0:
            # neighbor weights over memory locations (B, K, K)
            memory_locations = self.memory_locations
            memory_location_sds = self.memory_location_sds
            if detach_locations:
                memory_locations = memory_locations.detach()
                memory_location_sds = memory_location_sds.detach()
            location_weights = self.get_location_affinity(memory_locations, memory_location_sds, mask_diagonal=True)

            if scores.ndim < location_weights.ndim:
                s = scores.unqueeze(-2)
            else:
                s = scores

            s = scores
            for _ in range(diversity_steps):
                neighbor_mass = torch.bmm(s, location_weights.transpose(-2, -1)) # (B, Q, K)
                s = (s - neighbor_mass).clamp_min(0)
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-12)                 # renorm

            if s.ndim > scores.ndim:
                s = s.squeeze(-2)

            scores = s

        # now scores are filered to refer to different locations, so we can pict the top results
        num_results = min(num_results, scores.shape[1])
        top_scores, top_indices = scores.topk(num_results, dim=-1, largest=True, sorted=True)
        top_location_indices = top_indices[..., None].expand(-1, -1, self.location_dim)
        top_sensory_indices = top_indices[..., None].expand(-1, -1, self.embed_dim)
        top_locations = self.memory_locations.gather(dim=1, index=top_location_indices)
        top_location_sds = self.memory_location_sds.gather(dim=1, index=top_location_indices)
        top_senses = self.memory_senses.gather(dim=1, index=top_sensory_indices)

        found = top_scores > threshold
        num_found = found.long().sum(dim=-1)

        top_senses = self.sensory_read_proj(top_senses)

        return top_locations, top_location_sds, top_senses, found, num_found

    def write(self, location: torch.Tensor, location_sd: torch.Tensor, sense: torch.Tensor):
        """
        Write to the memory cache. 

        """
        sense = self.sensory_proj(sense)

        if self.memory_locations is None:
            self.memory_locations = location[:, None, :]
            self.memory_location_sds = location_sd[:, None, :]
            self.memory_senses = sense[:, None, :]
        else:
            self.memory_locations = torch.cat([self.memory_locations, location[:, None, :]], dim=-2)
            self.memory_location_sds = torch.cat([self.memory_location_sds, location_sd[:, None, :]], dim=-2)
            self.memory_senses = torch.cat([self.memory_senses, sense[:, None, :]], dim=-2)
    
