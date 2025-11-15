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
    def __init__(self, location_dim: int, sensory_dim: int, embed_dim: int, batch_size: int=1, max_memory_size: int=-1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        if embed_dim == sensory_dim:
            self.sensory_proj = torch.nn.Identity()
            self.sensory_read_proj = torch.nn.Identity()
        else:
            self.sensory_proj = torch.nn.Linear(sensory_dim, embed_dim, bias=False)
            self.sensory_read_proj = torch.nn.Linear(embed_dim, sensory_dim, bias=False)

            self.sensory_proj.weight.data /= torch.linalg.matrix_norm(self.sensory_proj.weight.data)
            self.sensory_read_proj.weight.data = torch.pinverse(self.sensory_proj.weight.data)

        self.memory_locations = None
        self.memory_senses = None
        self.memory_location_sds = None
        self.invalid_slots = None

        self.sensory_factor = sensory_dim ** -0.5
        self.location_factor = location_dim ** -0.5

    def reset(self):
        self.memory_locations = None
        self.memory_senses = None
        self.memory_location_sds = None
        self.invalid_slots = None

    def break_training_graph(self):
        if self.memory_locations is not None:
            self.memory_locations = self.memory_locations.detach()
        if self.memory_senses is not None:
            self.memory_senses = self.memory_senses.detach()
        if self.memory_location_sds is not None:
            self.memory_location_sds = self.memory_location_sds.detach()
        if self.invalid_slots is not None:
            self.invalid_slots = self.invalid_slots.detach()

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

    def get_location_affinity(self, location: torch.Tensor, location_sd: torch.Tensor, 
                              match_threshold: float=None, mask_diagonal: bool=False, raw_weights: bool=False, 
                              detach_locations: bool=True):
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
        memory_locations = self.memory_locations
        memory_location_sds = self.memory_location_sds
        if detach_locations:
            memory_locations = memory_locations.detach()
            memory_location_sds = memory_location_sds.detach()

        location_delta = location[..., None, :] - memory_locations[..., None, :, :]
        location_delta_var = location_sd[..., None, :]**2 + memory_location_sds[..., None, :, :]**2

        log_location_affinity = (
            - 0.5 * (location_delta.pow(2) / (location_delta_var + 1e-8)).pow(2).sum(dim=-1) 
            - 0.5 * math.log(2 * math.pi) * location_delta_var.shape[-1]
            - torch.log(location_delta_var).sum(dim=-1)
        )

        if match_threshold is not None:
            threshold_check = torch.norm(location_delta, dim=-1) > match_threshold
            log_location_affinity = log_location_affinity.masked_fill(threshold_check, float('-inf'))

        if mask_diagonal:
            # TODO: this will fail if num_queries != num_keys or if there is more than one batch dimension
            eye = torch.eye(log_location_affinity.size(-2), device=log_location_affinity.device).bool()
            log_location_affinity = log_location_affinity.masked_fill(eye.unsqueeze(0), float('-inf'))

        if raw_weights:
            return log_location_affinity

        # shape (batch_size, num_queries, num_keys)
        inactive_mask = (log_location_affinity <= float('-inf')).all(dim=-1, keepdim=True)
        print(f"Inactive mask: {inactive_mask.squeeze().cpu().numpy().tolist()}")
        print(f"location_delta.min(dim=-1): {torch.norm(location_delta, dim=-1).min(dim=-1).values.squeeze().cpu().numpy().tolist()}")
        location_weights = torch.softmax(log_location_affinity, dim=-1)
        location_weights = location_weights.masked_fill(inactive_mask, 0.0)

        # location_weight_mean = location_weights.mean(dim=-1, keepdim=True)
        # location_weights = location_weights.masked_fill(location_weights < location_weight_mean, 0.0)
        # location_weights = location_weights / (location_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return location_weights
    
    def get_location_and_sensory_affinity(self, location: torch.Tensor, location_sd: torch.Tensor, sensory: torch.Tensor,
                                          detach_senses: bool=True, detach_locations: bool=True):
        """
        Get the affinity of a location and sensory to the memory locations and senses.
        """
        location_affinity = self.get_location_affinity(location, location_sd, raw_weights=True, detach_locations=detach_locations)
        memory_senses = self.memory_senses
        if detach_senses:
            memory_senses = memory_senses.detach()
        
        sensory_affinity = self.score(sensory, memory_senses, factor=self.sensory_factor)
        return location_affinity, sensory_affinity

    def read(self, location: torch.Tensor, location_sd: torch.Tensor, match_threshold: float=None,
             detach_senses: bool=True, detach_locations: bool=True):
        """
        Read from the memory cache by keys.

        :param location: The location to read from the memory cache. Has shape (batch_size, query_dim).
        """
        if self.memory_locations is None:
            if location.ndim < 3:
                return torch.zeros(location.shape[0], self.sensory_dim, device=location.device, dtype=location.dtype)
            else:
                return torch.zeros(location.shape[0], location.shape[1], self.sensory_dim, device=location.device, dtype=location.dtype)
                
        squeeze = False
        if location.ndim < 3:
            squeeze = True
            location_sd = location_sd[..., None, :]
        
        if location_sd.ndim < 3:
            location_sd = location_sd[..., None, :]

        # shape (batch_size, num_queries, num_keys)
        location_weights = self.get_location_affinity(location, location_sd, match_threshold=match_threshold, 
                                                      detach_locations=detach_locations)

        # pre_sense has shape (batch_size, num_queries, embed_dim)
        # memory_senses has shape (batch_size, num_keys, embed_dim)
        memory_senses = self.memory_senses
        if detach_senses:
            memory_senses = memory_senses.detach()
        pre_sense = torch.bmm(location_weights, memory_senses)

        if squeeze:
            pre_sense = pre_sense.squeeze(-2)

        sense = self.sensory_read_proj(pre_sense)

        return sense

    def read_location_and_sensory(self, location: torch.Tensor, location_sd: torch.Tensor, sensory: torch.Tensor,
                                  skip_projection: bool=False, detach_senses: bool=True, detach_locations: bool=True):
        """
        Read from the memory cache by location and sensory.
        """
        if not skip_projection:
            sensory = self.sensory_proj(sensory)

        location_affinity, sensory_affinity = self.get_location_and_sensory_affinity(
            location, location_sd, sensory, detach_senses=detach_senses, detach_locations=detach_locations
        )
        affinity = location_affinity * sensory_affinity
        scores = torch.softmax(affinity, dim=-1)

        memory_locations = self.memory_locations
        memory_location_sds = self.memory_location_sds
        if detach_locations:
            memory_locations = memory_locations.detach()
            memory_location_sds = memory_location_sds.detach()

        location_out = torch.bmm(scores, memory_locations).squeeze(dim=-2)
        location_sd_out = torch.bmm(scores, memory_location_sds).squeeze(dim=-2)

        memory_senses = self.memory_senses
        if detach_senses:
            memory_senses = memory_senses.detach()

        sensory_out = torch.bmm(scores, memory_senses).squeeze(dim=-2)
        if not skip_projection:
            sensory_out = self.sensory_read_proj(sensory_out)

        return location_out, location_sd_out, sensory_out

    def sample(self, location: torch.Tensor, location_sd: torch.Tensor, search_key: torch.Tensor, 
               num_samples: int=1, temperature: float=1.0, sigma_scale: float=25.0,
               detach_locations: bool=True, detach_senses: bool=True):
        """
        Sample from the memory cache using a Gaussian mixture model.

        :param location: The location to sample from. Has shape (batch_size, location_dim).
        :param location_sd: The standard deviation of the location. Has shape (batch_size, location_dim).
        :param num_samples: The number of samples to return.
        :param sigma_scale: The scale of the Gaussian distribution.
        :param temperature: The temperature of the softmax.
        :return: The sampled locations, the sampled senses, and the sampled weights.
        """
        if self.memory_locations is None:
            return torch.zeros(self.batch_size, num_samples, self.location_dim, device=location.device, dtype=location.dtype)

        search_key = self.sensory_proj(search_key)
        
        if location is None:
            affinity = self.score(search_key, self.memory_senses, factor=self.sensory_factor)

        else:
            if location.ndim < 3:
                location = location[..., None, :]
                location_sd = location_sd[..., None, :]

            if search_key.ndim < 3:
                search_key = search_key[..., None, :]

            location_affinity, sensory_affinity = self.get_location_and_sensory_affinity(
                location, location_sd, search_key, detach_senses=detach_senses, detach_locations=detach_locations
            )
            affinity = location_affinity * sensory_affinity
        
        scores = torch.softmax(temperature * affinity, dim=-1)

        shape = tuple(list(scores.shape[:-1]) + [num_samples,])
        t = torch.multinomial(scores.view(-1, scores.shape[-1]), num_samples=num_samples, replacement=True)
        t = t.view(*shape)
        t = t.unsqueeze(-1).repeat(1, 1, 1, self.memory_locations.shape[-1]).view(self.batch_size, -1, self.location_dim)
        loc_mean = self.memory_locations.gather(dim=-2, index=t).view(self.batch_size, -1, num_samples, self.location_dim)
        loc_sd = self.memory_location_sds.gather(dim=-2, index=t).view(self.batch_size, -1, num_samples, self.location_dim)

        return loc_mean + torch.randn_like(loc_mean) * sigma_scale * loc_sd


    def search(self, senses: torch.Tensor, num_results: int=1, threshold: float=0.0, diversity_steps: int=5,
               detach_locations: bool=True, detach_senses: bool=True):
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
            if senses.ndim < 3:
                return (
                    torch.zeros(self.batch_size, num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                    torch.ones(self.batch_size, num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                    torch.zeros(self.batch_size, num_results, self.sensory_dim, device=senses.device, dtype=senses.dtype),
                    torch.zeros(self.batch_size, num_results, device=senses.device, dtype=torch.bool),
                    torch.zeros(self.batch_size, device=senses.device, dtype=torch.long)
                )
            else:
                return (
                    torch.zeros(self.batch_size, senses.shape[1], num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                    torch.ones(self.batch_size, senses.shape[1], num_results, self.location_dim, device=senses.device, dtype=senses.dtype),
                    torch.zeros(self.batch_size, senses.shape[1], num_results, self.sensory_dim, device=senses.device, dtype=senses.dtype),
                    torch.zeros(self.batch_size, senses.shape[1], num_results, device=senses.device, dtype=torch.bool),
                    torch.zeros(self.batch_size, senses.shape[1], device=senses.device, dtype=torch.long)
                )

        senses = self.sensory_proj(senses)

        squeeze = False
        if senses.ndim < 3:
            senses = senses.unsqueeze(-2)
            squeeze = True

        # scores has shape (batch_size, num_keys) or (batch_size, num_queries, num_keys)
        memory_senses = self.memory_senses
        if detach_senses:
            memory_senses = memory_senses.detach()
        
        scores = self.score(senses, memory_senses, factor=self.sensory_factor)

        memory_locations = self.memory_locations
        memory_location_sds = self.memory_location_sds
        if detach_locations:
            memory_locations = memory_locations.detach()
            memory_location_sds = memory_location_sds.detach()

        # now do attention to institute score competition among nearby locations
        if diversity_steps > 0:
            # neighbor weights over memory locations (B, K, K)
            location_weights = self.get_location_affinity(memory_locations, memory_location_sds, mask_diagonal=True, detach_locations=detach_locations)

            s = scores

            for _ in range(diversity_steps):
                neighbor_mass = torch.bmm(s, location_weights.transpose(-2, -1)) # (B, Q, K)
                s = (s - neighbor_mass).clamp_min(0)
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-12)                 # renorm

            scores = s

        # now scores are filered to refer to different locations, so we can pict the top results
        num_results = min(num_results, scores.shape[-1])
        top_scores, top_indices = scores.topk(num_results, dim=-1, largest=True, sorted=True)

        # now scores is (batch_size, num_queries, num_results) --> (batch_size, num_queries * num_results)
        top_scores = top_scores.view(top_scores.shape[0], -1)
        top_indices = top_indices.view(top_indices.shape[0], -1)

        top_location_indices = top_indices[..., None].expand(-1, -1, self.location_dim)
        top_sensory_indices = top_indices[..., None].expand(-1, -1, self.embed_dim)
        top_locations = memory_locations.gather(dim=1, index=top_location_indices)
        top_location_sds = memory_location_sds.gather(dim=1, index=top_location_indices)
        top_senses = memory_senses.gather(dim=1, index=top_sensory_indices)

        # unfold to (batch_size, num_queries, num_results, location_dim)
        if not squeeze:
            top_scores = top_scores.view(top_scores.shape[0], -1, num_results)
            top_locations = top_locations.view(top_locations.shape[0], -1, num_results, self.location_dim)
            top_location_sds = top_location_sds.view(top_location_sds.shape[0], -1, num_results, self.location_dim)
            top_senses = top_senses.view(top_senses.shape[0], -1, num_results, self.embed_dim)

        found = top_scores > threshold
        num_found = found.long().sum(dim=-1)

        top_senses = self.sensory_read_proj(top_senses)

        return top_locations, top_location_sds, top_senses, found, num_found

    def write(self, location: torch.Tensor, location_sd: torch.Tensor, sense: torch.Tensor, update: bool=False):
        """
        Write to the memory cache. 

        """
        if (self.memory_locations is None and location.ndim < 3) or (location.ndim < self.memory_locations.ndim):
            location = location[None, ...]
            location_sd = location_sd[None, ...]
            sense = sense[None, ..., :]

        if self.memory_locations is None:
            assert location.ndim == location_sd.ndim == sense.ndim == 3
        else:
            assert location.ndim == self.memory_locations.ndim == location_sd.ndim == sense.ndim == self.memory_senses.ndim
        
        sense = self.sensory_proj(sense)

        if self.memory_locations is None:
            self.memory_locations = location
            self.memory_location_sds = location_sd
            self.memory_senses = sense

            return
        
        if not update:
            self.memory_locations = torch.cat([self.memory_locations, location], dim=-2)
            self.memory_location_sds = torch.cat([self.memory_location_sds, location_sd], dim=-2)
            self.memory_senses = torch.cat([self.memory_senses, sense], dim=-2)

            self.truncate()
            return

        # check to see if the new location matches an existing location within 1 standard deviation
        location_delta = location[..., None, :] - self.memory_locations
        location_delta_var = location_sd[..., None, :]**2 + self.memory_location_sds**2
        match_location = torch.norm(location_delta / torch.clamp(location_delta_var.sqrt(), min=1e-8), dim=-1) < 1.0

        sensory_delta = sense[..., None, :] - self.memory_senses
        match_sensory = sensory_delta.norm(dim=-1) < 0.05 # * torch.linalg.matrix_norm(self.sensory_proj.weight)

        match = match_location | match_sensory

        if match.any():
            new_variance = location_sd[..., None, :].pow(2) * self.memory_location_sds.pow(2) / (
                torch.clamp(location_sd[..., None, :].pow(2) + self.memory_location_sds.pow(2), min=1e-8)
            )
            optimal_location = (
                location[..., None, :] / torch.clamp(location_sd[..., None, :].pow(2), min=1e-8)
                + self.memory_locations / torch.clamp(self.memory_location_sds.pow(2), min=1e-8)
            ) * new_variance
            optimal_sd = new_variance.sqrt()

            optimal_sense = (
                sense[..., None, :] / torch.clamp(torch.norm(location_sd[..., None, :], dim=-1, keepdim=True).pow(2), min=1e-8)
                + self.memory_senses / torch.clamp(torch.norm(self.memory_location_sds, dim=-1, keepdim=True).pow(2), min=1e-8) 
            ) / (
                1 / torch.clamp(torch.norm(location_sd[..., None, :], dim=-1, keepdim=True).pow(2), min=1e-8)
                + 1 / torch.clamp(torch.norm(self.memory_location_sds, dim=-1, keepdim=True).pow(2), min=1e-8)
            )

            self.memory_locations = torch.where(match[..., None], optimal_location, self.memory_locations)
            self.memory_location_sds = torch.where(match[..., None], optimal_sd, self.memory_location_sds)
            self.memory_senses = torch.where(match[..., None], optimal_sense, self.memory_senses)

            # now, to handle the rows that didn't match, we need to add them to the memory
            used = match.any(dim=-1)
            if (~used).any():
                locations = torch.where(~used, location, torch.zeros_like(location))
                location_sds = torch.where(~used, location_sd, torch.ones_like(location_sd))
                sense = torch.where(~used, sense, torch.zeros_like(sense))
                self.memory_locations = torch.cat([self.memory_locations, locations[:, None, :]], dim=-2)
                self.memory_location_sds = torch.cat([self.memory_location_sds, location_sds[:, None, :]], dim=-2)
                self.memory_senses = torch.cat([self.memory_senses, sense[:, None, :]], dim=-2)

                if self.invalid_slots is None:
                    self.invalid_slots = torch.cat([
                        torch.zeros(self.batch_size, match.shape[-1], device=location.device, dtype=torch.bool),
                        used.unsqueeze(-1),
                    ], dim=-1)
                else:
                    num_invalid = self.invalid_slots.long().sum(dim=-1)
                    num_to_remove = num_invalid.min()
                    if num_to_remove >= 1:
                        # Level the invalid slots to remove
                        num_after_remove = self.memory_locations.shape[1] - num_to_remove

                        # first, index the invalid slots
                        invalid_indexed = self.invalid_slots.long().cumsum(dim=-1)

                        # eliminate slots that have indices larger than what we will remove
                        slots_to_remove = (invalid_indexed <= num_to_remove) & (invalid_indexed > 0)
                        slots_to_keep = ~slots_to_remove[..., None]
                        view_tuple = (self.batch_size, num_after_remove, -1)

                        self.memory_locations = torch.masked_select(self.memory_locations, slots_to_keep).view(*view_tuple)
                        self.memory_location_sds = torch.masked_select(self.memory_location_sds, slots_to_keep).view(*view_tuple)
                        self.memory_senses = torch.masked_select(self.memory_senses, slots_to_keep).view(*view_tuple)
                        self.invalid_slots = torch.masked_select(self.invalid_slots, slots_to_keep).view(self.batch_size, -1)
        
        else:

            self.memory_locations = torch.cat([self.memory_locations, location[:, None, :]], dim=-2)
            self.memory_location_sds = torch.cat([self.memory_location_sds, location_sd[:, None, :]], dim=-2)
            self.memory_senses = torch.cat([self.memory_senses, sense[:, None, :]], dim=-2)
        
        self.truncate()

    def truncate(self):
        if self.max_memory_size < 0:
            return
        
        # TODO: prune more intelligently
        if self.memory_locations is not None:
            if self.memory_locations.shape[1] > self.max_memory_size:
                self.memory_locations = self.memory_locations[:, -self.max_memory_size:]
                self.memory_location_sds = self.memory_location_sds[:, -self.max_memory_size:]
                self.memory_senses = self.memory_senses[:, -self.max_memory_size:]
                if self.invalid_slots is not None:
                    self.invalid_slots = self.invalid_slots[:, -self.max_memory_size:]

    def interpret(self, location_model: torch.nn.Module, drive_classifier: torch.nn.Module, steps: int=10):
        """
        Interpret the memory locations using a location model.
        """
        if self.memory_locations is None:
            return None, None

        # first, read the memory up to a fixed point
        memory_locations = self.memory_locations.detach()
        memory_location_sds = self.memory_location_sds.detach()
        memory_senses = self.memory_senses.detach()

        for i in range(steps):
            memory_locations, memory_location_sds, memory_senses = self.read_location_and_sensory(
                memory_locations, memory_location_sds, memory_senses, skip_projection=True
            )

            # TODO: prune locations that are close to each other to reduce the number of locations to interpret

        memory_location_projected = location_model(memory_locations)
        memory_senses = self.sensory_read_proj(memory_senses)

        # will generate a softmax over "poison", "edible", and "neutral" drives
        # so we need to take the argmax to get the drive target
        drive_targets = torch.argmax(drive_classifier(memory_senses), dim=-1)

        return memory_location_projected, drive_targets

    def refine(self, location: torch.Tensor, location_sd: torch.Tensor, drive_classifier: torch.nn.Module,
               steps: int=100, eta: float=0.01):
        """
        Refine the memory locations using a drive classifier.
        """

        sense = self.read(location, location_sd, detach_senses=False, detach_locations=False)
        drive_targets = drive_classifier(sense)
        drive_to_search = drive_targets.argmax(dim=-1)

        print(f"Initial drive targets: {drive_targets[0].detach().cpu().numpy().tolist()}")

        for i in range(steps):
            
            dloc, = torch.autograd.grad(drive_targets[..., drive_to_search], (location,), 
                                            create_graph=True, retain_graph=True)
            
            location = location + eta * dloc
            
            sense = self.read(location, location_sd, detach_senses=False, detach_locations=False)
            drive_targets = drive_classifier(sense)
        
            print(f"{i}: Drive targets: {drive_targets[0].detach().cpu().numpy().tolist()}")

        return location, location_sd, sense


    def regularize(self, loss):
        sensory_mat = self.sensory_read_proj.weight @ self.sensory_proj.weight

        sensory_loss = torch.linalg.norm(sensory_mat - torch.eye(self.sensory_dim, device=sensory_mat.device))
        return loss + sensory_loss
    
