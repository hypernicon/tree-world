import torch
import math
from typing import Optional

from .metric import PseudoMetric


class SpatialMemory(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, embed_dim: int, batch_size: int=1, max_memory_size: int=-1, 
                 physical_dim: Optional[int]=None, physical_scale: float=10.0, physical_ratio: float=math.sqrt(2.0),
                 location_metric_rank: Optional[int]=None):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        if embed_dim != sensory_dim:
            self.sensory_proj = torch.nn.Linear(sensory_dim, embed_dim, bias=False)
            self.sensory_read_proj = torch.nn.Linear(embed_dim, sensory_dim, bias=False)
        else:
            self.sensory_proj = torch.nn.Identity()
            self.sensory_read_proj = torch.nn.Identity()

        self.memory_locations = None
        self.memory_values = None

        if location_metric_rank is None:
            location_metric_rank = embed_dim
        self.location_metric = PseudoMetric(location_dim, physical_dim, physical_scale, physical_ratio, metric_rank=location_metric_rank)

    def reset(self):
        self.memory_locations = None
        self.memory_values = None

    def break_training_graph(self):
        self.memory_locations = self.memory_locations.detach()
        self.memory_values = self.memory_values.detach()

    def memory_size(self):
        if self.memory_locations is None:
            return 0
        return self.memory_locations.shape[1]

    def write(
        self,
        location_codes: torch.Tensor,
        sensory_data: torch.Tensor,
    ) -> None:
        # memory_locations: (B, T, E)
        # memory_values: (B, T, E)
        # location_beliefs: (B, L) or (B, T, L)  # N is the number of memory locations
        # sensory_data: (B, D)
        # return (B, D) or (B, T, D)
        assert sensory_data.ndim == location_codes.ndim, "sensory_data and location_codes must have a compatible number of dimensions"
        
        if location_codes.ndim == 3:
            location_codes = location_codes.unsqueeze(1)
            sensory_data = sensory_data.unsqueeze(1)

        sensory_data = self.sensory_proj(sensory_data)
        location_codes = self.location_metric.prepare_k(location_codes)

        if self.memory_locations is None or self.memory_values is None:
            assert sensory_data.ndim == 3
            assert location_codes.ndim == 3
            self.memory_locations = location_codes.clone()
            self.memory_values = sensory_data.clone()
            return

        assert sensory_data.ndim == self.memory_values.ndim == 3  
        assert self.memory_locations.ndim == location_codes.ndim == 3
         
        self.memory_locations = torch.cat([self.memory_locations, location_codes], dim=1)
        self.memory_values = torch.cat([self.memory_values, sensory_data], dim=1)
        
    def read(
        self,
        location_codes: torch.Tensor,
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
        if location_codes.ndim == 3:
            single_query = True
            location_beliefs = location_codes.unsqueeze(1)

        if self.memory_locations is None or self.memory_values is None:
            result = torch.zeros(
                location_codes.shape[0], location_codes.shape[1], self.sensory_dim, 
                device=location_beliefs.device, dtype=location_codes.dtype
            )
            if single_query:
                return result.squeeze(1)
            
            if return_weights:
                return result, None
            else:
                return result

        batch_size, time_steps, E = self.memory_locations.shape

        if match_threshold < 0:
            # TODO: how to set match threshold? 
            match_threshold = 0.25

        num_queries = location_codes.shape[1]

        # compute the location affinity as (B, Q, T)
        # we use log here because the location affinity is a probability distribution, so the affinity is between 0 and 1
        location_affinity = self.location_metric.affinity(location_codes, self.memory_locations, prepared_k=True)

        if match_threshold is not None:
            assert 0.0 < match_threshold, "match_threshold must be greater than 0 and less than 1"
            location_affinity = location_affinity.masked_fill(location_affinity < math.log(match_threshold), float('-inf'))

        if mask_diagonal:
            assert time_steps == num_queries, "mask_diagonal is only supported when the number of query locations is equal to the number of memory locations"
            diagonal_mask = torch.eye(time_steps, device=location_affinity.device, dtype=torch.bool).unsqueeze(0)
            location_affinity = location_affinity.masked_fill(diagonal_mask, float('-inf'))

        # compute the attention weights as (B, Q, T)
        # we use log here because the location affinity is a probability distribution, so the affinity is between 0 and 1
        attention_weights = torch.softmax(location_affinity, dim=-1)

        if zero_invalid:
            invalid = location_affinity.sum(dim=-1, keepdim=True) <= 0
            attention_weights = attention_weights.masked_fill(invalid, 0.0)

        # compute the new memory values as (B, Q, D)
        new_memory_values = torch.bmm(attention_weights, self.memory_values)

        new_memory_values = self.sensory_read_proj(new_memory_values)

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
        reference_match_threshold: Optional[float]=None,
        aggregate: bool=False,
    ) -> Optional[torch.Tensor]:
        # memory_locations has shape (N, T, S, S)
        # memory_values has shape (N, T, D)
        # search_key has shape (N, D)
        if self.memory_locations is None or self.memory_values is None:
            return None

        N, T, _ = self.memory_locations.shape

        if temperature is None:
            temperature = self.memory_values.shape[-1]**(0.5)

        search_key = self.sensory_proj(search_key)

        # compute the alignment scores (N, T)
        s_t = torch.bmm(self.memory_values, search_key[..., None]).squeeze(-1)

        if reference_location is not None:
            location_affinity = self.location_metric.affinity(reference_location.unsqueeze(1), self.memory_locations, prepared_k=True)

            if reference_match_threshold is not None:
                invalid_mask = location_affinity < reference_match_threshold
                location_affinity = location_affinity.masked_fill(invalid_mask, float('-inf'))

            location_max = location_affinity.max(dim=-1, keepdim=True).values
            invalid_mask = invalid_mask | (location_max < 1e-8)
            location_affinity = location_affinity / location_max
                
            s_t = torch.where(invalid_mask, torch.zeros_like(s_t), s_t * location_affinity)

        w_t = torch.softmax(s_t / temperature, dim=-1)

        # sample from the mixture, result will be (N, num_samples)
        t = torch.multinomial(torch.nan_to_num(w_t, nan=0.0), num_samples=num_samples, replacement=True)
        
        invalid_ts = None
        if invalid_mask is not None:
            invalid_ts = invalid_mask.gather(dim=-1, index=t)

        t = t.unsqueeze(-1).repeat(1, 1, D)    # (N, num_samples, embed_dim)

        loc = self.memory_locations.gather(dim=-2, index=t)

        if invalid_ts is not None:
            loc = torch.where(invalid_ts[..., None], torch.zeros_like(loc), loc)

        if aggregate:
            return loc.mean(dim=-2).view(N, -1)
        else:
            return loc.view(N, -1, self.location_dim)

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
        N, T, L = self.memory_locations.shape
        _, _, D = self.memory_values.shape

        if match_threshold is None:
            match_threshold = 0.25

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

        pruned_indices_loc = pruned_indices[..., None].repeat(1, 1, L)
        pruned_indices_sense = pruned_indices[..., None].repeat(1, 1, D)

        self.memory_locations = memory_locations.gather(dim=-2, index=pruned_indices_loc).view(N, -1, L)
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
        
    