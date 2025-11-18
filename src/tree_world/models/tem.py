import torch
import math
from .memory import SpatialMemory

"""
Loosely based on the Tolman-Eichenbaum model of memory encoding and retrieval.

https://www.sciencedirect.com/science/article/pii/S009286742031388X
"""

def kl_divergence_gaussian(mean1, sd1, mean2, sd2, eps=1e-8):
    var1, var2 = sd1*sd1 + eps, sd2*sd2 + eps
    t = (var1 / var2) + (mean2 - mean1).pow(2) / var2 - 1.0 + torch.log(var2) - torch.log(var1)
    return 0.5 * t.sum(dim=-1).mean()


class TEMLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, action_dim: int, hidden_dim: int=128, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.localizer = torch.nn.Linear(action_dim + 1, location_dim, bias=False)
        self.localizer.weight.data /= math.sqrt(location_dim)

        if action_dim == location_dim:
            self.localizer.weight.data[:action_dim,:action_dim] = torch.eye(action_dim, device=self.localizer.weight.device)

        self.variance_proj = torch.nn.Sequential(
            torch.nn.Linear(location_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, location_dim),
        )

        self.min_sd = 0.1

        self.variance_scale_logits = torch.nn.Parameter(torch.ones(location_dim))

    def forward(self, location: torch.Tensor, action: torch.Tensor, scale: torch.Tensor=None, return_distribution: bool=False, force_location: torch.Tensor=None):

        if scale is None:
            scale = torch.ones(location.shape[0], 1, device=location.device, dtype=location.dtype)

        location_mean = location + self.localizer(torch.cat([action, scale], dim=1))
        location_sd = torch.nn.functional.softplus(self.variance_proj(location_mean - location))
        location_sd = self.min_sd + torch.nn.functional.softplus(self.variance_scale_logits) * location_sd

        if return_distribution:
            return location_mean, location_sd

        if force_location is not None:
            next_location = force_location
        else:
            next_location = location_mean # + torch.randn_like(location_mean) * location_sd

        if torch.isnan(next_location).any():
            if force_location is not None:
                print(f"Force location: {force_location[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Next location: {next_location[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Location mean: {location_mean[0,:4].detach().cpu().numpy().tolist()}, Location sd: {location_sd[0,:4].detach().cpu().numpy().tolist()}")
            print(f"self.variance_scale_logits: {self.variance_scale_logits.detach().cpu().numpy().tolist()}")
            raise RuntimeError("Next location is nan")

        # print(f"Distance traveled in location space: {torch.norm(next_location - location):.3f} vs weight norm {torch.linalg.matrix_norm(self.localizer.weight):.3f} based on action of norm {torch.norm(action):.3f}")

        return next_location, location_mean, location_sd

    def regularize(self, loss):
        # penalize the localizer weight from being too small
        return loss + torch.relu(1.0 - torch.linalg.matrix_norm(self.localizer.weight))


class SimpleTEMDecoder(torch.nn.Module):

    def __init__(self, memory: SpatialMemory, localizer: TEMLocalizer, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.memory = memory
        self.localizer = localizer

        self.min_sd = 0.1

        # accuracy of sensory read depends on how far away the object is
        self.sensory_variance_proj = torch.nn.Sequential(
            torch.nn.Linear(location_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, sensory_dim),
        )
        self.sensory_variance_logits = torch.nn.Parameter(torch.ones(sensory_dim))

    def reset(self):
        self.memory.reset()

    def forward(self, last_location: torch.Tensor, action: torch.Tensor, self_location: torch.Tensor=None, actual_sensory: torch.Tensor=None):
        """
        Forward pass for the TEM decoder.

        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :param actual_sensory: the actual sensory reading to write to memory
        :return: The next location, the next sensory reading, the log probability of the location and the log probability of the sensory reading.
        """

        next_location, location_mean, location_sd = self.localizer(last_location, action, force_location=self_location)

        sensory_mean = self.memory.read(next_location, location_sd)

        sensory_sd = torch.nn.functional.softplus(self.sensory_variance_proj(next_location))
        sensory_sd = self.min_sd + torch.nn.functional.softplus(self.sensory_variance_logits) * sensory_sd
        
        sensory = sensory_mean + torch.randn_like(sensory_mean) * sensory_sd

        if actual_sensory is None:
            actual_sensory = sensory

        log_prob_sensory = -0.5 * ((actual_sensory - sensory_mean) / sensory_sd).pow(2).sum(dim=-1).mean() - torch.log(sensory_sd).sum(dim=-1).mean() - 0.5 * math.log(2 * math.pi) * sensory_sd.shape[-1]

        return next_location, sensory, log_prob_sensory, location_mean, location_sd

    def break_training_graph(self):
        self.memory.break_training_graph()


class SimpleTEMEncoder(torch.nn.Module):
    def __init__(self, memory: SpatialMemory, localizer: TEMLocalizer, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_guesses: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_guesses = num_guesses
        self.dropout = dropout
        self.memory = memory
        self.localizer = localizer
        self.factor = location_dim ** -0.5

    def reset(self):
        self.memory.reset()

    def select_location(self, location_guesses: torch.Tensor, location_guess_sds: torch.Tensor, expected_location_mean: torch.Tensor, expected_location_sd: torch.Tensor,
                        invalid_guess_mask: torch.BoolTensor, ignore_guesses: torch.BoolTensor):
        """
        Choose the next location from the location guesses, or abandon the guesses and use the expected location.

        :param location_guesses: The location guesses. Has shape (batch_size, num_guesses, location_dim).
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action. Has shape (batch_size, action_dim).
        :param invalid_guess_mask: A mask of true for invalid guesses. Has shape (batch_size, num_guesses).
        :param ignore_guesses: A mask of true for the guesses to ignore. Has shape (batch_size,).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """

        # expected_location_mean has shape (batch_size, location_dim)
        # location_guesses has shape (batch_size, num_guesses, location_dim)
        location_delta = location_guesses - expected_location_mean[..., None, :]

        # matches has shape (batch_size, num_guesses)
        matches = (
            - 0.5 * (location_delta / (location_guess_sds + expected_location_sd[..., None, :] + 1e-6)).pow(2).sum(dim=-1)
            - 0.5 * math.log(2 * math.pi) * expected_location_sd.shape[-1]
            - torch.log(expected_location_sd[..., None, :]).sum(dim=-1)
        )
        matches = matches.masked_fill(invalid_guess_mask, -float('inf'))
        matches = matches.softmax(dim=-1)

        integrated_location = torch.bmm(matches[..., None, :], location_guesses).squeeze(dim=-2)
        
        # check if the integrated location is two standard deviations away from the expected location
        too_far = torch.norm(((integrated_location - expected_location_mean) / expected_location_sd), dim=-1) > 2.0

        expected_location = expected_location_mean + torch.randn_like(expected_location_mean) * expected_location_sd

        too_far = too_far[..., None]
        ignore_guesses = ignore_guesses[..., None]
        next_location = torch.where(too_far | torch.isnan(integrated_location) | ignore_guesses, expected_location, integrated_location)

        if torch.isnan(next_location).any():
            print(f"Integrated location: {integrated_location[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Expected location mean: {expected_location_mean[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Expected location sd: {expected_location_sd[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Matches: {matches[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Invalid guess mask: {invalid_guess_mask[0,:4].detach().cpu().numpy().tolist()}")
            print(f"Ignore guesses: {ignore_guesses[0].detach().cpu().numpy().tolist()}")
            raise RuntimeError("Integrated location is nan")

        return next_location
        
    def forward(self, sensory: torch.Tensor, last_location: torch.Tensor, action: torch.Tensor):
        """
        Forward pass for the TEM encoder.

        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        next_location, next_location_mean, next_location_sd = self.localizer(last_location, action)

        location_guesses, location_sds, location_senses, found, num_found = self.memory.search(
            sensory, num_results=self.num_guesses
        )
        num_found = num_found.view(-1)  # should be (batch_size, 1) --> (batch_size,)

        invalid_guess_mask = torch.arange(location_guesses.shape[1], device=location_guesses.device)[None, :] >= num_found[:, None]
        ignore_guesses = num_found == 0

        next_location = self.select_location(location_guesses, location_sds, next_location_mean, next_location_sd, 
                                                          invalid_guess_mask, ignore_guesses)

        return next_location, next_location_mean, next_location_sd

    def get_expected_sensory(self, location: torch.Tensor, location_sd: torch.Tensor):
        """
        Get the expected sensory for a location.
        """
        return self.memory.read(location, location_sd)

    def break_training_graph(self):
        self.memory.break_training_graph()
        

class SimpleTEMModel(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_guesses: int, dropout: float=0.1, batch_size: int=1, is_simple: bool=True):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.num_guesses = num_guesses
        self.dropout = dropout

        self.memory = SpatialMemory(location_dim, sensory_dim, embed_dim, batch_size)
        self.localizer = TEMLocalizer(location_dim, action_dim, embed_dim, dropout)
        
        if is_simple:
            self.encoder = SimpleTEMEncoder(self.memory, self.localizer, location_dim, sensory_dim, action_dim, embed_dim, num_guesses, dropout)
            self.decoder = SimpleTEMDecoder(self.memory, self.localizer, location_dim, sensory_dim, action_dim, embed_dim, dropout)

    def forward(self, sensory: torch.Tensor, action: torch.Tensor, last_location: torch.Tensor, force_location: torch.Tensor=None):
        """
        Forward pass for the TEM model.

        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :param action: The action to take. Has shape (batch_size, action_dim). Unit vector.
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim). Unit vector.
        :param distance: The distance to the object. Has shape (batch_size,).
        :param force_location: The location to force the model to use. Has shape (batch_size, location_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        if last_location is None:
            last_location = torch.zeros(sensory.shape[0], self.location_dim, device=sensory.device)

        if action is None:
            action = torch.zeros(sensory.shape[0], self.action_dim, device=sensory.device)

        next_location, next_location_mean, next_location_sd = self.encoder(
            sensory, last_location, action
        )
        next_location_gen, sensory_gen, log_prob_sensory_gen, loc_mean_gen, loc_sd_gen  = self.decoder(
            last_location, action, self_location=next_location, actual_sensory=sensory
        )

        return_location = loc_mean_gen if force_location is None else force_location

        kl_next = kl_divergence_gaussian(next_location_mean, next_location_sd, loc_mean_gen, loc_sd_gen)
        self.memory.write(return_location, next_location_sd, sensory)

        elbo = log_prob_sensory_gen - 10000 * kl_next

        return return_location, loc_sd_gen, elbo, log_prob_sensory_gen, kl_next

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()

    def search_for_target_location(self, target_prototype: torch.Tensor, num_results: int=1, diversity_steps: int=5):
        """
        Given a prototype of the desired sensory percept, search for locations that are likely to produce that percept.

        The results will be used to guide the agent's search for the prototype.

        :param target_prototype: The target prototype. Has shape (batch_size, sensory_dim,).
        :param num_results: The number of results to return.
        :param diversity_steps: The number of steps to take to diversify the results.
        :return: The target location. Has shape (batch_size, location_dim).
        """
        top_locations, top_senses, found, num_found = self.memory.search(target_prototype, num_results=num_results, diversity_steps=diversity_steps)
        return top_locations, top_senses, found, num_found

    def get_curiosity_target(self, location: torch.Tensor):
        """
        Get a curiosity target location.

        For now, just return a random location.
        """
        return torch.randn_like(location)

    def get_expected_sensory(self, location: torch.Tensor, location_sd: torch.Tensor):
        """
        Get the expected sensory for a location.
        """
        sensory = self.encoder.get_expected_sensory(location, location_sd)
        return sensory

    def break_training_graph(self):
        self.encoder.break_training_graph()
        self.decoder.break_training_graph()

    def interpret(self, location_model: torch.nn.Module, drive_classifier: torch.nn.Module):
        """
        Interpret the memory locations using a location model.
        """
        return self.memory.interpret(location_model, drive_classifier)

    def regularize(self, loss):
        return self.memory.regularize(self.localizer.regularize(loss))

    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        return cls(config.location_dim, config.sensory_embedding_dim, config.dim, config.embed_dim, config.num_guesses, config.dropout)


class TEMDecoder(SimpleTEMDecoder):

    def forward(self, last_location: torch.Tensor, action: torch.Tensor, heading: torch.Tensor, distance: torch.Tensor, self_location: torch.Tensor=None, obj_location: torch.Tensor=None, actual_sensory: torch.Tensor=None):
        """
        Forward pass for the TEM decoder.

        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :param actual_sensory: the actual sensory reading to write to memory
        :return: The next location, the next sensory reading, the log probability of the location and the log probability of the sensory reading.
        """

        next_location, location_mean, location_sd = self.localizer(last_location, action, force_location=self_location)

        if distance is None:
            # nothing sensed along heading, no need to guess where
            return next_location, None, None, location_mean, location_sd, None, None


        obj_location, obj_location_mean, obj_location_sd = self.localizer(next_location, heading, distance, force_location=obj_location)

        loc_pair = torch.cat([next_location, obj_location], dim=-1)

        sensory_mean = self.memory.read(obj_location, obj_location_sd)

        sensory_sd = torch.nn.functional.softplus(self.sensory_variance_proj(loc_pair))
        sensory_sd = self.min_sd + torch.nn.functional.softplus(self.sensory_variance_logits) * sensory_sd
        
        sensory = sensory_mean + torch.randn_like(sensory_mean) * sensory_sd

        if actual_sensory is None:
            actual_sensory = sensory

        log_prob_sensory = -0.5 * ((actual_sensory - sensory_mean) / sensory_sd).pow(2).sum(dim=-1).mean() - torch.log(sensory_sd).sum(dim=-1).mean() - 0.5 * math.log(2 * math.pi) * sensory_sd.shape[-1]

        return next_location, sensory, log_prob_sensory, location_mean, location_sd, obj_location_mean, obj_location_sd


class TEMEncoder(SimpleTEMEncoder):
    def forward(self, sensory: torch.Tensor, last_location: torch.Tensor, action: torch.Tensor, heading: torch.Tensor, distance: torch.Tensor):
        """
        Forward pass for the TEM encoder.

        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        next_location, next_location_mean, next_location_sd = self.localizer(last_location, action)

        if distance is None:
            # nothing sensed along heading, no need to guess where
            return next_location, None, next_location_mean, next_location_sd, None, None
        
        expected_obj_location_mean, expected_obj_location_sd = self.localizer(next_location, heading, distance, return_distribution=True)
        location_guesses, location_sds, location_senses, found, num_found = self.memory.search(sensory, num_results=self.num_guesses)
        num_found = num_found.view(-1)  # should be (batch_size, 1) --> (batch_size,)

        invalid_guess_mask = torch.arange(location_guesses.shape[1], device=location_guesses.device)[None, :] >= num_found[:, None]
        ignore_guesses = num_found == 0

        obj_location = self.select_location(location_guesses, location_sds, expected_obj_location_mean, expected_obj_location_sd, 
                                                          invalid_guess_mask, ignore_guesses)

        return next_location, obj_location, next_location_mean, next_location_sd, expected_obj_location_mean, expected_obj_location_sd


class TEMModel(SimpleTEMModel):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, num_guesses: int, dropout: float=0.1, batch_size: int=1):
        super().__init__(
            location_dim, sensory_dim, action_dim, embed_dim, num_guesses, dropout, batch_size, is_simple=False
        )

        self.encoder = TEMEncoder(self.memory, self.localizer, location_dim, sensory_dim, action_dim, embed_dim, num_guesses, dropout)
        self.decoder = TEMDecoder(self.memory, self.localizer, location_dim, sensory_dim, action_dim, embed_dim, dropout)

    def forward(self, sensory: torch.Tensor, action: torch.Tensor, last_location: torch.Tensor, 
                heading: torch.Tensor, distance: torch.Tensor, force_location: torch.Tensor=None):
        """
        Forward pass for the TEM model.

        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :param action: The action to take. Has shape (batch_size, action_dim). Unit vector.
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim). Unit vector.
        :param distance: The distance to the object. Has shape (batch_size,).
        :param force_location: The location to force the model to use. Has shape (batch_size, location_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        if last_location is None:
            last_location = torch.zeros(sensory.shape[0], self.location_dim, device=sensory.device)

        if action is None:
            action = torch.zeros(sensory.shape[0], self.action_dim, device=sensory.device)

        next_location, obj_location, next_location_mean, next_location_sd, obj_location_mean, obj_location_sd = self.encoder(
            sensory, last_location, action, heading, distance
        )
        next_location_gen, sensory_gen, log_prob_sensory_gen, loc_mean_gen, loc_sd_gen, obj_loc_mean_gen, obj_loc_sd_gen = self.decoder(
            last_location, action, heading, distance, self_location=next_location, obj_location=obj_location, actual_sensory=sensory
        )

        return_location = next_location_gen if force_location is None else force_location

        kl_next = kl_divergence_gaussian(next_location_mean, next_location_sd, loc_mean_gen, loc_sd_gen)
        if distance is None:
            elbo = -kl_next
            kl_obj = None

            if sensory_gen is not None and torch.norm(sensory_gen) > 0.05:
                self.memory.write(obj_loc_mean_gen, obj_loc_sd_gen, torch.zeros_like(sensory_gen))

        else:
            self.memory.write(obj_location, obj_location_sd, sensory)

            kl_obj = kl_divergence_gaussian(obj_location_mean, obj_location_sd, obj_loc_mean_gen, obj_loc_sd_gen)

            elbo = log_prob_sensory_gen - 10000 * (kl_next + kl_obj)

        return return_location, loc_sd_gen, obj_loc_mean_gen, obj_loc_sd_gen, elbo, log_prob_sensory_gen, kl_next, kl_obj
