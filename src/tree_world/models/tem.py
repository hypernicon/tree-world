import torch
from .memory import BidrectionalMemory, normalize_location, unnormalize_location

"""
Loosely based on the Tolman-Eichenbaum model of memory encoding and retrieval.

https://www.sciencedirect.com/science/article/pii/S009286742031388X
"""


class TEMLocalizer(torch.nn.Module):
    def __init__(self, location_dim: int, action_dim: int, hidden_dim: int=128, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.localizer = torch.nn.Sequential(
            torch.nn.Linear(location_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, location_dim),
        )

        self.variance_proj = torch.nn.Sequential(
            torch.nn.Linear(location_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, location_dim),
        )

        self.variance_scale_logits = torch.nn.Parameter(torch.zeros(location_dim - 1))

    def forward(self, location: torch.Tensor, action: torch.Tensor, return_distribution: bool=False):
        location_mean = self.localizer(torch.cat([location, action], dim=1))
        location_sd = torch.exp(self.variance_proj(location_mean))
        location_sd = torch.exp(self.variance_scale_logits) * location_sd + 1e-6

        if return_distribution:
            return location_mean, location_sd

        next_location = location_mean + torch.randn_like(location_mean) * location_sd

        location_log_prob = -0.5 * ((next_location - location_mean) / location_sd).pow(2).sum(dim=-1).mean() - torch.log(location_sd).sum(dim=-1).mean()

        return next_location, location_log_prob


class TEMDecoder(torch.nn.Module):

    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, memory_size: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.dropout = dropout

        self.memory = BidrectionalMemory(location_dim, sensory_dim, embed_dim, memory_size, dropout)

        self.localizer = TEMLocalizer(location_dim, action_dim, embed_dim, dropout)

        # accuracy of sensory read depends on how far away the object is
        self.sensory_variance_proj = torch.nn.Sequential(
            torch.nn.Linear(2 * location_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, sensory_dim),
        )
        self.sensory_variance_logits = torch.nn.Parameter(torch.zeros(sensory_dim))

    def reset(self):
        self.memory.reset()

    def forward(self, last_location: torch.Tensor, action: torch.Tensor, heading: torch.Tensor, actual_sensory: torch.Tensor=None):
        """
        Forward pass for the TEM decoder.

        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :param actual_sensory: the actual sensory reading to write to memory
        :return: The next location, the next sensory reading, the log probability of the location and the log probability of the sensory reading.
        """

        next_location, location_log_prob = self.localizer(last_location, action)
        obj_location, obj_location_log_prob = self.localizer(next_location, heading)

        loc_pair = torch.cat([next_location, obj_location], dim=-1)

        sensory_mean = self.memory.read(obj_location[..., None, :])

        sensory_sd = torch.exp(self.sensory_variance_proj(loc_pair))
        sensory_sd = torch.exp(self.sensory_variance_logits) * sensory_sd + 1e-6
        
        sensory = sensory_mean + torch.randn_like(sensory_mean) * sensory_sd

        log_prob_sensory = -0.5 * ((sensory - sensory_mean) / sensory_sd).pow(2).sum(dim=-1).mean() - torch.log(sensory_sd).sum(dim=-1).mean()

        # TODO: what do we WRITE to the memory? can we write the correct thing during training?
        if actual_sensory is not None:
            self.memory.write(obj_location[..., None, :], actual_sensory[..., None, :])
        else:
            self.memory.write(obj_location[..., None, :], sensory[..., None, :])

        return next_grid, sensory, log_prob_grid, log_prob_sensory


class TEMEncoder(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, heading_dim: int, embed_dim: int, num_guesses: int, memory_size: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.heading_dim = heading_dim
        self.embed_dim = embed_dim
        self.num_guesses = num_guesses
        self.memory_size = memory_size
        self.dropout = dropout

        self.memory = BidrectionalMemory(location_dim, sensory_dim, embed_dim, memory_size, dropout)

        self.localizer = TEMLocalizer(location_dim, action_dim, embed_dim, dropout)

    def reset(self):
        self.memory.reset()

    def select_location(self, location_guesses: torch.Tensor, expected_location_mean: torch.Tensor, expected_location_sd: torch.Tensor,
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
        expected_location_spherical = normalize_location(expected_location_mean)
        location_guesses_spherical = normalize_location(location_guesses)

        # expected_location_spherical has shape (batch_size, location_dim)
        # location_guesses has shape (batch_size, num_guesses, location_dim)
        # matches has shape (batch_size, num_guesses)
        matches = torch.bmm(location_guesses_spherical, expected_location_spherical[..., None]).squeeze(-1)
        matches = matches.where(invalid_guess_mask, torch.tensor(float('-inf')))
        matches = torch.softmax(matches, dim=-1)

        integrated_location_spherical = (matches[..., None, :] * location_guesses).sum(dim=-2)
        integrated_location = unnormalize_location(integrated_location_spherical)

        # check if the integrated location is two standard deviations away from the expected location
        too_far = torch.norm(-0.5 * ((integrated_location - expected_location_mean) / expected_location_sd), dim=-1) > 2.0

        expected_location = expected_location_mean + torch.randn_like(expected_location_mean) * expected_location_sd

        too_far = too_far[..., None]
        ignore_guesses = ignore_guesses[..., None]
        next_location = integrated_location.where(too_far | ignore_guesses | torch.isnan(integrated_location), expected_location)
        next_location_log_prob = -0.5 * ((next_location - expected_location_mean) / expected_location_sd).pow(2).sum(dim=-1).mean() - torch.log(expected_location_sd).sum(dim=-1).mean()

        return next_location, next_location_log_prob
        
    def forward(self, sensory: torch.Tensor, last_location: torch.Tensor, action: torch.Tensor, heading: torch.Tensor):
        """
        Forward pass for the TEM encoder.

        :param grid: The current grid state. Has shape (batch_size, grid_dim).
        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        location_guesses, indices, found, scores = self.memory.search(sensory[..., None, :], num_results=self.num_guesses)
        found = found.view(-1)  # should be (batch_size, 1) --> (batch_size,)

        next_location, next_location_log_prob = self.localizer(last_location, action)
        expected_obj_location_mean, expected_obj_location_sd = self.localizer(next_location, heading, return_distribution=True)

        invalid_guess_mask = torch.arange(self.num_guesses, device=location_guesses.device)[None, :] >= found[:, None]
        ignore_guesses = found == 0

        obj_location, obj_log_prob = self.select_location(location_guesses, expected_obj_location_mean, expected_obj_location_sd, 
                                                          invalid_guess_mask, ignore_guesses)

        self.memory.write(obj_location[..., None, :], sensory[..., None, :])

        return next_location, next_location_log_prob + obj_log_prob


class TEMModel(torch.nn.Module):
    def __init__(self, location_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, 
                 heading_dim: int, memory_size: int, num_guesses: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.heading_dim = heading_dim
        self.memory_size = memory_size
        self.num_guesses = num_guesses
        self.dropout = dropout

        self.encoder = TEMEncoder(location_dim, sensory_dim, action_dim, embed_dim, heading_dim, memory_size, num_guesses, dropout)
        self.decoder = TEMDecoder(location_dim, sensory_dim, action_dim, embed_dim, heading_dim, memory_size, num_guesses, dropout)

    def forward(self, sensory: torch.Tensor, action: torch.Tensor, last_location: torch.Tensor, heading: torch.Tensor, last_location_gen: torch.Tensor=None):
        """
        Forward pass for the TEM model.

        :param sensory: The sensory input. Has shape (batch_size, sensory_dim).
        :param action: The action to take. Has shape (batch_size, action_dim).
        :param last_location: The last location. Has shape (batch_size, location_dim).
        :param heading: The heading. Has shape (batch_size, heading_dim).
        :param last_location_gen: The last location generated by the decoder. Has shape (batch_size, location_dim).
        :return: The next grid state. Has shape (batch_size, grid_dim).
        """
        if last_location_gen is None:
            last_location_gen = last_location

        next_location, log_prob_location = self.encoder(sensory, last_location, action, heading)
        next_location_gen, sensory_gen, log_prob_location_gen, log_prob_sensory_gen = self.decoder(
            last_location_gen, action, heading, actual_sensory=sensory
        )

        elbo = log_prob_sensory_gen - log_prob_location + log_prob_location_gen

        return next_grid, elbo

    def reset(self):
        self.encoder.reset()
        self.decoder.reset()

    def search_for_target_location(self, target_prototype: torch.Tensor):
        """
        Given a prototype of the desired sensory percept, search for locations that are likely to produce that percept.

        The results will be used to guide the agent's search for the prototype.

        :param target_prototype: The target prototype. Has shape (batch_size, sensory_dim,).
        :return: The target location. Has shape (batch_size, location_dim).
        """
        return self.memory.search(target_prototype[..., None, :], num_results=1)


class TEMActionEncoder(torch.nn.Module):
    def __init__(self, grid_dim: int, sensory_dim: int, action_dim: int, embed_dim: int, dropout: float=0.1):
        super().__init__()
        self.grid_dim = grid_dim
        self.sensory_dim = sensory_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.action_model = torch.nn.Sequential(
            torch.nn.Linear(2*grid_dim, embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, action_dim)
        )

    def forward(self, next_grid: torch.Tensor, target_grid: torch.Tensor):
        action = self.action_model(torch.cat([next_grid, target_grid], dim=1))
        return action

    def log_action_prob(self, last_grid: torch.Tensor, next_grid: torch.Tensor, target_grid: torch.Tensor, grid_variance: torch.Tensor):
        log_prob = -((next_grid - target_grid) / grid_variance).pow(2).sum(dim=-1).mean() - torch.log(grid_variance).sum(dim=-1).mean()
        return log_prob
