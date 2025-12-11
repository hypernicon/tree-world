from typing import Optional
import math
import torch

from tree_world.simulation import AgentModel, TreeWorldConfig
from tree_world.models.memory import SpatialMemory
from tree_world.states import Location, DriveManager, DriveTarget
from tree_world.models.drives import train_drive_classifier, DriveEmbeddingClassifierNonExclusive


class DriveBasedAgentWithMemory(AgentModel):
    action_scale: float = 1.0
    space_scale: float = 500.0
    sigma_scale: float = 25.0
    hunger_threshold: float = 0.5
    
    def __init__(self, 
        sensory_embedding_dim: int, 
        sensory_embedding_model: str, 
        dim: int=2, 
        can_see_fruit_distance: float=10.0, 
        max_distance: float=100.0,
        memory: SpatialMemory=None,
        drive_manager: DriveManager=None,
    ):
        super().__init__(sensory_embedding_dim, sensory_embedding_model, dim, can_see_fruit_distance, max_distance)
        assert memory is not None
        assert drive_manager is not None

        self.memory = memory
        self.drive_manager = drive_manager

        self.is_hungry = False
        self.target = None
        self.recent_targets = []
    
    def reset(self):
        super().reset()
        self.memory.reset()
        self.is_hungry = False
        self.target = None
        self.recent_targets = []

    def is_recent_target(self, target: Optional[DriveTarget]) -> bool:
        if target is None:
            return False

        for other_target in self.recent_targets:
            diff = target.target_location.location - other_target.target_location.location
            if torch.norm(diff) < self.sigma_scale:
                return True

        return False

    def select_target(self, agent_location: Location):
        target = None
        if self.is_hungry:
            if self.target is not None and isinstance(self.target, DriveTarget):
                target = self.target
            else:
                target = self.drive_manager.choose_hunger_target(
                    agent_location,
                    temperature=1.0,
                    sigma_scale=self.sigma_scale,
                    num_samples=25,
                    location_temperature=10_000,
                    match_threshold=None,
                    lower_match_threshold=25.0,
                )

                if self.is_recent_target(target):
                    target = self.drive_manager.choose_hunger_target(
                        agent_location,  
                        temperature=1.0,
                        sigma_scale=self.sigma_scale,
                        num_samples=25,
                        location_temperature=10_000,
                        match_threshold=None,
                        lower_match_threshold=25.0,
                        use_location=False, # search for ANY hunger target without location, trying to escape local well
                    )
                
                    if self.is_recent_target(target):
                        target = None

        if target is None:
            target = self.drive_manager.choose_curiosity_target(agent_location, self.space_scale)

        if target is not None:
            self.recent_targets.append(target)
            self.recent_targets = self.recent_targets[-5:]
    
        return target
    
    def adjust_heading(self, heading: torch.Tensor, agent_location: Location):
        return heading

    def update_rewards(self, reward: float, agent_location: Location=None):
        pass

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None, reward: float=0.0):
        assert agent_location is not None, "DriveBasedAgent requires perfect localization"
        agent_location = Location(agent_location, torch.ones_like(agent_location) * 5.0)

        self.update_rewards(reward, agent_location)

        was_hungry = self.is_hungry
        self.is_hungry = health <= self.hunger_threshold
        drive_changed = self.is_hungry != was_hungry

        if self.target is not None:
            self.target.update_current_location(agent_location)
            if self.target.has_arrived():
                # print(f"Agent reached target: ({self.target.target_location.location[0]:.2f}, {self.target.target_location.location[1]:.2f}) -- ", end="")
                # print(f"Agent location: ({agent_location.location[0]:.2f}, {agent_location.location[1]:.2f})")
                self.target = None

        if drive_changed or self.target is None:
            self.target = self.select_target(agent_location)

            # print(f"Agent set target (hunger:{self.is_hungry}): {self.target.__class__.__name__}({self.target.target_location.location[0].item():.2f}, {self.target.target_location.location[1].item():.2f})")
            
        assert self.target is not None, "No target found"
        position_delta = self.target.get_heading() * self.action_scale

        position_delta = self.adjust_heading(position_delta, agent_location)

        self.memory.write(agent_location.location[None, :], agent_location.location_sd[None, :], embedding[None, :])

        return position_delta, position_delta.clone()

    def train(self):
        self.memory.prune()
    
    def get_drive_field_from_memory(self, grid_locations: torch.Tensor):
        grid_locations = grid_locations[None, ...]
        grid_locations_sd = torch.ones_like(grid_locations) * 5.0
        memory_values = self.memory.read(grid_locations, grid_locations_sd, match_threshold=25.0).squeeze(0)
        drive_values = self.drive_manager.drive_embedding_model(memory_values)
        return drive_values
    
    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        memory = SpatialMemory.from_config(config)
        drive_classifier, drive_keys = train_drive_classifier(config, with_ids=False, nonexclusive=True)
        drive_manager = DriveManager(drive_classifier, drive_keys, memory)
        return cls(
            config.sensory_embedding_dim,
            config.sensory_embedding_model,
            config.dim,
            config.can_see_fruit_distance,
            config.max_sense_distance,
            memory=memory,
            drive_manager=drive_manager,
        )


def build_local_map(
    memory: SpatialMemory, 
    location: Location, 
    drive_manager: DriveManager,
    num_grid_points: int=10, 
    grid_size: float=10.0,
    match_threshold: float=25.0,
):
    grid_extent = grid_size * num_grid_points
    grid_points = torch.linspace(-grid_extent, grid_extent, num_grid_points)
    grid_locations = (
        location.location.view(1, 1, 2) 
        + torch.cartesian_prod(grid_points, grid_points).view(1, -1, 2)
    )
    grid_locations_sd = torch.ones_like(grid_locations) * grid_size / 2

    memory_values = memory.read(grid_locations, grid_locations_sd, match_threshold=match_threshold).squeeze(0)
    grid_valence = drive_manager.assess_valence(memory_values)

    return grid_locations, grid_valence


def build_forward_local_map(
    memory: SpatialMemory,
    location: Location,
    heading: torch.Tensor,
    drive_manager: DriveManager,
    num_grid_points: int=10,
    grid_size: float=10.0,
    match_threshold: float=25.0,
    return_raw_drives: bool=False,
):
    axis_points = torch.arange(num_grid_points, dtype=heading.dtype, device=heading.device)
    axis_grid = torch.cartesian_prod(axis_points, axis_points)

    # we want to get the vector to the diagonal of the grid, which is y=x
    rotation_matrix = math.sqrt(0.5) * torch.tensor([
        [1, 1],
        [-1, 1],
    ], dtype=heading.dtype, device=heading.device)

    grid_points_relative = (axis_grid @ rotation_matrix.T) * grid_size

    heading_normalized = heading / torch.norm(heading)
    heading_orthogonal = torch.tensor([-heading_normalized[1].item(), heading_normalized[0].item()], 
                                      dtype=heading.dtype, device=heading.device)[None, :]

    grid_points_absolute = location.location[None, None, :] + (
        grid_points_relative[..., 0, None] * heading_normalized + grid_points_relative[..., 1, None] * heading_orthogonal
    )[None, ...]

    grid_locations_sd = torch.ones_like(grid_points_absolute) * grid_size / 2

    memory_values = memory.read(grid_points_absolute, grid_locations_sd, match_threshold=match_threshold).squeeze(0)
    grid_valence = drive_manager.assess_valence(memory_values, return_raw_drives=return_raw_drives)

    deviations = grid_points_relative[..., 1] 

    return grid_points_absolute, grid_valence, deviations, heading_orthogonal


def apply_deviation(
    location: Location, heading: torch.Tensor, memory: SpatialMemory, drive_manager: DriveManager,
    deviation_strength: float=1.0,
    num_grid_points: int=10, grid_size: float=10.0, match_threshold: float=25.0,
    return_coefficient: bool=False,
):
    locs, valence, deviations, heading_orthogonal = build_forward_local_map(
        memory, location, heading, drive_manager, num_grid_points, grid_size, match_threshold
    )

    grid_extent = grid_size * num_grid_points
    max_deviation_norm = grid_extent * math.sqrt(0.5)

    # deviation is the scalar magnitude and direction of the deviation vector
    # positive or negative depending on whether the point is to the left or right of the heading

    # for negative valence, we want to push away HARDER if the point is closer to our line of travel
    # whereas for positive valence, we want to push towards HARDER if the point is further away from our line of travel
    # we also want to choose the sign of the response based on the valence sign, so that we push away from poison and towards edible fruit
    negative_deviations = -(max_deviation_norm - deviations.abs()) * torch.sign(deviations)

    delta_delta = deviation_strength * torch.where(valence < 0.0, negative_deviations, deviations) * valence.abs()

    if return_coefficient:
        return delta_delta.mean()

    raw_heading = heading + delta_delta.mean() * heading_orthogonal
    heading_norm = torch.norm(heading)
    return heading_norm * raw_heading / torch.norm(raw_heading)


class DriveBasedAgentWithMemoryAndLocalMap(DriveBasedAgentWithMemory):
    deviation_strength: float = 1.0

    def adjust_heading(self, heading: torch.Tensor, agent_location: Location):
        return apply_deviation(
            agent_location, heading, self.memory, self.drive_manager, 
            deviation_strength=self.deviation_strength,
            num_grid_points=5, grid_size=5.0, match_threshold=25.0
        ).detach().squeeze()


class LocalMapPolicy(torch.nn.Module):
    def __init__(self, drive_field_dim: int, hidden_dim: int=128, scale: float=25.0):
        super().__init__()
        self.scale = scale

        self.fc1 = torch.nn.Linear(drive_field_dim + 1, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, last_coefficient: torch.Tensor, output: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = torch.relu(self.fc1(torch.cat([x, last_coefficient], dim=1)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        mean_output = x[:, 0] * self.scale
        std_output = torch.nn.functional.softplus(x[:, 1])

        if output is None:
            output = mean_output + std_output * torch.randn_like(mean_output)

        log_prob = -torch.log(std_output) - 0.5 * ((output - mean_output) / std_output) ** 2 - 0.5 * math.log(2 * math.pi)
        return output, log_prob.mean()


class DriveBasedAgentWithLocalPolicy(DriveBasedAgentWithMemory):
    deviation_strength: float = 5.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = 5.0
        self.num_grid_points = 5

        drive_field_dim = self.num_grid_points * self.num_grid_points * 3
        self.policy = LocalMapPolicy(drive_field_dim, scale=self.grid_size * self.num_grid_points / 4)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-3)
        self.total_rewards = 0.0
        self.total_rewards_baseline = 0.0
        self.total_rewards_baseline_decay = 0.9
        self.total_log_probs = 0.0

        self.use_policy = True

        self.last_coefficient = 0.0

    def reset(self):
        super().reset()
        self.total_rewards = 0.0
        self.last_coefficient = 0.0

        self.use_policy = not self.use_policy
        # print the execution stack to see who is calling reset
        if self.use_policy:
            print(f"Using policy model")
        else:
            print(f"Using baseline model")
    
    def update_rewards(self, reward: float, agent_location: Location=None):
        self.total_rewards += reward

    def heading_from_policy(self, heading: torch.Tensor, agent_location: Location, output: Optional[torch.Tensor]=None):
        _, drive_field, _, heading_orthogonal = build_forward_local_map(
            self.memory, agent_location, heading, self.drive_manager, 
            num_grid_points=self.num_grid_points, grid_size=self.grid_size, 
            match_threshold=25.0, return_raw_drives=True
        )

        drive_field = drive_field.view(1, -1)
        last_coefficient = torch.tensor([[self.last_coefficient]], dtype=drive_field.dtype, device=drive_field.device)

        coefficient, log_prob =self.policy(drive_field.detach(), last_coefficient, output)
        self.total_log_probs = self.total_log_probs + log_prob

        raw_heading = heading + coefficient.squeeze(0) * heading_orthogonal.squeeze(0)
        heading_norm = torch.norm(heading)
        return (heading_norm * raw_heading / torch.norm(raw_heading)).detach(), coefficient.squeeze(0)
    
    def heading_from_deviation(self, heading: torch.Tensor, agent_location: Location):
        return apply_deviation(
            agent_location, heading, self.memory, self.drive_manager, 
            deviation_strength=self.deviation_strength,
            num_grid_points=5, grid_size=5.0, match_threshold=25.0,
            return_coefficient=True
        ).detach().squeeze()
    
    def adjust_heading(self, heading: torch.Tensor, agent_location: Location):
        if self.use_policy:
            heading, coefficient = self.heading_from_policy(heading, agent_location)
            self.last_coefficient = coefficient.item()
            return heading
        else:
            output = self.heading_from_deviation(heading, agent_location)
            return self.heading_from_policy(heading, agent_location, output)[0]

    def train(self):
        super().train()

        if self.total_rewards == 0.0:
            return

        self.optimizer.zero_grad()

        print(f"Training policy with total rewards {self.total_rewards} vs. baseline {self.total_rewards_baseline}")
        advantage = self.total_rewards - self.total_rewards_baseline
        loss = -advantage * self.total_log_probs

        loss.backward()

        self.optimizer.step()

        if not self.use_policy:
            d = self.total_rewards_baseline_decay
            self.total_rewards_baseline = self.total_rewards_baseline * d + self.total_rewards * (1.0 - d)

        self.total_log_probs = 0.0
        self.total_rewards = 0.0
