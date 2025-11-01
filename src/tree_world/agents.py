import torch

from tree_world.simulation import AgentModel, Sensor, TreeWorldConfig
from tree_world.models.drives import DriveEmbeddingClassifier, train_drive_classifier
from tree_world.models.tem import TEMModel, TEMActionEncoder


class StateBasedAgent(AgentModel):
    target_position: torch.Tensor = None
    travel_distance: float = 0.0
    start_heading: torch.Tensor = None
    half_circle = False

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float, valence: float=1.0):
        if self.start_heading is None and self.target_position is None:
            self.start_heading = heading
            self.half_circle = False

        if distance is not None and valence > 0:
            # there is fruit, move towards it
            position_delta = heading
            new_heading = heading
        
        elif distance is not None and valence < 0:
            # there is poison, move away from it
            position_delta = -heading
            new_heading = heading
        
        elif self.target_position is None:
            # spin in place
            orthogonal_direction = self.get_orthogonal_direction(heading)
            new_heading = heading + 0.1 * orthogonal_direction
            new_heading = new_heading / torch.norm(new_heading)

            dp = torch.dot(new_heading, self.start_heading)
            if dp < 0.0:
                self.half_circle = True
            elif self.half_circle and dp > 0.8:
                self.start_heading = None
                self.target_position = torch.randn(self.dim) * 250
                self.travel_distance = 0.0

            position_delta = torch.zeros(self.dim)

        else:
            # move towards the target position
            new_heading = self.target_position / torch.norm(self.target_position)
            position_delta = new_heading
            self.travel_distance = self.travel_distance + torch.norm(position_delta)

            if self.travel_distance > 25:
                self.half_circle = False
                self.start_heading = new_heading
                self.target_position = None
                self.travel_distance = 0.0

        return position_delta, new_heading


class StateBasedAgentWithDriveEmbedding(StateBasedAgent):
    def __init__(self, sensory_embedding_dim: int, sensory_embedding_model: str, dim: int=2, can_see_fruit_distance: float=10.0, drive_embedding_model: DriveEmbeddingClassifier=None, drive_keys: dict=None):
        super().__init__(sensory_embedding_dim, sensory_embedding_model, dim, can_see_fruit_distance)
        self.drive_embedding_model = drive_embedding_model
        self.drive_keys = drive_keys

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float):

        if distance is None:
            valence = 1.0

        else:
            drive_targets = self.drive_embedding_model(embedding.clone()[None, :])[0]
            if drive_targets[self.drive_keys["poison"]] > 0.1:
                valence = -1.0
            elif drive_targets[self.drive_keys["edible"]] > 0.1:
                valence = 1.0
            else:
                valence = 0.0
        
        print(f"Valence: {valence}")

        position_delta, new_heading = super().get_action(distance, embedding, heading, health, valence)
        return position_delta, new_heading

    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        drive_embedding_model, drive_keys = train_drive_classifier(config)
        return cls(config.sensory_embedding_dim, config.sensory_embedding_model, config.dim, config.can_see_fruit_distance, drive_embedding_model, drive_keys)


class TEMAgent(AgentModel):
    def __init__(self, sensory_embedding_dim: int, sensory_embedding_model: str, dim: int=2, can_see_fruit_distance: float=10.0, tem_model: TEMModel=None, action_encoder: TEMActionEncoder=None, drive_embedding_model: DriveEmbeddingClassifier=None, drive_keys: dict=None):
        super().__init__(sensory_embedding_dim, sensory_embedding_model, dim, can_see_fruit_distance)
        self.tem_model = tem_model
        self.action_encoder = action_encoder
        self.drive_embedding_model = drive_embedding_model
        self.drive_keys = drive_keys

        self.exploration_mode = True

        self.last_action = None
        self.last_grid = None

        self.target_grid = None

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float):

        # report the current sensory information to the TEM model to update the map
        next_grid, elbo = self.tem_model(sensory, self.last_action, self.last_grid, heading)

        # do we have a target? If so should we keep it?
        log_action_prob = None
        if self.target_grid is not None:

            # we were trying to get somewhere ... did we get there?
            log_action_prob = self.action_encoder.log_action_prob(self.last_grid, next_grid, self.target_grid)

            # have we arrived at the target?
            if torch.norm(next_grid - self.target_grid) < 1.0:
                self.target_grid = None

            elif self.exploration_mode and health < 0.5:
                # cancel exploration ... we are too hungry
                self.target_grid = None
                self.exploration_mode = False

        elif health > 0.5:

                self.exploration_mode = True
        
        
        # Now we have data to update the TEM model and action encoder

        # select a target
        if self.target_grid is None:
            if self.exploration_mode:
                # curiosity based selection of target
                target_grid = self.tem_model.get_curiosity_target(next_grid)
                target_grid = target_grid / torch.norm(target_grid)

            else:
                # select a target grid based on the drive embedding
                hunger_idx = self.drive_keys["edible"]
                hunger_value = self.drive_embedding_model.drive_embeddings.weight[hunger_idx]

                self.tem_model.memory.search(hunger_value[None, :], num_results=5)





        


        self.last_grid = next_grid.detach()
        self.last_action = action.detach()


        return position_delta, new_heading
    
    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        tem_model = TEMModel.from_config(config)
        action_encoder = TEMActionEncoder.from_config(config)
        drive_embedding_model, drive_keys = train_drive_classifier(config)
        return cls(config.sensory_embedding_dim, config.sensory_embedding_model, config.dim, config.can_see_fruit_distance, 
                   tem_model, action_encoder, drive_embedding_model, drive_keys)

