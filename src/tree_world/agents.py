import sys
import torch
from typing import Tuple

from tree_world.simulation import AgentModel, Sensor, TreeWorldConfig
from tree_world.models.drives import DriveEmbeddingClassifier, train_drive_classifier
from tree_world.models.tem import TEMModel
from tree_world.models.actions import ActionEncoder


class StateBasedAgent(AgentModel):
    target_position: torch.Tensor = None
    travel_distance: float = 0.0
    start_heading: torch.Tensor = None
    half_circle = False

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None, valence: float=1.0):
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

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None):

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
        
        # print(f"Valence: {valence}")

        position_delta, new_heading = super().get_action(distance, embedding, heading, health, valence)
        return position_delta, new_heading

    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        drive_embedding_model, drive_keys = train_drive_classifier(config)
        return cls(config.sensory_embedding_dim, config.sensory_embedding_model, config.dim, config.can_see_fruit_distance, drive_embedding_model, drive_keys)


class DriveManager:
    def __init__(self, drive_embedding_model: DriveEmbeddingClassifier, drive_keys: dict, tem_model: TEMModel):
        self.drive_embedding_model = drive_embedding_model
        self.drive_keys = drive_keys
        self.tem_model = tem_model
    
    def choose_hunger_target(self, location: 'Location', tem_model: TEMModel) -> 'Location':
        hunger_idx = self.drive_keys["edible"]
        hunger_value = self.drive_embedding_model.drive_embeddings.weight[hunger_idx]

        top_locations, top_location_sds, top_senses, _, num_found = self.tem_model.memory.search(hunger_value[None, :], num_results=5)
        if num_found[0] > 0:
            # print(f"Selected target location to satisfy hunger")
            target_location = Location(top_locations[0,:1].detach(), top_location_sds[0,:1].detach())
            target = DriveTarget(hunger_value.detach(), top_senses[0,:1].detach(), location, target_location)
            return target
        else:
            return None

    def assess_valence(self, sensory: torch.Tensor) -> float:
        drive_targets = self.drive_embedding_model(sensory.clone()[None, :])[0]
        if drive_targets[self.drive_keys["poison"]] > 0.1:
            return -1.0
        elif drive_targets[self.drive_keys["edible"]] > 0.1:
            return 1.0
        else:
            return 0.0


class Location:
    def __init__(self, location: torch.Tensor, location_sd: torch.Tensor):
        self.location = location
        self.location_sd = location_sd
        self.estimated_location = None

    def interpret(self, location_model: torch.nn.Module):
        self.estimated_location = location_model(self.location)
        if self.estimated_location is not None:
            self.estimated_location = self.estimated_location.detach().squeeze()


class Target:
    arrive_z_threshold: float = 0.1

    def __init__(self, 
        start_location: Location, 
        target_location: Location,
        location_model: torch.nn.Module=None
    ):
        self.start_location = start_location
        self.current_location = start_location
        self.target_location = target_location

        self.location_model = location_model
        self.target_location_estimated = None

        if self.location_model is not None:
            self.start_location.interpret(self.location_model)
            self.current_location.interpret(self.location_model)
            self.target_location.interpret(self.location_model)

    def update_current_location(self, location: Location):
        self.current_location = location
        if self.location_model is not None:
            self.current_location.interpret(self.location_model)
    
    def update_location_model(self, location_model: torch.nn.Module):
        self.location_model = location_model
        if self.location_model is not None:
            self.start_location.interpret(self.location_model)
            self.current_location.interpret(self.location_model)
            self.target_location.interpret(self.location_model)

    def has_arrived(self) -> bool:
        z = self.current_location.location - self.target_location.location
        if self.target_location.location_sd is not None:
            sd = self.target_location.location_sd + self.current_location.location_sd
        else:
            sd = self.current_location.location_sd
        
        z = z / (sd + 1e-6)

        return torch.norm(z) < self.arrive_z_threshold


class DriveTarget(Target):
    def __init__(self, 
        drive_embedding: torch.Tensor,
        sensory_target: torch.Tensor,
        start_location: Location, 
        target_location: Location,
        location_model: torch.nn.Module=None
    ):
        super().__init__(start_location, target_location, location_model)
        self.drive_embedding = drive_embedding
        self.sensory_target = sensory_target


class State:
    avoid_distance: float = 10.0
    health_cutoff: float = 0.5
    last_heading_delta: torch.Tensor = None

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        raise NotImplementedError("Subclasses must implement this method")

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_orthogonal_direction(self, heading: torch.Tensor) -> torch.Tensor:
        dim = heading.shape[-1]
        orthogonal_direction = torch.randn(dim)
        orthogonal_direction = orthogonal_direction - torch.dot(orthogonal_direction, heading) * heading
        orthogonal_direction = orthogonal_direction / torch.norm(orthogonal_direction)
        if self.last_heading_delta is not None:
            dp = torch.dot(orthogonal_direction, self.last_heading_delta)
            if dp < 0:
                # keep the orthogonal direction in the same direction
                orthogonal_direction = -orthogonal_direction
        self.last_heading_delta = orthogonal_direction
        return orthogonal_direction


class ExploreState(State):
    def __init__(self, force: bool=False):
        self.force = force

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        if health <= self.health_cutoff and not self.force:
            return SelectTargetState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        target_location = Location(tem_model.get_curiosity_target(location.location).detach(), None)
        target = Target(location, target_location, location_model)
        return GoToState(target).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This state always transitions to another state, so it should not be asked to get an action")


class SelectTargetState(State):
    health_cutoff: float = 0.6

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        if health > self.health_cutoff:
            return ExploreState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
    
        target = drive_manager.choose_hunger_target(location, tem_model)
        if target is None:
            return ExploreState(force=True).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
        
        target.update_location_model(location_model)
        return GoToState(target).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This state always transitions to another state, so it should not be asked to get an action")


class ExploitState(State):
    def __init__(self):
        self.last_health = 0.0
        self.heading = None

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        self.heading = heading
        last_health = self.last_health
        self.last_health = health
        if health > last_health:
            return self

        elif health < self.health_cutoff:
            return SelectTargetState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
        
        else: 
            return ExploreState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        position_delta = torch.zeros_like(self.heading)

        orthogonal_direction = self.get_orthogonal_direction(self.heading)
        new_heading = self.heading + 0.1 * orthogonal_direction
        new_heading = new_heading / torch.norm(new_heading)

        return position_delta, new_heading


class GoToState(State):
    def __init__(self, target: Target):
        self.target = target
        self.action_encoder = None
        self.heading = None

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        self.target.update_current_location(location)

        if self.target.has_arrived():
            if self.health_cutoff > health:
                return ExploitState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            else:
                return ExploreState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        if distance is not None:
            valence = drive_manager.assess_valence(embedding)
            if valence > 0.0:
                return ApproachState(embedding, self).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            elif valence < 0.0 and distance < self.avoid_distance:
                return AvoidState(embedding, self).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
        
        self.action_encoder = action_encoder
        self.heading = heading
        return self
    
    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        position_delta = self.action_encoder(self.target.current_location.location, self.target.target_location.location)[0].detach()

        # spin as we move
        orthogonal_direction = self.get_orthogonal_direction(self.heading)
        new_heading = self.heading + 0.1 * orthogonal_direction
        new_heading = new_heading / torch.norm(new_heading)

        return position_delta, new_heading


class RespondState(State):
    embedding_diff_threshold: float = 0.1

    def __init__(self, sensory: torch.Tensor, return_to: State, valence: float=1.0):
        self.sensory = sensory
        self.return_to = return_to
        self.valence = valence
        self.heading = None

    def update(self, 
        tem_model: TEMModel, drive_manager: DriveManager,
        action_encoder: ActionEncoder,
        location: Location, object_location: Location,
        heading: torch.Tensor, distance: float, 
        health: float, embedding: torch.Tensor, 
        agent_location: torch.Tensor=None, obj_location: torch.Tensor=None,
        location_model: torch.nn.Module=None
    ) -> 'State':
        if self.valence > 0.0 and (distance is not None and distance < 5.0):
            return ExploitState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        if distance is None or (distance > 2 * self.avoid_distance and self.valence < 0.0):
            return self.return_to.update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        embedding_diff = torch.norm(embedding - self.sensory)
        if embedding_diff > self.embedding_diff_threshold:
            valence = drive_manager.assess_valence(embedding)
            if valence > 0.0:
                return ApproachState(embedding, self.return_to).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            elif valence < 0.0 and distance < self.avoid_distance:
                return AvoidState(embedding, self.return_to).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            else:
                return self.return_to.update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        self.heading = heading
        return self

    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.valence * self.heading.clone().detach(), self.heading


class ApproachState(RespondState):
    def __init__(self, sensory: torch.Tensor, return_location: Location):
        super().__init__(sensory, return_location, 1.0)


class AvoidState(RespondState):
    def __init__(self, sensory: torch.Tensor, return_location: Location):
        super().__init__(sensory, return_location, -1.0)


class TEMAgent(AgentModel):
    def __init__(self, sensory_embedding_dim: int, sensory_embedding_model: str, dim: int=2, can_see_fruit_distance: float=10.0, tem_model: TEMModel=None, action_encoder: TEMActionEncoder=None, drive_embedding_model: DriveEmbeddingClassifier=None, drive_keys: dict=None):
        super().__init__(sensory_embedding_dim, sensory_embedding_model, dim, can_see_fruit_distance)
        self.tem_model = tem_model
        self.action_encoder = action_encoder
        self.drive_embedding_model = drive_embedding_model
        self.drive_keys = drive_keys
        self.drive_manager = DriveManager(drive_embedding_model, drive_keys, tem_model)

        self.state = ExploreState()

        self.last_action = None
        self.last_location= None
        self.last_location_gen = None

        self.target_location = None
        self.target_location_sd = None
        self.target_location_estimated = None
        self.target_location_is_for_food = False
        self.target_sensory = None
        self.target_changed = False

        self.elbo = None
        self.log_prob = None

        self.tem_optimizer = torch.optim.AdamW(self.tem_model.parameters(), lr=0.001)

        self.location_dataset = []
        self.obj_location_dataset = []
        self.location_model = None

        self.total_movement = 0.0
        self.turns_with_a_target = 0
        self.turns_without_a_target = 0
        self.train_t = 0

    def reset(self):
        self.tem_model.reset()
        self.last_action = None
        self.last_location = None
        self.target_location = None
        self.elbo = None
        self.log_prob = None
        self.exploration_mode = True
        self.total_movement = 0.0
        self.train_t = 0
        self.turns_with_a_target = 0
        self.turns_without_a_target = 0
        self.target_location_estimated = None
        self.target_location_sd = None
        self.target_location_is_for_food = False
        self.target_sensory = None
        self.target_changed = False

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None):

        heading = heading.detach().clone().requires_grad_(True)
        self.target_changed = False

        if distance is None:
            embedding = torch.zeros(self.sensory_embedding_dim, requires_grad=True)
            distance_modified = None
        else:
            embedding = embedding.clone()
            embedding.requires_grad = True
            distance_modified = torch.empty(1, 1, device=embedding.device, dtype=embedding.dtype).fill_(distance / self.can_see_fruit_distance)

        # report the current sensory information to the TEM model to update the map
        location, location_sd, obj_loc_est, obj_loc_sd, elbo, log_prob, kl_next, kl_obj = self.tem_model(
            embedding[None,:], self.last_action, self.last_location, heading[None,:], distance_modified
        )
        self.train_t = 1 if self.elbo is None else self.train_t + 1
        self.elbo = torch.nan_to_num(elbo) if self.elbo is None else self.elbo + torch.nan_to_num(elbo)

        if self.last_location is not None:
            self.total_movement = self.total_movement + torch.norm(location - self.last_location)

        if distance is not None:
            self.log_prob = log_prob.item()
            print(f"Log prob: {self.log_prob:.3f} ELBO: {elbo.item():.3f} KL NEXT: {kl_next.item():.3f} KL OBJ: {kl_obj.item():.3f} SD: {location_sd.min().item():.3f} - {location_sd.max().item():.3f} OBJ SD: {obj_loc_sd.min().item():.3f} - {obj_loc_sd.max().item():.3f}\t\t\t\t\t", end="\r")
            sys.stdout.flush()

        self.location_dataset.append((location.detach().clone(), agent_location))
        if obj_location is not None:
            self.obj_location_dataset.append((obj_loc_est.detach().clone(), obj_location))
        
        self.last_location = location.detach()

        location = Location(location.detach(), location_sd.detach())
        if obj_loc_est is not None:
            obj_loc = Location(obj_loc_est.detach(), obj_loc_sd.detach())
        else:
            obj_loc = None

        self.state = self.state.update(
            self.tem_model, self.drive_manager, self.action_encoder, location, obj_loc, 
            heading, distance, health, embedding, agent_location, obj_location, self.predict_location
        )

        position_delta, new_heading = self.state.get_action()
        
        # print(f"Action: {action.detach().cpu().numpy().tolist()}, ELBO: {self.elbo}")
        self.last_action = position_delta[None, :]

        # spin as we move
        # orthogonal_direction = self.get_orthogonal_direction(heading)
        # new_heading = heading + 0.1 * orthogonal_direction
        # new_heading = new_heading / torch.norm(new_heading)

        return position_delta, new_heading

    def train(self):
        """
        Train the TEM from the ELBO, and the action encoder from the TEM localizer
        """
        # print(f"Training TEM for {self.train_t} steps")
        # Maximize the ELBO -- but backward will optimizer will minimize the negative of the ELBO
        (-self.elbo).backward(retain_graph=True)

        # print the gradients of the tem model
        for name, param in self.tem_model.named_parameters():
            if param.grad is not None:
                # print(f"Gradient for {name}: {param.grad.shape} -- {param.grad.abs().max().item()}")
                param.grad.data = torch.nan_to_num(param.grad.data)
   
        torch.nn.utils.clip_grad_norm_(self.tem_model.parameters(), 1.0)
        self.tem_optimizer.step()
        self.tem_optimizer.zero_grad()
        self.elbo = None
        self.tem_model.break_training_graph()

        self.last_location = self.last_location.detach()
        self.last_action = self.last_action.detach()

        self.action_encoder.train_from_localizer(self.tem_model.encoder.localizer, training_batches=100)
        encoder_action_loss = self.action_encoder.test_on_localizer(self.tem_model.encoder.localizer)
        decoder_action_loss = self.action_encoder.test_on_localizer(self.tem_model.decoder.localizer)
        print(f"Encoder action loss: {encoder_action_loss:.3f}, Decoder action loss: {decoder_action_loss:.3f}")

        print(f"Total movement this epoch: {self.total_movement}; sensor data ratio: {len(self.location_dataset) / (len(self.location_dataset) + len(self.obj_location_dataset)):.3f}")
        
        # if len(self.location_dataset) > 0:
        #     mse = self.train_location_model(self.location_dataset)
        #     print(f"LOCATION model with {len(self.location_dataset)} samples, mse loss: {mse:.3f}")
        # else:
        #     print("No location dataset, skipping location model training")

        # if len(self.obj_location_dataset) > 0:
        #     mse = self.train_location_model(self.obj_location_dataset)
        #     print(f"OBJECT LOCATION model with {len(self.obj_location_dataset)} samples, mse loss: {mse:.3f}")
        # else:
        #     print("No obj location dataset, skipping obj location model training")
        
        if len(self.location_dataset) > 0 and len(self.obj_location_dataset) > 0:
            mse = self.train_location_model(self.location_dataset + self.obj_location_dataset)
            print(f"COMBINED location model with {len(self.location_dataset) + len(self.obj_location_dataset)} samples, mse loss: {mse:.3f}")
 
        self.location_dataset = self.location_dataset[-1000:]
        self.obj_location_dataset = self.obj_location_dataset[-1000:]
        self.total_movement = 0.0
    
    def train_location_model(self, dataset):
        x = torch.stack([p[0] for p in dataset]).squeeze()
        y = torch.stack([p[1] for p in dataset]).squeeze()

        if x.ndim != 2 or y.ndim != 2:
            print(f"Skipping location model training, x shape: {x.shape}, y shape: {y.shape}")
            return float("inf")

        # Standardize inputs and targets (per-dim); clamp std to avoid divide-by-zero
        x_mu, x_sigma = x.mean(0), x.std(0).clamp_min(1e-6)
        y_mu, y_sigma = y.mean(0), y.std(0).clamp_min(1e-6)
        x_n = (x - x_mu) / x_sigma
        y_n = (y - y_mu) / y_sigma

        # randomly separate into 90% training and 10% validation
        random_indices = torch.randperm(x.shape[0])
        num_train = int(x.shape[0] * 0.9)
        train_x_n = x_n[random_indices][:num_train]
        train_y_n = y_n[random_indices][:num_train]
        val_x_n = x_n[random_indices][num_train:]
        val_y = y[random_indices][num_train:]

        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, y.shape[1]),
        )

        # (Optional) initialize last-layer bias to 0 since y_n is zero-mean
        torch.nn.init.zeros_(model[-1].bias)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()

        for _ in range(500):
            opt.zero_grad()
            pred_n = model(train_x_n)
            loss = mse(pred_n, train_y_n)      # train in normalized space
            loss.backward()
            opt.step()

        # Report RMSE in world units
        with torch.no_grad():
            pred_n = model(val_x_n)
            pred = pred_n * y_sigma + y_mu            # de-normalize
            rmse_world = ((pred - val_y).pow(2).sum(dim=-1).mean()).sqrt().item()

        self.location_model = (model, x_mu, x_sigma, y_mu, y_sigma)

        return rmse_world
    
    def predict_location(self, location: torch.Tensor):
        if self.location_model is None:
            return None

        model, x_mu, x_sigma, y_mu, y_sigma = self.location_model
        location_n = (location - x_mu) / x_sigma
        return model(location_n[None,:]).squeeze() * y_sigma + y_mu

    
    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        tem_model = TEMModel.from_config(config)
        action_encoder = ActionEncoder.from_config(config)
        drive_embedding_model, drive_keys = train_drive_classifier(config)
        return cls(config.sensory_embedding_dim, config.sensory_embedding_model, config.dim, config.can_see_fruit_distance, 
                   tem_model, action_encoder, drive_embedding_model, drive_keys)

