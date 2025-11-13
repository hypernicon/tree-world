import torch
from typing import Tuple
from functools import wraps

from tree_world.models.tem import TEMModel
from tree_world.models.actions import ActionEncoder
from tree_world.models.drives import DriveEmbeddingClassifier


def _monitor(fn):
    fn_name = fn.__name__

    @wraps(fn)
    def __monitored_fn(self, *args, **kwargs):
        
        self.event(f"before.{fn_name}", *args, **kwargs)
        result = fn(self, *args, **kwargs)
        self.event(f"after.{fn_name}", result, *args, **kwargs)

        return result

    return __monitored_fn


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

        self.arrival_count = 0
        self.arrival_check_count = 0

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

        result = torch.norm(z) < self.arrive_z_threshold

        self.arrival_check_count = self.arrival_check_count + 1
        if result:
            self.arrival_count = self.arrival_count + 1

        return result


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
    approach_distance: float = 25.0
    health_cutoff: float = 0.5
    last_heading_delta: torch.Tensor = None

    def event(self, event_name, *args, **kwargs):
        # print(f"Event {event_name} called with args: {args} and kwargs: {kwargs}")
        pass

    @_monitor
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

    @_monitor
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

    @_monitor
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

    @_monitor
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
        self.location_model = None

    def event(self, event_name, *args, **kwargs):
        event_name, fn_name = event_name.split(".")
        if fn_name == "update" and event_name == "after":
            result = args[0]
            if result is not self:
                distance_at_start = torch.norm(self.target.start_location.location - self.target.target_location.location)
                distance_at_end = torch.norm(self.target.target_location.location - self.target.current_location.location)
                print(f"Distance at start: {distance_at_start:.3f} -> Distance at end: {distance_at_end:.3f} :: progress {(distance_at_end - distance_at_start):.3f}", end="\t")
                if self.target.arrival_count > 0:
                    print(f"Reached target: {self.target.arrival_count > 0} after {self.target.arrival_check_count} steps; transitioned to {result.__class__.__name__}")
                else:
                    print(f"Failed to reach target after {self.target.arrival_check_count} steps; transitioned to {result.__class__.__name__}")
        if fn_name == "update" and event_name == "before":
            if self.target.arrival_check_count == 0:
                distance_at_start = torch.norm(self.target.start_location.location - self.target.target_location.location)
                if self.location_model is not None and self.location_model(self.target.start_location.location) is not None:
                    start_location = self.location_model(self.target.start_location.location)
                    target_location = self.location_model(self.target.target_location.location)
                    print(f"Starting to go to target at distance {distance_at_start:.3f}: {start_location.detach().cpu().numpy().tolist()} -> {target_location.detach().cpu().numpy().tolist()}")
                else:
                    start_location_norm = torch.norm(self.target.start_location.location)
                    target_location_norm = torch.norm(self.target.target_location.location)
                    print(f"Starting to go to target at distance {distance_at_start:.3f} -- NORMS: start {start_location_norm:.3f} target {target_location_norm:.3f}")
            else:
                return
                if self.location_model is not None and self.location_model(self.target.start_location.location) is not None:
                    start_location = self.location_model(self.target.start_location.location)
                    target_location = self.location_model(self.target.target_location.location)
                    current_location = self.location_model(self.target.current_location.location)
                    distance_at_start = torch.norm(self.target.start_location.location - self.target.target_location.location)
                    distance_now = torch.norm(self.target.current_location.location - self.target.target_location.location)
                    distance_delta = torch.norm(self.target.current_location.location - self.target.start_location.location)
                    print(f"GOTO in progress, distance (loc mod) at start: {distance_at_start:.3f} -> Distance now: {distance_now:.3f} :: progress {(distance_at_start - distance_now):.3f} :: traveled {distance_delta:.3f} :: {start_location.detach().cpu().numpy().tolist()} -> {target_location.detach().cpu().numpy().tolist()} at {current_location.detach().cpu().numpy().tolist()}")
                else:
                    distance_at_start = torch.norm(self.target.start_location.location - self.target.target_location.location)
                    distance_now = torch.norm(self.target.current_location.location - self.target.target_location.location)
                    distance_delta = torch.norm(self.target.current_location.location - self.target.start_location.location)
                    print(f"GOTO in progress, distance at start: {distance_at_start:.3f} -> Distance now: {distance_now:.3f} :: progress {(distance_at_start - distance_now):.3f} :: traveled {distance_delta:.3f}")

    @_monitor
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
            
        elif not isinstance(self.target, DriveTarget) and self.health_cutoff > health:
            return SelectTargetState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

        if distance is not None:
            valence = drive_manager.assess_valence(embedding)
            if valence > 0.0 and distance < self.approach_distance:
                return ApproachState(embedding, self).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            elif valence < 0.0 and distance < self.avoid_distance:
                return AvoidState(embedding, self).update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
        
        self.action_encoder = action_encoder
        self.heading = heading
        self.location_model = location_model
        return self
    
    def get_action(self) -> Tuple[torch.Tensor, torch.Tensor]:
        position_delta = self.action_encoder(self.target.current_location.location, self.target.target_location.location)[0].detach()
        position_delta = position_delta / torch.norm(position_delta)

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

    @_monitor
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
            return ExploreState().update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)
            #return self.return_to.update(tem_model, drive_manager, action_encoder, location, object_location, heading, distance, health, embedding, agent_location, obj_location, location_model)

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