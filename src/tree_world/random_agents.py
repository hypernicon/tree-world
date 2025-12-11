import sys
import math
from typing import Optional

import torch

from tree_world.simulation import AgentModel
from tree_world.models.tem_t import TemLocalizer, TemTransformerLayer


class Pruner(torch.nn.Module):
    def __init__(self, localizer, num_slots, location_dim, sensory_dim):
        super().__init__()
        self.localizer = [localizer] # Hide it from the graph

        self.location_prefix = torch.nn.Parameter(torch.empty(1, num_slots, location_dim).uniform_(-1, 1))
        self.sensory_prefix = torch.nn.Parameter(torch.randn(1, num_slots, sensory_dim))
        self.sensory_key_prefix = torch.nn.Parameter(torch.randn(1, num_slots, sensory_dim))

    def forward(self, locations, sensory):
        assert self.location_prefix.shape[1] == locations.shape[1]
        sensory_out = self.localizer[0].sensory_predictor(locations, self.location_prefix, self.sensory_prefix)
        sensory_with_location = sensory + self.localizer[0].position_encoder(locations)
        location_out = self.localizer[0].location_refiner(sensory_with_location, self.sensory_key_prefix, self.sensory)

        sensory_error = (sensory - sensory_out).pow(2).sum(dim=-1).mean()
        location_error = (location - location_out).pow(2).sum(dim=-1).mean()
        
        return sensory_error + location_error



class RandomTemTAgent(AgentModel):
    def __init__(
        self, 
        tem_model: TemLocalizer=None,
        step_size: float=5.0,
        lmbda: float=1.0,
        dim: int=2,
        context_window: int=1024
    ):
        self.t = 0

        self.tem = tem_model
        self.last_location = None
        self.last_action = None
        self.last_sensory = None

        self.location_history = []
        self.actual_location_history = []

        self.loss = []
        self.loc_loss = []
        self.sens_loss = []

        self.location_prefix = None
        self.sensory_prefix = None
        self.sensory_key_prefix = None

        self.step_size = step_size

        self.lmbda = lmbda
        self.dim = dim

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Moving TEM-t model to cuda")
            self.dtype = torch.bfloat16
            self.tem.to("cuda")
            self.tem.to(self.dtype)

        self.optimizer = torch.optim.AdamW(self.tem.parameters(), lr=1e-3)

        self.context_window = context_window

    def reset(self):
        self.t = 0
        self.location_history = []
        self.actual_location_history = []
        self.loss = []
        self.loc_loss = []
        self.sens_loss = []

        self.last_location = None
        self.last_action = None
        self.last_sensory = None

        self.location_prefix = None
        self.sensory_prefix = None
        self.sensory_key_prefix = None

        if self.use_cuda:
            torch.cuda.empty_cache()

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None, reward: float=0.0):

        if self.last_action is not None:
            last_action = torch.stack(self.last_action, dim=1).requires_grad_().detach()
        else:
            last_action = None

        if self.use_cuda:
            embedding = embedding.to("cuda").to(self.dtype)

        if self.last_sensory is not None:
            self.last_sensory = torch.cat([
                    self.last_sensory.detach(),
                    embedding[None, None, :]
            ], dim=1).requires_grad_().detach()
        else:
            self.last_sensory = embedding[None, None, :].requires_grad_().detach()

        if self.use_cuda:
            # self.last_sensory = self.last_sensory.to("cuda").to(self.dtype)
            # if self.last_location is not None:
            #     self.last_location = self.last_location.to("cuda").to(self.dtype)
            if last_action is not None:
                last_action = last_action.to("cuda").to(self.dtype)

        next_location, sensory_location, sensory_predicted, sensory_keys, sensory_error, location_disagreement = (
            self.tem(
                self.last_sensory, self.last_location, last_action, 
                location_prefix=self.location_prefix, sensory_prefix=self.sensory_prefix, sensory_key_prefix=self.sensory_key_prefix
            )
        )

        if torch.isnan(next_location).any() or torch.isnan(sensory_location).any() or torch.isnan(sensory_predicted).any():
            print(f"NaN detected at t={self.t}")
            print(f"last_location: {torch.argmax(torch.isnan(self.last_location).cpu().float(), dim=1)}")
            print(f"last_action: {self.last_action}")
            print(f"last_sensory: {torch.argmax(torch.isnan(self.last_sensory).cpu().float(), dim=1)}")
            print(f"next_location: {torch.argmax(torch.isnan(next_location).cpu().float(), dim=1)}")
            print(f"sensory_location: {torch.argmax(torch.isnan(sensory_location).cpu().float(), dim=1)}")
            print(f"sensory_predicted: {torch.argmax(torch.isnan(sensory_predicted).cpu().float(), dim=1)}")
            raise ValueError("NaN detected")
        
        # location_belief, loss = self.tem(self.last_location, self.last_action, embedding[None, :])
        self.last_location = next_location.detach().requires_grad_()
        self.location_history = sensory_location.detach()
        self.actual_location_history.append(agent_location)
        self.loss.append(sensory_error + self.lmbda * location_disagreement)
        self.loc_loss.append(location_disagreement)
        self.sens_loss.append(sensory_error)

        position_delta = torch.randn(self.dim) * self.step_size

        new_heading = position_delta / torch.norm(position_delta)

        if self.last_action is None:
            self.last_action = [position_delta.clone()[None, :]]
        else:
            self.last_action.append(position_delta.clone()[None, :])

        self.t = self.t + 1

        return position_delta, new_heading

    def train(self, epoch: int=None):
        print(f"Epoch {epoch} Step {self.t}: Taking an optimizer step with {len(self.loss)} loss values: {math.sqrt(sum(self.loss) / len(self.loss))}", end="")
        print(f" loc_loss: {math.sqrt(sum(self.loc_loss) / len(self.loc_loss))} sens_loss: {math.sqrt(sum(self.sens_loss) / len(self.sens_loss))}")
        sys.stdout.flush()
        self.optimizer.zero_grad()
        (sum(self.loss) / len(self.loss)).backward()
        torch.nn.utils.clip_grad_norm_(self.tem.parameters(), 1.0)
        self.optimizer.step()
        self.loss = []
        self.loc_loss = []
        self.sens_loss = []

        if self.last_location.shape[1] > self.context_window:
            self.prune()

        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def prune(self, steps=1000):
        pruner = Pruner(self.tem, self.context_window, self.tem.location_dim, self.tem.sensory_dim)
        opt = torch.optim.Adam(pruner.parameters(), lr=1e-3)

        assert len([p for p in pruner.parameters()]) == 3

        T = self.last_location.shape[1]

        for i in range(steps):
            indices = torch.randperm(T-1, device=self.last_location.device)[:self.context_window]
            locations = self.last_location[0][indices]
            sensory = self.last_sensory[0][indices]

            opt.zero_grad()
            loss = pruner(locations, sensory)
            loss.backward()
            opt.step()

            if i % step == 0:
                print(f"PRUNING: loss {loss.item()}", end="\r")
                sys.stdout.flush()
        
        print(f"PRUNED: final loss {loss.item()}")
        self.location_prefix = pruner.location_prefix.data.detach().clone()
        self.sensory_prefix = pruner.sensory_prefix.data.detach().clone()
        self.sensory_key_prefix = pruner.sensory_key_prefix.data.detach().clone()

        self.last_location = self.last_location[:, -1:]
        self.last_sensory = self.sensory[:, -1:]
        self.last_action = self.last_action[-1:]
    
    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        tem_model = TemLocalizer.from_config(config)
        return cls(tem_model, dim=config.dim)

