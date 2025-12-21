import sys
import math
from typing import Optional

import torch

from tree_world.simulation import AgentModel
from tree_world.models.tem_t import TemLocalizer, TemTransformerLayer


class RandomTemTAgent(AgentModel):
    def __init__(
        self, 
        tem_model: TemLocalizer=None,
        step_size: float=5.0,
        lmbda: float=1.0,
        beta: float=1.0,
        dim: int=2,
        context_window: int=512,
        buffer: int=256
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

        self.step_size = step_size

        self.lmbda = lmbda
        self.beta = beta
        self.dim = dim

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("Moving TEM-t model to cuda")
            self.dtype = torch.bfloat16
            self.tem.to("cuda")
            self.tem.to(self.dtype)

        self.optimizer = torch.optim.AdamW(self.tem.parameters(), lr=1e-3)

        self.context_window = context_window
        self.buffer = buffer

        self.salience_scores = []
        self.rewards = []

        self.prefix_length = 0

    def reset(self):
        self.t = 0
        self.location_history = []
        self.actual_location_history = []
        self.loss = []
        self.loc_loss = []
        self.sens_loss = []
        self.displacement_loss = []

        self.last_location = None
        self.last_action = None
        self.last_sensory = None

        self.salience_scores = []
        self.rewards = []

        self.prefix_length = 0

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

        next_location, sensory_location, sensory_predicted, elbo, sensory_error, location_disagreement, displacement_loss = (
            self.tem(
                self.last_sensory, self.last_location, last_action, prefix_length=self.prefix_length
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
        self.loss.append(-elbo + self.beta * displacement_loss)
        self.loc_loss.append(location_disagreement)
        self.sens_loss.append(sensory_error)
        self.displacement_loss.append(displacement_loss)

        if torch.torch.is_tensor(sensory_error):
            sensory_error = sensory_error.item()

        self.salience_scores.append(0.001 * sensory_error + math.fabs(reward))
        self.rewards.append(reward)

        position_delta = torch.randn(self.dim) * self.step_size

        new_heading = position_delta / torch.norm(position_delta)

        if self.last_action is None:
            self.last_action = [position_delta.clone()[None, :]]
        else:
            self.last_action.append(position_delta.clone()[None, :])

        self.t = self.t + 1

        return position_delta, new_heading

    def train(self, epoch: int=None):
        loss_sum = sum(self.loss)
        print(f"Epoch {epoch} Step {self.t}: Taking an optimizer step with {len(self.loss)} loss values: {math.copysign(1.0, loss_sum) * math.sqrt(loss_sum.abs() / len(self.loss))}", end="")
        print(f" loc_loss: {math.sqrt(sum(self.loc_loss) / len(self.loc_loss))} sens_loss: {math.sqrt(sum(self.sens_loss).abs() / len(self.sens_loss))}", end="")
        print(f" displacement_loss: {math.sqrt(sum(self.displacement_loss).abs() / len(self.displacement_loss))}")
        sys.stdout.flush()
        self.optimizer.zero_grad()
        (sum(self.loss) / len(self.loss)).backward()
        torch.nn.utils.clip_grad_norm_(self.tem.parameters(), 1.0)
        self.optimizer.step()
        self.loss = []
        self.loc_loss = []
        self.sens_loss = []
        self.displacement_loss = []

        if self.last_location.shape[1] > self.context_window + self.buffer:
            self.prune()

        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def prune(self):
        T = self.last_location.shape[1]

        old_locations = self.last_location[0][:-self.buffer]
        old_sensory = self.last_sensory[0][:-self.buffer]
        old_salience_scores = torch.tensor(self.salience_scores[:-self.buffer], dtype=old_sensory.dtype, device=old_sensory.device)
        old_rewards = torch.tensor(self.rewards[:-self.buffer], dtype=old_sensory.dtype, device=old_sensory.device)
        old_actions = torch.stack(self.last_action[:-self.buffer], dim=1)[0].to(old_sensory.device).to(old_sensory.dtype)

        indices = torch.argsort(old_salience_scores, dim=0, descending=True)[:self.context_window]

        location_prefix = old_locations[indices].detach()
        sensory_prefix = old_sensory[indices].detach()
        salience_score_prefix = old_salience_scores[indices].detach()
        reward_prefix = old_rewards[indices].detach()
        action_prefix = old_actions[indices].detach()

        self.prefix_length = location_prefix.shape[0]

        self.last_location = torch.cat([location_prefix[None, ...], self.last_location[:, -self.buffer:]], dim=1)
        self.last_sensory = torch.cat([sensory_prefix[None, ...], self.last_sensory[:, -self.buffer:]], dim=1)
        self.last_action = [x[None, ...] for x in action_prefix] + self.last_action[-self.buffer:]
        self.salience_scores = [x for x in torch.cat([salience_score_prefix, self.salience_scores[-self.buffer:]], dim=0)]
        self.rewards = [x for x in torch.cat([reward_prefix, self.rewards[-self.buffer:]], dim=0)]
    
    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        tem_model = TemLocalizer.from_config(config)
        return cls(tem_model, dim=config.dim)

