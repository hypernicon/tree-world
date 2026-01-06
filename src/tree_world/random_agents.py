import sys
import math
from typing import Optional

import torch

from tree_world.simulation import AgentModel
from tree_world.models.tem_t import TemLocalizer, TemTransformerLayer


def assert_params_finite(model):
    for n, p in model.named_parameters():
        if p is None: 
            continue
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"Param blew up: {n}")


class RandomTemTAgent(AgentModel):
    def __init__(
        self, 
        tem_model: TemLocalizer=None,
        step_size: float=5.0,
        lmbda: float=1.0,
        beta: float=1.0,
        gamma: float=1.0,
        dim: int=2,
        context_window: int=256,
        buffer: int=64
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
        self.gamma = gamma
        self.dim = dim

        self.dataset = []

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
        self.data_frequency = 25
        torch.autograd.set_detect_anomaly(True)

    def reset(self):
        self.t = 0
        self.location_history = []
        self.actual_location_history = []
        self.loss = []
        self.loc_loss = []
        self.sens_loss = []
        self.displacement_loss = []

        self.dataset = []

        self.last_location = None
        self.last_action = None
        self.last_sensory = None

        self.salience_scores = []
        self.rewards = []

        self.prefix_length = 0

        if self.use_cuda:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None, reward: float=0.0):

        self.tem.eval()

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
        
        if self.t % self.data_frequency == self.data_frequency - 1:
            self.dataset.append((
                self.last_sensory.detach().clone(), 
                self.last_location.detach().clone(), 
                last_action.detach().clone() if last_action is not None else None, 
                self.prefix_length
            ))

        next_location, sensory_location, sensory_predicted, elbo, _, sensory_error, location_disagreement, displacement_loss = (
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
        # self.loss.append(-elbo + self.beta * displacement_loss)
        # self.loc_loss.append(location_disagreement)
        # self.sens_loss.append(sensory_error)
        # self.displacement_loss.append(displacement_loss)

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

    def rerun_for_training(self, epoch: int=None):
        dtype = self.dataset[0][0].dtype
        device = self.dataset[0][0].device

        sensory_lengths = torch.tensor([p[0].shape[1] for p in self.dataset], dtype=torch.long, device=device)
        location_lengths = torch.tensor([p[1].shape[1] for p in self.dataset], dtype=torch.long, device=device)
        action_lengths = torch.tensor([p[2].shape[1] for p in self.dataset], dtype=torch.long, device=device)

        sensory_context_length = sensory_lengths.max()
        location_context_length = location_lengths.max()
        action_context_length = action_lengths.max()

        sensory = torch.cat([
            torch.cat([p[0], torch.zeros(1, sensory_context_length - p[0].shape[1], p[0].shape[2], dtype=dtype, device=device)], dim=1)
            for p in self.dataset
        ], dim=0)
        locations = torch.cat([
            torch.cat([p[1], torch.zeros(1, location_context_length - p[1].shape[1], p[1].shape[2], dtype=dtype, device=device)], dim=1)
            for p in self.dataset
        ], dim=0)
        actions = torch.cat([
            torch.cat([p[2], torch.zeros(1, action_context_length - p[2].shape[1], p[2].shape[2], dtype=dtype, device=device)], dim=1)
            for p in self.dataset
        ], dim=0)

        prefix_lengths = torch.tensor([p[3] for p in self.dataset], dtype=torch.long, device=device)

        print(f"DATASET: Sensory: {sensory.shape} Locations: {locations.shape} Actions: {actions.shape} Prefix Lengths: {prefix_lengths.shape}")

        _, _, _, elbo, sensory_logprobs, sensory_error, location_disagreement, displacement_loss = (
            self.tem(
                sensory, locations, actions, prefix_length=prefix_lengths, batch_lengths=sensory_lengths, kl_weight=self.gamma
            )
        )

        loss = -elbo + self.beta * displacement_loss
        print(f"Epoch {epoch} Step {self.t}: Loss: {loss.item():.3f} ELBO: {elbo.item():.3f} Sensory Log Probs: {sensory_logprobs.item():.3f} ", end="")
        print(f"Displacement Loss: {displacement_loss.item():.3f} Sensory Error: {sensory_error.item():.3f} ", end="")
        print(f"Location Disagreement: {location_disagreement.item():.3f}")
        sys.stdout.flush()

        return loss, location_disagreement

    def print_location_comparison(self):
        _, implied_locations = self.tem.location_metric.interpret(self.last_location[0])
        actual_locations = torch.stack(self.actual_location_history, dim=0)
        for i in range(len(self.last_action)):
            print(f"Implied {implied_locations[-i-1].detach().cpu().float().numpy().tolist()}", end="\t")
            print(f"Actual {actual_locations[-i-1].detach().cpu().float().numpy().tolist()}", end="\t")
            print(f"Action {self.last_action[-i-1].detach().cpu().float().numpy().tolist()}")

    def train(self, epoch: int=None):

        self.tem.train()

        loss, location_disagreement = self.rerun_for_training(epoch=epoch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.tem.update_geometric_scale(location_disagreement.item())

        assert_params_finite(self.tem)

        self.dataset = []

        if self.last_location.shape[1] > self.context_window + self.buffer:
            self.prune()

        self.tem.reset()

        if self.use_cuda:
            torch.cuda.empty_cache()
    
    def prune(self):
        T = self.last_location.shape[1]

        old_locations = self.last_location[0][:-self.buffer]
        old_sensory = self.last_sensory[0][:-self.buffer]
        old_salience_scores = torch.tensor(self.salience_scores[:-self.buffer], dtype=old_sensory.dtype, device=old_sensory.device)
        old_rewards = torch.tensor(self.rewards[:-self.buffer], dtype=old_sensory.dtype, device=old_sensory.device)

        indices = torch.argsort(old_salience_scores, dim=0, descending=True)[:self.context_window]

        location_prefix = old_locations[indices].detach()
        sensory_prefix = old_sensory[indices].detach()
        salience_score_prefix = old_salience_scores[indices].detach()
        reward_prefix = old_rewards[indices].detach()

        self.prefix_length = location_prefix.shape[0]

        self.last_location = torch.cat([location_prefix[None, ...], self.last_location[:, -self.buffer:]], dim=1)
        self.last_sensory = torch.cat([sensory_prefix[None, ...], self.last_sensory[:, -self.buffer:]], dim=1)
        self.last_action = self.last_action[-self.buffer:]
        self.salience_scores = [x for x in salience_score_prefix] + self.salience_scores[-self.buffer:]
        self.rewards = [x for x in reward_prefix] + self.rewards[-self.buffer:]
    
    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        tem_model = TemLocalizer.from_config(config)
        return cls(tem_model, dim=config.dim)

