import math
import torch

from tree_world.simulation import TreeWorldConfig
from tree_world.models.drives import DriveTargetProposer
from tree_world.models.actions import ActionEncoder
from tree_world.models.memory import BidirectionalMemory
from tree_world.models.curiosity import CuriositySampler
from tree_world.models.utils import TorchBlocker


class HomeostaticController(torch.nn.Module):
    def __init__(self, diagnostic_dim: int, location_dim: int, action_dim: int, sensory_dim: int, num_drives: int, 
                       memory: BidirectionalMemory, action_encoder: ActionEncoder, num_results: int=5, threshold: float=0.1, diversity_steps: int=5, 
                       dropout: float=0.1):
        super().__init__()
        self.diagnostic_dim = diagnostic_dim
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.sensory_dim = sensory_dim
        self.dropout = dropout

        self.num_results = num_results
        self.num_drives = num_drives

        self.memory = TorchBlocker(memory)

        # the extra "drive" is curiosity
        self.drive_value = torch.nn.Linear(diagnostic_dim, (num_drives + 1))

        self.cost_discount = torch.nn.Parameter(torch.tensor(1.0))

        self.heading_control = torch.nn.Sequential(
            torch.nn.Linear(3 + action_dim + 2 * location_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, action_dim),
        )

        self.heading_control_sd = torch.nn.Sequential(
            torch.nn.Linear(3 + action_dim + 2 * location_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, action_dim),
        )

        self.drive_target_proposer = DriveTargetProposer(
            location_dim, sensory_dim, num_drives, memory, num_results, threshold, diversity_steps, dropout
        )
        self.curiosity_sampler = CuriositySampler(location_dim)
        self.action_encoder = TorchBlocker(action_encoder)

    def propose_targets(self, batch_size: int):

        # drive targets are (batch_size, num_drives, num_results, location_dim)
        # drives are (batch_size, num_drives, num_results)
        # num_found is (batch_size, num_drives)
        drive_targets, target_sds, top_senses, num_found = self.drive_target_proposer()

        num_results = min(self.num_results, drive_targets.shape[2])

        # curiosity targets are (batch_size, num_results, location_dim)
        curiosity_targets = self.curiosity_sampler(batch_size, num_results)
        curiosity_senses = self.memory.module.read(curiosity_targets, torch.ones_like(curiosity_targets), detach=True)
        
        # concatenate the drive targets and the curiosity targets, but only keep the ones that were found
        # targets are (batch_size, (num_drives + 1), num_results, location_dim)
        targets = torch.cat([drive_targets, curiosity_targets[:, None, ...]], dim=1)
        target_sds = torch.cat([target_sds, torch.ones_like(target_sds[:, :1])], dim=1)
        senses = torch.cat([top_senses, curiosity_senses[:, None, ...]], dim=1)

        curiosity_scores = torch.cat([
            torch.zeros(batch_size, self.num_drives, num_results, device=curiosity_targets.device, dtype=curiosity_targets.dtype),
            torch.ones(batch_size, 1, num_results, device=curiosity_targets.device, dtype=curiosity_targets.dtype),
        ], dim=1)

        return targets, target_sds, senses, curiosity_scores

    def plan_action(self, location: torch.Tensor, targets: torch.Tensor):
        action, cost = self.action_encoder.module(location, targets)
        return action.detach(), cost.detach()
    
    def forward(self, diagnostics: torch.Tensor, heading: torch.Tensor, location: torch.Tensor, location_sd: torch.Tensor, 
                      has_current_target: torch.BoolTensor, current_target: torch.Tensor=None, current_target_sd: torch.Tensor=None, 
                      probabilistic: bool = False):
        batch_size = location.shape[0]

        drive_probs = torch.softmax(self.drive_value(diagnostics), dim=-1)

        # TODO: if there is a current target and drive status hasn't changed, then perhaps we should just keep moving towards it
        # the challenge is that we might learn that the target is not correct in some way (wrong location, wrong contents, wrong drive)

        # targets: (batch_size, (num_drives + 1), num_results, location_dim)
        targets, target_sds, senses, curiosity_scores = self.propose_targets(batch_size)

        targets = targets.view(batch_size, -1, self.location_dim)
        target_sds = target_sds.view(batch_size, -1, self.location_dim)
        senses = senses.view(batch_size, -1, self.sensory_dim)
        curiosity_scores = curiosity_scores.view(batch_size, -1)

        sensory_at_location = self.memory.module.read(location, location_sd, detach=True)

        if current_target is None:
            sensory_at_target = torch.zeros_like(sensory_at_location)
            current_target = torch.zeros_like(location)
            current_target_sd = torch.ones_like(location_sd)
        else:
            sensory_at_target = self.memory.module.read(current_target, torch.ones_like(current_target), detach=True)

        senses = torch.cat([senses, sensory_at_location[:, None, ...], sensory_at_target[:, None, ...]], dim=1)
        targets = torch.cat([targets, location[:, None, ...], current_target[:, None, ...]], dim=1)
        target_sds = torch.cat([target_sds, location_sd[:, None, ...], current_target_sd[:, None, ...]], dim=1)
        curiosity_scores = torch.cat([curiosity_scores, torch.zeros_like(curiosity_scores[:, :1]), torch.ones_like(curiosity_scores[:, :1])], dim=1)
        
        drive_affinity = self.drive_target_proposer.affinity_to_drives(senses)
        drive_affinity = torch.cat([
            drive_affinity,
            curiosity_scores.unsqueeze(-1),
        ], dim=-1)

        action, cost = self.plan_action(location, targets)

        # The current location NEEDS a cost to bias away from staying put
        cost[..., -2] = 5.0
        action[..., -2, :] = 0.0

        # now I have the following inputs ready:
        # drive_probs: (batch_size, num_drives + 1)
        # drive_affinity: (batch_size, num_targets, num_drives + 1)
        # targets: (batch_size, num_targets, location_dim)
        # senses: (batch_size, num_targets, sensory_dim)
        # action: (batch_size, num_targets, action_dim)
        # cost: (batch_size, num_targets)

        # Use the drive probabilities to score the targets based on the drive affinity
        # drive_score: (batch_size, num_targets)
        drive_score = torch.bmm(torch.relu(drive_affinity), drive_probs.unsqueeze(-1)).squeeze(-1)

        # TODO: fix the cost discount to be a more complicated function
        # TODO: perhaps discount cost differently for curiosity targets? -- but drive_probs can account for that
        cost_discount = torch.nn.functional.softplus(self.cost_discount) 
        adjusted_score = drive_score + cost_discount * (1/1 + cost)

        target_logits = torch.log_softmax(adjusted_score, dim=-1)
        target_probs = torch.exp(target_logits)

        # print(f"Target probs: {torch.exp(target_logits).detach().cpu().numpy().tolist()}")

        # keep the current target if it is still somewhat likely (more than half what even odds would give it)
        keep_current_target = has_current_target & (target_probs[..., -1] > 0.5 * (1. / target_probs.shape[-1]))

        if probabilistic:
            final_target = torch.multinomial(torch.exp(target_logits), 1).squeeze(-1)

        else:
            final_target = torch.argmax(adjusted_score, dim=-1)

        final_target = torch.where(~keep_current_target, final_target, adjusted_score.shape[-1] - 1)

        target = targets.gather(1, final_target.unsqueeze(-1).expand(batch_size, 1, targets.shape[-1])).squeeze(1)
        target_sd = target_sds.gather(1, final_target.unsqueeze(-1).expand(batch_size, 1, target_sds.shape[-1])).squeeze(1)

        keep_current_target = keep_current_target | (final_target == adjusted_score.shape[-1] - 1)
        staying_at_current_location = final_target == adjusted_score.shape[-1] - 2
        choosing_new_target = ~(keep_current_target | staying_at_current_location)
        
        movement_log_probs = target_logits.gather(1, final_target.unsqueeze(-1).expand(batch_size, 1)).mean()

        movement_action = action.gather(1, final_target.unsqueeze(-1).expand(batch_size, 1, action.shape[-1])).squeeze(1)

        # I also need to choose among the following heading options:
        # 1. look towards the selected target
        # 2. look towards some other target
        # 3. keep looking in the same direction

        # we'll do it based on 
        # (1) whether we keep the current target, 
        # (2) whether we are staying put,
        # (3) whether we are choosing a new target,
        # (4) the current heading
        # (5) the current location
        # (6) the current location standard deviation
        heading_input = torch.cat([
            keep_current_target.unsqueeze(-1).to(heading.dtype),
            staying_at_current_location.unsqueeze(-1).to(heading.dtype),
            choosing_new_target.unsqueeze(-1).to(heading.dtype),
            heading,
            location,
            location_sd,
        ], dim=-1)

        heading_action_mean = self.heading_control(heading_input)
        heading_action_sd = self.heading_control_sd(heading_input)

        heading_action_mean = 0.0 * heading_action_mean + movement_action.detach()

        if probabilistic:
            heading_action = heading_action_mean + torch.randn_like(heading_action_mean) * heading_action_sd
        else:
            heading_action = heading_action_mean

        heading_log_probs = (
            -0.5 * ((heading_action - heading_action_mean) / heading_action_sd).pow(2).sum(dim=-1).mean() 
            - torch.log(heading_action_sd).sum(dim=-1).mean() 
            - 0.5 * math.log(2 * math.pi) * heading_action_sd.shape[-1]
        )

        # heading_action is a real vector, and we need a unit vector
        heading_action = heading_action / torch.norm(heading_action, dim=-1, keepdim=True)


        log_probs = movement_log_probs + heading_log_probs
        return movement_action, heading_action, target, target_sd, log_probs

