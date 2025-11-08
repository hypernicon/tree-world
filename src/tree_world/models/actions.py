import torch
from tree_world.models.tem import TEMLocalizer
from tree_world.simulation import TreeWorldConfig


class ActionEncoder(torch.nn.Module):
    def __init__(self, location_dim: int, action_dim: int, embed_dim: int, dropout: float=0.1):
        super().__init__()
        self.location_dim = location_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.action_model = torch.nn.Sequential(
            torch.nn.LayerNorm(location_dim),
            torch.nn.Linear(location_dim, embed_dim),
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, action_dim + 1),
        )

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        self.times_trained = 0

    def forward(self, location: torch.Tensor, target_location: torch.Tensor):
        delta = target_location - location
        action = self.action_model(delta)
        action = action[..., :action_dim]
        cost = 1 + torch.exp(action[..., action_dim])
        action = action / (torch.norm(action, dim=-1, keepdim=True) + 1e-6)
        return action, cost

    def reset_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.kaiming_normal_(param)


    def train_from_localizer(self, localizer: TEMLocalizer, training_batches: int=1000, batch_size: int=128, lookahead: int=15,
                             device: torch.device=torch.device('cpu')):
        """
        Train the action encoder from the localizer.
        """
        if self.times_trained % 1000 == 0:
            self.reset_weights()

        for i in range(training_batches):
            locations = torch.randn(batch_size, self.location_dim, device=device)
            actions = torch.randn(batch_size, self.action_dim, device=device)
            costs = torch.ones(batch_size, device=device)

            targets, _ = localizer(locations, actions, return_distribution=True)
            
            for i in range(lookahead):
                locations = torch.cat([locations, targets[-batch_size:]], dim=0)
                targets = torch.cat([targets, localizer(locations[-batch_size:], actions, return_distribution=True)[0]], dim=0)
                costs = torch.cat([costs, torch.full((batch_size,), i+2, device=device, dtype=locations.dtype)], dim=0)

            actions = actions.repeat(lookahead + 1, 1)

            action_guess, cost_guess = self(locations, targets.detach())

            loss = (action_guess - actions).pow(2).sum(dim=-1).mean()

            # note that costs is >= 1 so that log(costs) is positive; by construction, cost_guess is >= 1 as well.
            loss = loss + (torch.log(cost_guess / costs)).pow(2).mean()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        print()
        print(f"Action encoder trained ({self.times_trained} times) for {training_batches} batches, loss: {loss.item()}")

        self.times_trained += 1

    @torch.no_grad()
    def test_on_localizer(self, localizer: TEMLocalizer, batch_size: int=128, lookahead: int=7, device: torch.device=torch.device('cpu')):
        """
        Test the action encoder on the localizer.
        """
        locations = torch.randn(batch_size, self.location_dim, device=device)
        actions = torch.randn(batch_size, self.action_dim, device=device)

        targets, _ = localizer(locations, actions, return_distribution=True)
            
        for i in range(lookahead):
            locations = torch.cat([locations, targets[-batch_size:]], dim=0)
            targets = torch.cat([targets, localizer(locations[-batch_size:], actions, return_distribution=True)[0]], dim=0)

        actions = actions.repeat(lookahead + 1, 1)

        action_guess, _ = self(locations, targets.detach())

        loss = (action_guess - actions).pow(2).sum(dim=-1).mean()
        return loss.item()

    @classmethod
    def from_config(cls, config: 'TreeWorldConfig'):
        return cls(config.location_dim, config.dim, config.embed_dim, config.dropout)