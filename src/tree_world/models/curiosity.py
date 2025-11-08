import torch

class CuriositySampler(torch.nn.Module):
    """
    Sample a target location for the agent to move to based on the curiosity of the agent.
    """
    def __init__(self, location_dim: int):
        super().__init__()
        self.location_dim = location_dim

    def forward(self, num_results: int, dtype: torch.dtype=torch.float32, device: torch.device=torch.device('cpu')):
        return torch.randn(self.batch_size, num_results, self.location_dim, dtype=dtype, device=device)

