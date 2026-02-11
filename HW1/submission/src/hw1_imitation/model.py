"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """action_chunk: (batch, chunk_size, action_dim); flattened for MSE."""
        pred_action_chunk = self.model(state)  
        flattened_target = action_chunk.reshape(-1, self.chunk_size * self.action_dim)
        return nn.functional.mse_loss(pred_action_chunk, flattened_target)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.model(state).reshape(-1, self.chunk_size, self.action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128,128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.model = nn.Sequential(
            nn.Linear(state_dim+action_dim*chunk_size+1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], chunk_size * action_dim),
        )
    ### action input [batch,chunk_size,action_dim]
    def forward(self, state, action_chunk, timestep) -> torch.Tensor:
        action_chunk = action_chunk.reshape(state.shape[0],-1) ### [batch,chunk_size*action_dim]
        timestep = timestep.reshape(state.shape[0],-1) ### [batch,1]
        return self.model(torch.cat([state, action_chunk, timestep], dim=-1)) 

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor, ###[batch,chunk_size,action_dim]
    ) -> torch.Tensor:
        noise_action_chunk = torch.randn_like(action_chunk,device=state.device) ### [batch,chunk_size,action_dim]
        batch_size = state.shape[0]
        timestep = torch.rand(batch_size, 1, device=state.device, dtype=state.dtype) ### [batch,1]
        
        t = timestep.unsqueeze(-1) ### [batch,1,1]
        interpolated_action_chunk = t * action_chunk + (1 - t) * noise_action_chunk
        pred_interpolated_action_chunk = self.forward(state,interpolated_action_chunk,timestep) ## [batch,chunk_size*action_dim]
        action_vector = action_chunk-noise_action_chunk
        flow_matching_loss = nn.functional.mse_loss(pred_interpolated_action_chunk, action_vector.view(state.shape[0],-1))
        return flow_matching_loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        scaler = 1/num_steps
        noise_action_chunk = torch.randn(batch_size,self.chunk_size, self.action_dim,device=state.device) ### [batch,chunk_size,action_dim]
        for i in range(num_steps):
            
            timestep = i*scaler
            timestep_tensor = torch.full((batch_size,1),timestep,device=state.device)
            
            update = self.forward(state,noise_action_chunk,timestep_tensor)
            update = update.reshape(batch_size,self.chunk_size,self.action_dim)
            noise_action_chunk = noise_action_chunk + update*scaler
        
        return noise_action_chunk



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
