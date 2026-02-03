"""
Neural network model for learning control actions from vehicle trajectories.

This module defines a PyTorch neural network that takes a sequence of vehicle states
as input and outputs a sequence of control inputs (acceleration and steering angle).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple




def create_model(
    model_type: str = "full",
    state_dim: int = 4,
    control_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ("full" or "simple")
        state_dim: Dimension of state vector
        control_dim: Dimension of control vector
        hidden_dim: Hidden dimension for LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    
    Returns:
        Initialized model
    """
    if model_type == "full":
        return TrajectoryToControlNet(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=True
        )
    elif model_type == "simple":
        return SimpleTrajectoryToControlNet(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'full' or 'simple'.")


class QNetwork(nn.Module):
    """
    Q-network (critic) for SAC.
    Estimates Q(s, a) where s includes actual trajectory and current state.
    """
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Process actual trajectory
        self.traj_proj = nn.Linear(state_dim, hidden_dim)
        self.traj_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Process current state
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        
        # Process action
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        
        # Combine and output Q-value
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            actual_trajectory: [batch_size, seq_len, state_dim]
            current_state: [batch_size, state_dim]
            action: [batch_size, action_dim]
        
        Returns:
            Q-value: [batch_size, 1]
        """
        batch_size = actual_trajectory.shape[0]
        
        # Process actual trajectory
        traj_proj = self.traj_proj(actual_trajectory)  # [batch, seq_len, hidden_dim]
        traj_lstm, _ = self.traj_lstm(traj_proj)  # [batch, seq_len, hidden_dim]
        traj_context = traj_lstm[:, -1, :]  # [batch, hidden_dim]
        
        # Process current state
        state_encoded = self.state_proj(current_state)  # [batch, hidden_dim]
        
        # Process action
        action_encoded = self.action_proj(action)  # [batch, hidden_dim]
        
        # Concatenate
        combined = torch.cat([traj_context, state_encoded, action_encoded], dim=1)  # [batch, hidden_dim * 3]
        
        # Q-value
        q_value = self.q_net(combined)  # [batch, 1]
        
        return q_value


class ActorNetwork(nn.Module):
    """
    Actor network for RL algorithms with discrete action space (e.g., LunarLander).
    Outputs logits for each discrete action.
    Can operate in deterministic mode (returns argmax action) or stochastic mode (samples from Categorical distribution).
    """
    def __init__(
        self,
        state_dim: int = 8,  # LunarLander has 8-dimensional state
        num_actions: int = 4,  # LunarLander has 4 discrete actions: 0=noop, 1=left, 2=main, 3=right
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        deterministic: bool = False,
        time_scale: float = 100.0
    ):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.deterministic = deterministic
        self.time_scale = float(time_scale)
        
        # Actor now only conditions on (current_state, current_time_index).
        # We augment the current state with a (scaled) time index: [state_dim, t_norm]
        self.state_time_proj = nn.Linear(state_dim + 1, hidden_dim)

        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Logits head for discrete actions
        self.logits_head = nn.Linear(hidden_dim // 2, num_actions)
        
        # Initialize logits head with small weights for stable training
        nn.init.orthogonal_(self.logits_head.weight, gain=0.01)
        nn.init.constant_(self.logits_head.bias, 0.0)
    
    def forward(
        self,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            current_state: [batch_size, state_dim] or [state_dim] (single state without batch)
            current_time_index: [batch_size] or scalar time index of current state
                              If None, uses 0 for all
        
        Returns:
            logits: [batch_size, num_actions] or [num_actions] (if input was single state)
        """
        device = current_state.device
        
        # Normalize current_state to have batch dimension
        current_state_has_batch = len(current_state.shape) == 2
        current_state = current_state.unsqueeze(0) if not current_state_has_batch else current_state
        batch_size_current = current_state.shape[0]
        
        # Normalize current_time_index
        if current_time_index is None:
            current_time_index = torch.zeros(batch_size_current, device=device, dtype=torch.float32)
        elif current_time_index.dim() == 0:
            current_time_index = current_time_index.unsqueeze(0).expand(batch_size_current).float()
        elif current_time_index.shape[0] != batch_size_current:
            current_time_index = current_time_index.expand(batch_size_current).float()
        else:
            current_time_index = current_time_index.float()
        
        # Scale time index to a roughly [0, 1] range (avoid batch-dependent normalization).
        # If your trajectories are ~90 steps, time_scale=100 is a reasonable default.
        t_norm = current_time_index / max(self.time_scale, 1.0)

        # Augment current state with time index: [batch, state_dim + 1]
        current_state_augmented = torch.cat([current_state, t_norm.unsqueeze(-1)], dim=-1)

        # Encode and predict
        encoded = self.state_time_proj(current_state_augmented)  # [batch, hidden_dim]
        shared = self.shared_net(encoded)  # [batch, hidden_dim // 2]
        
        # Logits for discrete actions
        logits = self.logits_head(shared)  # [batch_size_current, num_actions]
        
        # Remove batch dimension if it was added
        if not current_state_has_batch and batch_size_current == 1:
            logits = logits.squeeze(0)  # [num_actions]
        
        # Replace any NaN values with zeros
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        return logits 
    
    def sample(
        self,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        If deterministic=True, returns the argmax action (most likely action).
        Otherwise, samples from the Categorical distribution.
        
        Args:
            current_state: [batch_size, state_dim] or [state_dim]
            current_time_index: [batch_size] or scalar time index of current state
        
        Returns:
            action: [batch_size] discrete action indices (0 to num_actions-1)
            log_prob: [batch_size] log probability of the sampled action
            logits: [batch_size, num_actions] action logits (for compatibility)
        """
        logits = self.forward(current_state, current_time_index)
        
        # Normalize logits to have batch dimension for consistent processing
        from utils import ensure_batch_dim
        logits_has_batch = len(logits.shape) == 2
        logits = ensure_batch_dim(logits, 1) if not logits_has_batch else logits
        
        # Ensure logits are valid (replace NaN with zeros)
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        # Sample action from policy
        if self.deterministic:
            # Deterministic: return the action with highest probability
            action = logits.argmax(dim=1)  # [batch_size]
            # Compute log_prob for the selected action
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)  # [batch_size]
        else:
            # Stochastic: sample from Categorical distribution
            try:
                dist = Categorical(logits=logits)
                action = dist.sample()  # [batch_size]
                log_prob = dist.log_prob(action)  # [batch_size]
                
                # Replace any NaN log probabilities with a very negative value
                log_prob = torch.where(
                    torch.isnan(log_prob),
                    torch.full_like(log_prob, -1e6),
                    log_prob
                )
            except Exception as e:
                print(f"Warning: Sampling failed, using argmax: {e}")
                action = logits.argmax(dim=1)  # [batch_size]
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(action)  # [batch_size]
        
        # Remove batch dimension if it was added
        if not logits_has_batch:
            action = action.squeeze(0)  # [] -> scalar
            log_prob = log_prob.squeeze(0)  # [] -> scalar
            logits = logits.squeeze(0)  # [num_actions]
        
        return action, log_prob, logits
    
    def log_prob(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute log probability of given actions.
        
        Args:
            current_state: [batch_size, state_dim] or [state_dim]
            action: [batch_size] discrete action indices (0 to num_actions-1)
            current_time_index: [batch_size] or scalar time index of current state
        
        Returns:
            log_prob: [batch_size] log probability of the given actions
        """
        logits = self.forward(current_state, current_time_index)
        
        # Normalize logits to have batch dimension
        from utils import ensure_batch_dim
        logits_has_batch = len(logits.shape) == 2
        logits = ensure_batch_dim(logits, 1) if not logits_has_batch else logits
        
        # Normalize action to have batch dimension
        action_has_batch = len(action.shape) > 0
        action = action.unsqueeze(0) if not action_has_batch else action
        
        # Ensure logits are valid
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        # Compute log probability
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)  # [batch_size]
        
        # Replace any NaN log probabilities
        log_prob = torch.where(
            torch.isnan(log_prob),
            torch.full_like(log_prob, -1e6),
            log_prob
        )
        
        # Remove batch dimension if it was added
        if not logits_has_batch:
            log_prob = log_prob.squeeze(0)
        
        return log_prob


class ValueNetwork(nn.Module):
    """
    Value network (critic) for PPO.
    Estimates V(s) where s includes actual trajectory and current state.
    """
    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        time_scale: float = 100.0
    ):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.time_scale = float(time_scale)
        
        # Critic now only conditions on (current_state, current_time_index).
        self.state_time_proj = nn.Linear(state_dim + 1, hidden_dim)

        # Output value
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            current_state: [batch_size, state_dim]
            current_time_index: [batch_size] time index of current state
                              If None, uses 0 for all
        
        Returns:
            Value: [batch_size, 1]
        """
        # device = current_state.device
        # batch_size = current_state.shape[0]

        
  
        # if current_time_index is not None:
        # # Just validate it has correct shape to avoid any internal PyTorch indexing issues
        #     if current_time_index.dim() > 0 and current_time_index.shape[0] != batch_size:
        #         # Silently ignore shape mismatch - we don't use it anyway
        #         pass

        t_norm = current_time_index / max(self.time_scale, 1.0)
        current_state_augmented = torch.cat([current_state, t_norm.unsqueeze(-1)], dim=-1)  # [batch, state_dim+1]
        encoded = self.state_time_proj(current_state_augmented)  # [batch, hidden_dim]
        value = self.value_net(encoded)  # [batch, 1]
        
        return value
