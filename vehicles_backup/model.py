"""
Neural network model for learning control actions from vehicle trajectories.

This module defines a PyTorch neural network that takes a sequence of vehicle states
as input and outputs a sequence of control inputs (acceleration and steering angle).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


class TrajectoryToControlNet(nn.Module):
    """
    Neural network that maps vehicle trajectory states to control inputs.
    
    Input: Sequence of vehicle states [batch_size, seq_len, state_dim]
           where state_dim = 4 for [x, y, heading, velocity]
    
    Output: Sequence of control inputs [batch_size, seq_len, control_dim]
            where control_dim = 2 for [acceleration, angular_velocity]
    
    Architecture:
        - Input projection layer
        - Bidirectional LSTM encoder
        - Attention mechanism (optional)
        - Output projection layer
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        control_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        """
        Initialize the neural network.
        
        Args:
            state_dim: Dimension of state vector (default: 4 for [x, y, θ, v])
            control_dim: Dimension of control vector (default: 2 for [a, ω])
            hidden_dim: Hidden dimension for LSTM (default: 128)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            use_attention: Whether to use attention mechanism (default: True)
        """
        super(TrajectoryToControlNet, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Output projection layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, control_dim)
        )
        
    def forward(
        self, 
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            actual_trajectory: Actual trajectory states [batch_size, actual_seq_len, state_dim]
            current_state: Current predicted state [batch_size, state_dim] or [batch_size, 1, state_dim]
        
        Returns:
            controls: Predicted control input [batch_size, control_dim]
        """
        batch_size = actual_trajectory.shape[0]
        
        # Ensure current_state is [batch_size, state_dim]
        if len(current_state.shape) == 3:
            current_state = current_state.squeeze(1)  # [batch_size, state_dim]
        
        # Process actual trajectory through LSTM
        actual_proj = self.input_proj(actual_trajectory)  # [batch_size, actual_seq_len, hidden_dim]
        lstm_out, _ = self.lstm(actual_proj)  # [batch_size, actual_seq_len, hidden_dim * 2]
        
        # Attention on actual trajectory
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            actual_encoded = self.attention_norm(lstm_out + attn_out)  # [batch_size, actual_seq_len, hidden_dim * 2]
        else:
            actual_encoded = lstm_out
        
        # Take the last output from the actual trajectory encoding
        actual_context = actual_encoded[:, -1, :]  # [batch_size, hidden_dim * 2]
        
        # Process current state
        current_proj = self.input_proj(current_state.unsqueeze(1))  # [batch_size, 1, hidden_dim]
        current_encoded = current_proj.squeeze(1)  # [batch_size, hidden_dim]
        
        # Concatenate actual trajectory context with current state
        # Expand current_encoded to match hidden_dim * 2 by projecting
        current_expanded = torch.cat([current_encoded, current_encoded], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Combine actual trajectory context and current state
        combined = actual_context + current_expanded  # [batch_size, hidden_dim * 2]
        
        # Output projection (single control output)
        combined_expanded = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim * 2]
        controls = self.output_proj(combined_expanded)  # [batch_size, 1, control_dim]
        controls = controls.squeeze(1)  # [batch_size, control_dim]
        
        return controls


class SimpleTrajectoryToControlNet(nn.Module):
    """
    Simpler version without attention for faster training.
    
    Uses a simpler architecture with just LSTM and fully connected layers.
    """
    
    def __init__(
        self,
        state_dim: int = 4,
        control_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize the simple neural network.
        
        Args:
            state_dim: Dimension of state vector (default: 4)
            control_dim: Dimension of control vector (default: 2)
            hidden_dim: Hidden dimension for LSTM (default: 128)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
        """
        super(SimpleTrajectoryToControlNet, self).__init__()
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, control_dim)
        )
        
    def forward(
        self, 
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            actual_trajectory: Actual trajectory states [batch_size, actual_seq_len, state_dim]
            current_state: Current predicted state [batch_size, state_dim] or [batch_size, 1, state_dim]
        
        Returns:
            controls: Predicted control input [batch_size, control_dim]
        """
        batch_size = actual_trajectory.shape[0]
        
        # Ensure current_state is [batch_size, state_dim]
        if len(current_state.shape) == 3:
            current_state = current_state.squeeze(1)  # [batch_size, state_dim]
        
        # Process actual trajectory through LSTM
        actual_proj = self.input_proj(actual_trajectory)  # [batch_size, actual_seq_len, hidden_dim]
        lstm_out, _ = self.lstm(actual_proj)  # [batch_size, actual_seq_len, hidden_dim]
        
        # Take the last output from the actual trajectory encoding
        actual_context = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Process current state
        current_proj = self.input_proj(current_state.unsqueeze(1))  # [batch_size, 1, hidden_dim]
        current_encoded = current_proj.squeeze(1)  # [batch_size, hidden_dim]
        
        # Combine actual trajectory context and current state
        combined = actual_context + current_encoded  # [batch_size, hidden_dim]
        
        # Output projection (single control output)
        combined_expanded = combined.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        controls = self.output_proj(combined_expanded)  # [batch_size, 1, control_dim]
        controls = controls.squeeze(1)  # [batch_size, control_dim]
        
        return controls


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
    Actor network for RL algorithms with stochastic or deterministic policy.
    Outputs mean and log_std for actions.
    Can operate in deterministic mode (returns mean action) or stochastic mode (samples from distribution).
    """
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        deterministic: bool = False
    ):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.deterministic = deterministic
        
        # Process next downsampled state (augmented with time index)
        # Input will be state_dim + 1 (state + time_index)
        self.next_state_proj = nn.Linear(state_dim + 1, hidden_dim)
        
        # Process current state (augmented with time index)
        # Input will be state_dim + 1 (state + time_index)
        self.current_state_proj = nn.Linear(state_dim + 1, hidden_dim)
        
        # Combine next state context and current state
        # Input: next_state_encoded (hidden_dim) + current_state_encoded (hidden_dim) = 2*hidden_dim
        self.shared_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Initialize log_std_head to output reasonable initial variance
        # Initialize bias to -1.0 so initial std ≈ 0.37 (good for exploration)
        # Initialize weights to small values to prevent large initial variance
        nn.init.constant_(self.log_std_head.bias, -1.0)
        nn.init.normal_(self.log_std_head.weight, mean=0.0, std=0.01)
        
        # Log std bounds: prevent std from being too small (min std ≈ 0.01) or too large (max std ≈ 7.4)
        # This ensures the policy maintains some exploration while preventing extreme values
        self.log_std_min = -4.6  # exp(-4.6) ≈ 0.01 (minimum reasonable std)
        self.log_std_max = 2.0   # exp(2.0) ≈ 7.4 (maximum reasonable std)
    
    def forward(
        self,
        next_downsampled_state: torch.Tensor,
        next_time_index: torch.Tensor,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            next_downsampled_state: [batch_size, state_dim] next downsampled state to reach
            next_time_index: [batch_size] time index of next downsampled state
            current_state: [batch_size, state_dim] or [state_dim] (single state without batch)
            current_time_index: [batch_size] or scalar time index of current state
                              If None, uses 0 for all
        
        Returns:
            mean: [batch_size, action_dim] or [action_dim] (if input was single state)
            log_std: [batch_size, action_dim] or [action_dim] (if input was single state)
        """
        device = next_downsampled_state.device
        batch_size = next_downsampled_state.shape[0]
        
        # Normalize current_state to have batch dimension
        current_state_has_batch = len(current_state.shape) == 2
        current_state = current_state.unsqueeze(0) if not current_state_has_batch else current_state
        batch_size_current = current_state.shape[0]
        
        # Normalize next_time_index to have batch dimension
        if next_time_index.dim() == 0:
            next_time_index = next_time_index.unsqueeze(0).expand(batch_size_current)
        elif next_time_index.shape[0] != batch_size_current:
            next_time_index = next_time_index.expand(batch_size_current)
        
        # Normalize current_time_index
        if current_time_index is None:
            current_time_index = torch.zeros(batch_size_current, device=device, dtype=torch.float32)
        elif current_time_index.dim() == 0:
            current_time_index = current_time_index.unsqueeze(0).expand(batch_size_current).float()
        elif current_time_index.shape[0] != batch_size_current:
            current_time_index = current_time_index.expand(batch_size_current).float()
        else:
            current_time_index = current_time_index.float()
        
        # Normalize time indices: use max of next_time_index as normalization factor
        max_time_idx = next_time_index.max().item() if next_time_index.numel() > 0 else 1.0
        if max_time_idx > 0:
            next_time_index_norm = next_time_index / max_time_idx
            current_time_index_norm = current_time_index / max_time_idx
        else:
            next_time_index_norm = next_time_index
            current_time_index_norm = current_time_index
        
        # Expand next_downsampled_state to match current_state batch size if needed
        if batch_size_current != batch_size:
            if batch_size_current == 1:
                next_downsampled_state = next_downsampled_state[0:1]
                next_time_index_norm = next_time_index_norm[0:1]
            else:
                next_downsampled_state = next_downsampled_state.expand(batch_size_current, -1)
                next_time_index_norm = next_time_index_norm.expand(batch_size_current)
        
        # Concatenate time index to next downsampled state: [batch_size_current, state_dim + 1]
        next_state_augmented = torch.cat([next_downsampled_state, next_time_index_norm.unsqueeze(-1)], dim=-1)
        
        # Process next downsampled state
        next_state_encoded = self.next_state_proj(next_state_augmented)  # [batch_size_current, hidden_dim]
        
        # Concatenate time index to current state: [batch_size_current, state_dim + 1]
        current_state_augmented = torch.cat([current_state, current_time_index_norm.unsqueeze(-1)], dim=-1)
        
        # Process current state
        current_state_encoded = self.current_state_proj(current_state_augmented)  # [batch_size_current, hidden_dim]
        
        # Concatenate next state context and current state
        combined = torch.cat([next_state_encoded, current_state_encoded], dim=1)  # [batch_size_current, hidden_dim * 2]
        
        # Shared network
        shared = self.shared_net(combined)  # [batch_size_current, hidden_dim // 2]
        
        # Mean and log_std
        mean = self.mean_head(shared)  # [batch_size_current, action_dim]
        log_std = self.log_std_head(shared)  # [batch_size_current, action_dim]
        
        # Remove batch dimension if it was added
        if not current_state_has_batch and batch_size_current == 1:
            mean = mean.squeeze(0)  # [action_dim]
            log_std = log_std.squeeze(0)  # [action_dim]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        # Clip mean to prevent extreme values and NaN
        # Use reasonable bounds for control actions (acceleration and angular velocity)
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        
        # Replace any NaN values with zeros
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        log_std = torch.where(torch.isnan(log_std), torch.zeros_like(log_std), log_std)
        
        return mean, log_std 
    
    def sample(
        self,
        next_downsampled_state: torch.Tensor,
        next_time_index: torch.Tensor,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        If deterministic=True, returns the mean action directly without sampling.
        Otherwise, samples from the normal distribution.
        
        Args:
            next_downsampled_state: [batch_size, state_dim] next downsampled state to reach
            next_time_index: [batch_size] time index of next downsampled state
            current_state: [batch_size, state_dim]
            current_time_index: [batch_size] time index of current state
        
        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size]
            mean: [batch_size, action_dim]
        """
        mean, log_std = self.forward(next_downsampled_state, next_time_index, current_state, current_time_index)
        
        # Normalize mean and log_std to have batch dimension for consistent processing
        from utils import ensure_batch_dim
        mean_has_batch = len(mean.shape) == 2
        mean = ensure_batch_dim(mean, 1) if not mean_has_batch else mean
        log_std = ensure_batch_dim(log_std, 1) if not mean_has_batch else log_std
        
        # Ensure mean is valid
        mean = torch.clamp(torch.where(torch.isnan(mean), torch.zeros_like(mean), mean), min=-10.0, max=10.0)
        
        # Sample action from policy
        if self.deterministic:
            action = mean
            tiny_std = torch.ones_like(mean) * 1e-8
            log_prob = Normal(mean, tiny_std).log_prob(action).sum(dim=1)
        else:
            std = torch.clamp(torch.where(torch.isnan(torch.exp(log_std)), torch.ones_like(log_std) * 0.1, torch.exp(log_std)), min=1e-6, max=10.0)
            try:
                normal = Normal(mean, std)
                action = normal.rsample()
                # Clamp actions such that acceleration (assumed index 1) >= 0.0001,
                # and steering (assumed index 0) in [-pi/2, pi/2]
                action_clamped = action.clone()
                action_clamped[:, 0] = torch.clamp(action[:, 0], min=-torch.pi/3, max=torch.pi/3)  # steering
                action_clamped[:, 1] = torch.clamp(action[:, 1], min=-10, max=10.0)            # acceleration
                action = action_clamped
                log_prob = torch.where(torch.isnan(normal.log_prob(action).sum(dim=1)), torch.full((mean.shape[0],), -1e6, device=mean.device), normal.log_prob(action).sum(dim=1))
            except Exception as e:
                print(f"Warning: Sampling failed, using mean: {e}")
                action, log_prob = mean, torch.zeros(mean.shape[0], device=mean.device)
        
        # Remove batch dimension if it was added
        if not mean_has_batch:
            action, log_prob, mean = action.squeeze(0), log_prob.squeeze(0), mean.squeeze(0)
        
        return action, log_prob, mean


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
        dropout: float = 0.2
    ):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Process next downsampled state (augmented with time index)
        # Input will be state_dim + 1 (state + time_index)
        self.next_state_proj = nn.Linear(state_dim + 1, hidden_dim)
        
        # Process current state (augmented with time index)
        # Input will be state_dim + 1 (state + time_index)
        self.current_state_proj = nn.Linear(state_dim + 1, hidden_dim)
        
        # Combine and output value
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        next_downsampled_state: torch.Tensor,
        next_time_index: torch.Tensor,
        current_state: torch.Tensor,
        current_time_index: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            next_downsampled_state: [batch_size, state_dim] next downsampled state to reach
            next_time_index: [batch_size] time index of next downsampled state
            current_state: [batch_size, state_dim]
            current_time_index: [batch_size] time index of current state
                              If None, uses 0 for all
        
        Returns:
            Value: [batch_size, 1]
        """
        device = next_downsampled_state.device
        batch_size = next_downsampled_state.shape[0]
        
        # Normalize time indices
        if next_time_index.dim() == 0:
            next_time_index = next_time_index.unsqueeze(0).expand(batch_size)
        elif next_time_index.shape[0] != batch_size:
            next_time_index = next_time_index.expand(batch_size)
        
        if current_time_index is None:
            current_time_index = torch.zeros(batch_size, device=device, dtype=torch.float32)
        elif current_time_index.dim() == 0:
            current_time_index = current_time_index.unsqueeze(0).expand(batch_size).float()
        elif current_time_index.shape[0] != batch_size:
            current_time_index = current_time_index.expand(batch_size).float()
        else:
            current_time_index = current_time_index.float()
        
        # Normalize time indices: use max of next_time_index as normalization factor
        max_time_idx = next_time_index.max().item() if next_time_index.numel() > 0 else 1.0
        if max_time_idx > 0:
            next_time_index_norm = next_time_index / max_time_idx
            current_time_index_norm = current_time_index / max_time_idx
        else:
            next_time_index_norm = next_time_index
            current_time_index_norm = current_time_index
        
        # Concatenate time index to next downsampled state: [batch_size, state_dim + 1]
        next_state_augmented = torch.cat([next_downsampled_state, next_time_index_norm.unsqueeze(-1)], dim=-1)
        
        # Process next downsampled state
        next_state_encoded = self.next_state_proj(next_state_augmented)  # [batch_size, hidden_dim]
        
        # Concatenate time index to current state: [batch_size, state_dim + 1]
        current_state_augmented = torch.cat([current_state, current_time_index_norm.unsqueeze(-1)], dim=-1)
        
        # Process current state
        current_state_encoded = self.current_state_proj(current_state_augmented)  # [batch_size, hidden_dim]
        
        # Concatenate next state context and current state
        combined = torch.cat([next_state_encoded, current_state_encoded], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Output value
        value = self.value_net(combined)  # [batch_size, 1]
        
        return value
