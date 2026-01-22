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
        deterministic: bool = False,
        action_history_len: int = 20
    ):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.deterministic = deterministic
        self.action_history_len = action_history_len
        
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
        
        # Process action history with transformer
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # Positional encoding for transformer (learnable)
        self.action_pos_encoding = nn.Parameter(torch.randn(1, action_history_len, hidden_dim))
        
        # Transformer encoder for action history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.action_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Combine trajectory context, state, and action history
        # Input: traj_context (hidden_dim) + state_encoded (hidden_dim) + action_context (hidden_dim) = 3*hidden_dim
        self.shared_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(
        self,
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor,
        action_history: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            actual_trajectory: [batch_size, seq_len, state_dim]
            current_state: [batch_size, state_dim]
            action_history: [batch_size, action_history_len, action_dim] previous executed actions
                           If None, uses zeros (no action history)
        
        Returns:
            mean: [batch_size, action_dim]
            log_std: [batch_size, action_dim]
        """
        batch_size = actual_trajectory.shape[0]
        device = actual_trajectory.device
        
        # Process actual trajectory
        traj_proj = self.traj_proj(actual_trajectory)  # [batch, seq_len, hidden_dim]
        traj_lstm, _ = self.traj_lstm(traj_proj)  # [batch, seq_len, hidden_dim]
        traj_context = traj_lstm[:, -1, :]  # [batch, hidden_dim]
        
        # Process current state
        state_encoded = self.state_proj(current_state)  # [batch, hidden_dim]
        
        # Process action history with transformer
        if action_history is None:
            # Use zeros if no action history provided
            action_history = torch.zeros(batch_size, self.action_history_len, self.action_dim, device=device)
        else:
            # Ensure action_history has the right length (pad or truncate)
            action_seq_len = action_history.shape[1]
            if action_seq_len < self.action_history_len:
                # Pad with zeros at the beginning
                padding = torch.zeros(batch_size, self.action_history_len - action_seq_len, self.action_dim, device=device)
                action_history = torch.cat([padding, action_history], dim=1)  # [batch, action_history_len, action_dim]
            elif action_seq_len > self.action_history_len:
                # Take the most recent actions
                action_history = action_history[:, -self.action_history_len:, :]  # [batch, action_history_len, action_dim]
        
        # Project actions to hidden dimension
        action_embeddings = self.action_proj(action_history)  # [batch, action_history_len, hidden_dim]
        
        # Add positional encoding (learnable)
        # self.action_pos_encoding: [1, action_history_len, hidden_dim]
        action_embeddings = action_embeddings + self.action_pos_encoding  # [batch, action_history_len, hidden_dim]
        
        # Process with transformer encoder
        action_encoded = self.action_transformer(action_embeddings)  # [batch, action_history_len, hidden_dim]
        
        # Take the last output (most recent action context)
        action_context = action_encoded[:, -1, :]  # [batch, hidden_dim]
        
        # Concatenate trajectory context, state, and action history context
        combined = torch.cat([traj_context, state_encoded, action_context], dim=1)  # [batch, hidden_dim * 3]
        
        # Shared network
        shared = self.shared_net(combined)  # [batch, hidden_dim // 2]
        
        # Mean and log_std
        mean = self.mean_head(shared)  # [batch, action_dim]
        log_std = self.log_std_head(shared)  # [batch, action_dim]
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
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor,
        action_history: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        If deterministic=True, returns the mean action directly without sampling.
        Otherwise, samples from the normal distribution.
        
        Args:
            actual_trajectory: [batch_size, seq_len, state_dim]
            current_state: [batch_size, state_dim]
            action_history: [batch_size, action_history_len, action_dim] previous executed actions
                           If None, uses zeros (no action history)
        
        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size]
            mean: [batch_size, action_dim]
        """
        mean, log_std = self.forward(actual_trajectory, current_state, action_history)
        
        # Ensure mean is valid
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        mean = torch.clamp(mean, min=-10.0, max=10.0)
        
        # Sample action from policy
        if self.deterministic:
            # Deterministic policy: return mean action directly
            action = mean
            
            # For log_prob in deterministic mode, use a very small std for numerical stability
            # This allows PPO to still compute ratios, but action is essentially deterministic
            tiny_std = torch.ones_like(mean) * 1e-8
            normal = Normal(mean, tiny_std)
            log_prob = normal.log_prob(action).sum(dim=1)  # [batch]
        else:
            # Stochastic policy: sample from normal distribution
            std = torch.exp(log_std)
            
            # Ensure std is valid (no NaN or inf)
            std = torch.clamp(std, min=1e-6, max=10.0)
            std = torch.where(torch.isnan(std), torch.ones_like(std) * 0.1, std)
            
            # Sample from normal distribution
            try:
                normal = Normal(mean, std)
                action = normal.rsample()  # Use rsample for reparameterization trick
                
                # Clip actions to reasonable bounds
                action = torch.clamp(action, min=-10.0, max=10.0)
                
                # Compute log probability
                log_prob = normal.log_prob(action)  # [batch, action_dim]
                log_prob = log_prob.sum(dim=1)  # [batch]
                
                # Replace NaN log probs with a large negative value
                log_prob = torch.where(torch.isnan(log_prob), torch.full_like(log_prob, -1e6), log_prob)
            except Exception as e:
                # Fallback: use mean as action if sampling fails
                print(f"Warning: Sampling failed, using mean: {e}")
                action = mean
                log_prob = torch.zeros(mean.shape[0], device=mean.device)
        
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
        actual_trajectory: torch.Tensor,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            actual_trajectory: [batch_size, seq_len, state_dim]
            current_state: [batch_size, state_dim]
        
        Returns:
            Value: [batch_size, 1]
        """
        # Process actual trajectory
        traj_proj = self.traj_proj(actual_trajectory)  # [batch, seq_len, hidden_dim]
        traj_lstm, _ = self.traj_lstm(traj_proj)  # [batch, seq_len, hidden_dim]
        traj_context = traj_lstm[:, -1, :]  # [batch, hidden_dim]
        
        # Process current state
        state_encoded = self.state_proj(current_state)  # [batch, hidden_dim]
        
        # Concatenate
        combined = torch.cat([traj_context, state_encoded], dim=1)  # [batch, hidden_dim * 2]
        
        # Value
        value = self.value_net(combined)  # [batch, 1]
        
        return value
