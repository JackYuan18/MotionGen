"""
Utility functions for reinforcement learning training.

This module provides shared utility functions used across different RL training scripts.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Time step for dynamics (seconds)
DT = 0.1

# Try to import torchdiffeq for continuous ODE solving
try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None


def simple_car_dynamics_torch(state: torch.Tensor, controls: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """
    Simple car dynamics in PyTorch: ẋ = v*cos(θ), ẏ = v*sin(θ), θ̇ = v*ω, v̇ = a
    (Matches the implementation in compare_trajectories_dynamics.py)
    
    Args:
        state: [batch_size, 4] tensor with [x, y, θ, v]
        controls: [batch_size, 2] tensor with [a, ω]
        dt: Time step (unused, kept for interface consistency)
    
    Returns:
        State derivative [batch_size, 4]
    """
    x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    a, omega = controls[:, 0], controls[:, 1]
    
    # Ensure non-negative velocity
    v = torch.clamp(v, min=0.0)
    
    # State derivatives (matching compare_trajectories_dynamics.py line 136)
    dx = v * torch.cos(theta)
    dy = v * torch.sin(theta)
    dtheta = v * omega  # Note: v*omega as in original implementation
    dv = a
    
    return torch.stack([dx, dy, dtheta, dv], dim=1)


def rk4_step(
    state: torch.Tensor,
    controls: torch.Tensor,
    dt: float,
    dynamics_func: callable
) -> torch.Tensor:
    """
    Perform one step of 4th order Runge-Kutta integration.
    
    Args:
        state: [batch_size, 4] current state
        controls: [batch_size, 2] control inputs
        dt: Time step
        dynamics_func: Function that computes state derivative (state, controls) -> state_derivative
    
    Returns:
        Next state [batch_size, 4]
    """
    k1 = dynamics_func(state, controls, dt)
    k2 = dynamics_func(state + dt * k1 / 2, controls, dt)
    k3 = dynamics_func(state + dt * k2 / 2, controls, dt)
    k4 = dynamics_func(state + dt * k3, controls, dt)
    
    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


def ensure_batch_dim(tensor: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """
    Ensure tensor has batch dimension. If 1D, add batch dimension.
    
    Args:
        tensor: Input tensor [state_dim] or [batch, state_dim]
        batch_size: Desired batch size (default: 1)
    
    Returns:
        Tensor with shape [batch_size, ...]
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] != batch_size:
        tensor = tensor.expand(batch_size, *tensor.shape[1:]) if tensor.shape[0] == 1 else tensor[:batch_size]
    return tensor


def normalize_time_indices(
    downsampled_indices: Optional[torch.Tensor],
    seq_len: int,
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, float]:
    """
    Normalize time indices, handling None, empty, and shape mismatches.
    
    Args:
        downsampled_indices: Time indices [batch, seq_len] or None
        seq_len: Sequence length
        batch_size: Batch size
        device: Device for tensor creation
    
    Returns:
        Tuple of (normalized_indices, max_time_idx)
    """
    # Create default sequential indices if None or empty
    if downsampled_indices is None or downsampled_indices.numel() == 0:
        indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        indices = indices.unsqueeze(0).expand(batch_size, -1)
    else:
        # Ensure float32 and correct batch size
        indices = downsampled_indices.float()
        if indices.shape[0] != batch_size:
            indices = indices.expand(batch_size, -1) if indices.shape[0] == 1 else indices[:batch_size]
    
    # Normalize to [0, 1] range
    max_time_idx = indices.max().item() if indices.numel() > 0 else 0.0
    indices_norm = indices / max_time_idx if max_time_idx > 0 else indices
    
    return indices_norm, max_time_idx


def prepare_time_index(
    current_time_index: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    max_time_idx: float = 1.0
) -> torch.Tensor:
    """
    Prepare and normalize current time index.
    
    Args:
        current_time_index: Time index tensor or None
        batch_size: Batch size
        device: Device for tensor creation
        max_time_idx: Maximum time index for normalization
    
    Returns:
        Normalized time index [batch_size]
    """
    if current_time_index is None:
        time_idx = torch.zeros(batch_size, device=device, dtype=torch.float32)
    else:
        time_idx = current_time_index.float()
        if time_idx.dim() == 0:
            time_idx = time_idx.unsqueeze(0).expand(batch_size)
        elif time_idx.shape[0] != batch_size:
            time_idx = time_idx[0:1].expand(batch_size) if time_idx.shape[0] > 0 else torch.zeros(batch_size, device=device, dtype=torch.float32)
    
    # Normalize
    return time_idx / max_time_idx if max_time_idx > 0 else time_idx


def ensure_tensor_shape(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Ensure tensor has target shape, reshaping/expanding as needed.
    
    Args:
        tensor: Input tensor
        target_shape: Desired shape
    
    Returns:
        Tensor with target shape
    """
    if tensor.shape == target_shape:
        return tensor
    
    # Handle dimension mismatches
    if tensor.dim() < len(target_shape):
        tensor = tensor.unsqueeze(0)
    
    # Handle batch size mismatches
    if tensor.shape[0] != target_shape[0]:
        if tensor.shape[0] == 1:
            tensor = tensor.expand(target_shape[0], *tensor.shape[1:])
        else:
            tensor = tensor[:target_shape[0]]
    
    # Handle feature dimension mismatches
    if len(target_shape) > 1 and tensor.shape[1:] != target_shape[1:]:
        if tensor.numel() == target_shape[0] * target_shape[1]:
            tensor = tensor.view(target_shape)
        else:
            tensor = tensor[:, :target_shape[1]]
    
    return tensor


class StateFeedbackODE(nn.Module):
    """
    ODE function wrapper for state-feedback control policy.
    
    This class implements the ODE dynamics where controls are computed from
    the actor policy based on the current state and actual trajectory.
    For torchdiffeq, states are single vectors [4], not batched.
    """
    
    def __init__(
        self,
        actor: nn.Module,
        actual_trajectory: torch.Tensor,
        control_update_dt: float = DT,
        downsampled_indices: torch.Tensor = None
    ):
        """
        Initialize the ODE function.
        
        Args:
            actor: ActorNetwork policy
            actual_trajectory: [1, actual_seq_len, 4] actual trajectory states (single batch)
            control_update_dt: Time interval between control updates
            downsampled_indices: [1, seq_len_downsampled] time indices of downsampled states (optional)
        """
        super().__init__()
        self.actor = actor
        self.actual_trajectory = actual_trajectory  # [1, actual_seq_len, 4]
        self.control_update_dt = control_update_dt
        self.downsampled_indices = downsampled_indices  # [1, seq_len_downsampled] or None
        
        # Cache for controls (updated at discrete intervals)
        self.last_control_time = -float('inf')
        self.cached_control = None
        self.cached_state = None
        self.total_time = None  # Will be set from integration times
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivative for ODE solver.
        
        Args:
            t: Current time (scalar tensor)
            state: [4] current state vector (torchdiffeq handles single states)
        
        Returns:
            State derivative [4]
        """
        time_val = t.item() if isinstance(t, torch.Tensor) else float(t)
        
        # Check if we need to update control (at discrete intervals)
        # This allows continuous ODE integration with discrete control updates
        need_update = (
            self.cached_control is None or
            (time_val - self.last_control_time) >= self.control_update_dt
        )
        
        if need_update:
            # Reshape state for actor (expects [batch, 4])
            state_batch = state.unsqueeze(0)  # [1, 4]
            
            # Convert continuous time to discrete time index
            # We approximate the time index based on the ratio of current time to total time
            # This is an approximation since we don't know the exact sequence length during ODE integration
            # For now, we'll use a simple mapping: time_index = int(time / dt) where dt is approximate
            # Since we don't have dt here, we'll use 0 as a fallback (will be corrected at output times)
            current_time_idx = torch.tensor([0], device=state.device, dtype=torch.long)
            
            # Find next downsampled state for current time index
            
            next_downsampled_state, next_time_index = get_next_downsampled_state(
                current_time_idx, self.actual_trajectory, self.downsampled_indices
            )
            
            # Get control from actor using mean (deterministic for ODE integration)
            # We'll sample controls later at output times for log_probs
            mean, _ = self.actor.forward(
                next_downsampled_state,
                next_time_index,
                state_batch, 
                current_time_index=current_time_idx
            )
            control = mean.squeeze(0)  # [2]
            
            # Cache control and time
            self.cached_control = control.clone()
            self.cached_state = state.clone()
            self.last_control_time = time_val
        else:
            # Use cached control (keep control constant between update intervals)
            control = self.cached_control
        
        # Reshape for dynamics function (expects [batch, 2])
        control_batch = control.unsqueeze(0)  # [1, 2]
        state_batch = state.unsqueeze(0)  # [1, 4]
        
        # Compute state derivative
        state_derivative = simple_car_dynamics_torch(state_batch, control_batch, dt=0.0)
        
        return state_derivative.squeeze(0)  # [4]


def get_next_downsampled_state(
    current_time_index: torch.Tensor,
    downsampled_states: torch.Tensor,
    downsampled_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the next downsampled state and its time index for each current time index.
    
    Args:
        current_time_index: [batch_size] current time indices
        downsampled_states: [batch_size, seq_len_downsampled, state_dim] or [1, seq_len_downsampled, state_dim] downsampled states
        downsampled_indices: [batch_size, seq_len_downsampled] or [1, seq_len_downsampled] time indices of downsampled states
    
    Returns:
        next_state: [batch_size, state_dim] next downsampled state
        next_time_index: [batch_size] time index of next downsampled state
    """
    batch_size = current_time_index.shape[0]
    device = current_time_index.device
    seq_len_downsampled = downsampled_states.shape[1]
    
    # Expand downsampled_states and downsampled_indices to match batch_size if needed
    if downsampled_states.shape[0] == 1 and batch_size > 1:
        downsampled_states = downsampled_states.expand(batch_size, -1, -1)
    if downsampled_indices.shape[0] == 1 and batch_size > 1:
        downsampled_indices = downsampled_indices.expand(batch_size, -1)
    
    # Expand for broadcasting: [batch_size, 1] vs [batch_size, seq_len_downsampled]
    current_time_expanded = current_time_index.unsqueeze(1).float()  # [batch_size, 1]
    downsampled_indices_expanded = downsampled_indices.unsqueeze(1).float()  # [batch_size, 1, seq_len_downsampled]
    
    # Find next downsampled states: mask where downsampled_indices > current_time
    next_mask = downsampled_indices_expanded > current_time_expanded.unsqueeze(2)  # [batch_size, 1, seq_len_downsampled]
    
    # Get next time indices (smallest time index > current_time for each batch)
    if downsampled_indices_expanded.dtype.is_floating_point:
        max_value = torch.finfo(downsampled_indices_expanded.dtype).max
    else:
        max_value = torch.iinfo(downsampled_indices_expanded.dtype).max
    
    next_time_indices = torch.where(next_mask, downsampled_indices_expanded, torch.full_like(downsampled_indices_expanded, max_value))
    next_time_idx_values, next_time_idx_positions = next_time_indices.min(dim=2)  # [batch_size, 1]
    next_time_idx_values = next_time_idx_values.squeeze(1)  # [batch_size]
    next_time_idx_positions = next_time_idx_positions.squeeze(1)  # [batch_size]
    
    # Check if there's a valid next state (not max_value)
    has_next = next_time_idx_values < max_value
    
    # Get next target states using advanced indexing
    target_indices = torch.where(has_next, next_time_idx_positions, torch.full_like(next_time_idx_positions, seq_len_downsampled - 1, dtype=torch.long))
    
    # Use batch indexing to get target states: [batch_size, state_dim]
    batch_indices = torch.arange(batch_size, device=device)
    next_states = downsampled_states[batch_indices, target_indices]  # [batch_size, state_dim]
    next_time_indices = torch.where(has_next, next_time_idx_values, downsampled_indices[batch_indices, -1])
    
    return next_states, next_time_indices


def rollout_trajectory_state_feedback(
    actor: nn.Module,
    initial_state: torch.Tensor,
    actual_trajectory: torch.Tensor,
    seq_len: int,
    dt: float = DT,
    use_history: bool = True,
    history_length: int = 10,
    use_ode_solver: bool = True,
    use_continuous_ode: bool = True,
    downsampled_indices: torch.Tensor = None
) -> tuple:
    """
    Rollout trajectory using state-feedback control policy with ActorNetwork.
    
    The policy observes the actual trajectory and current predicted state, then outputs
    a control action, which is used to simulate the next state using dynamics.
    
    Args:
        actor: ActorNetwork policy (stochastic policy with built-in exploration)
        initial_state: [batch_size, 4] initial state [x, y, θ, v]
        actual_trajectory: [batch_size, actual_seq_len, 4] actual trajectory states
        seq_len: Number of desired output time steps
        dt: Time step for desired output discretization
        use_history: If True, use sliding window of recent predicted states for actual_trajectory context
        history_length: Length of history window when use_history=True
        use_ode_solver: If True, use RK4 ODE solver (default). If False, use Euler integration.
                       Only used if use_continuous_ode=False.
        use_continuous_ode: If True, use continuous ODE solver (torchdiffeq) to solve over full
                           time horizon and sample at desired discrete times. Requires torchdiffeq.
                           If False, use discrete-time integration (default).
    
    Returns:
        tuple: (predicted_states [batch, seq_len, 4], sampled_controls [batch, seq_len, 2], log_probs [batch, seq_len])
    """
    batch_size = initial_state.shape[0]
    device = initial_state.device
    
    # If using continuous ODE, solve over full time horizon and sample at discrete times
    # if use_continuous_ode and TORCHDIFFEQ_AVAILABLE:
    #     return rollout_trajectory_continuous_ode(
    #         actor=actor,
    #         initial_state=initial_state,
    #         actual_trajectory=actual_trajectory,
    #         seq_len=seq_len,
    #         dt=dt,
    #         use_history=use_history,
    #         history_length=history_length,
    #         downsampled_indices=downsampled_indices
    #     )
    
    # Otherwise, use discrete-time integration
    # Adjust sequence length based on dt ratio
    # seq_len = int(seq_len * DT // dt)
    # Initialize trajectory and controls
    # predicted_states will match dataset format: [initial_state, state_after_1_step, ..., state_after_seq_len-1_steps]
    predicted_states = torch.zeros(batch_size, seq_len, 4, device=device)
    sampled_controls = torch.zeros(batch_size, seq_len, 2, device=device)
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    
    # Initialize current state and store initial state (avoid unnecessary cloning)
    current_state = initial_state  # Use reference, will update in-place where possible
    predicted_states[:, 0, :] = initial_state  # Direct assignment
    
    # Pre-allocate history buffer as tensor for better GPU utilization
    # Use a circular buffer approach with tensor instead of list
    if use_history:
        history_buffer = torch.zeros(batch_size, history_length, 4, device=device)
        history_buffer[:, 0, :] = initial_state
        history_ptr = 1
    else:
        history_buffer = None
        history_ptr = 0
    
    # Pre-allocate time indices tensor
    time_indices = torch.arange(seq_len - 1, device=device, dtype=torch.long)
    
    for t_idx, t in enumerate(range(seq_len - 1)):
        # Use pre-allocated time index
        current_time_idx = time_indices[t_idx:t_idx+1].expand(batch_size)
        
        # Find next downsampled state for current time index
        next_downsampled_state, next_time_index = get_next_downsampled_state(
            current_time_idx, actual_trajectory, downsampled_indices
        )
        
        # Sample action from actor (stochastic policy) - batch processing on GPU
        sampled_control, log_prob, _ = actor.sample(
            next_downsampled_state,
            next_time_index,
            current_state, 
            current_time_index=current_time_idx
        )
        
        # Store control and log prob
        sampled_controls[:, t, :] = sampled_control
        log_probs[:, t] = log_prob
        # print(sampled_control)
        # Apply dynamics - vectorized on GPU
        if use_ode_solver:
            next_state = rk4_step(current_state, sampled_control, dt, simple_car_dynamics_torch)
        else:
            state_derivative = simple_car_dynamics_torch(current_state, sampled_control, dt)
            next_state = current_state + dt * state_derivative
        
        # Store next state and update current state (avoid clone if possible)
        predicted_states[:, t + 1, :] = next_state
        current_state = next_state  # Update reference
        
        # Update history buffer if needed (circular buffer)
        if use_history:
            history_buffer[:, history_ptr % history_length, :] = next_state
            history_ptr = (history_ptr + 1) % history_length
    
    return predicted_states, sampled_controls, log_probs


def rollout_trajectory_continuous_ode(
    actor: nn.Module,
    initial_state: torch.Tensor,
    actual_trajectory: torch.Tensor,
    seq_len: int,
    dt: float = DT,
    use_history: bool = True,
    history_length: int = 10,
    downsampled_indices: torch.Tensor = None
) -> tuple:
    """
    Rollout trajectory using continuous ODE solver (torchdiffeq).
    
    This function solves the ODE continuously over the full time horizon and then
    samples the states at desired discrete time points. Controls are updated at
    fine intervals (control_update_dt) during integration, but we only output
    states and sample controls at the desired output times.
    
    Args:
        actor: ActorNetwork policy
        initial_state: [batch_size, 4] initial state [x, y, θ, v]
        actual_trajectory: [batch_size, actual_seq_len, 4] actual trajectory states
        seq_len: Number of desired output time steps
        dt: Time step for desired output discretization
        use_history: If True, use sliding window of recent predicted states (not used in ODE mode)
        history_length: Length of history window (not used in ODE mode)
    
    Returns:
        tuple: (predicted_states [batch, seq_len, 4], sampled_controls [batch, seq_len, 2], log_probs [batch, seq_len])
    """
    if not TORCHDIFFEQ_AVAILABLE:
        raise ImportError(
            "torchdiffeq is required for continuous ODE solving. "
            "Install it with: pip install torchdiffeq"
        )
    
    batch_size = initial_state.shape[0]
    device = initial_state.device
    
    # Define time points for desired output sampling
    total_time = (seq_len - 1) * dt
    output_times = torch.linspace(0, total_time, seq_len, device=device)  # [seq_len]
    
    # Use finer time grid for ODE integration to get smooth solution
    # This allows adaptive step sizing within the ODE solver
    # We'll still only output at desired times
    num_integration_steps = max(seq_len * 2, 50)  # At least 2x resolution
    integration_times = torch.linspace(0, total_time, num_integration_steps, device=device)
    
    # Initialize output arrays
    predicted_states = torch.zeros(batch_size, seq_len, 4, device=device)
    sampled_controls = torch.zeros(batch_size, seq_len, 2, device=device)
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    
    # Process each batch element separately (torchdiffeq works with single states)
    for batch_idx in range(batch_size):
        # Get initial state for this batch element
        init_state_single = initial_state[batch_idx].clone()  # [4]
        actual_traj_single = actual_trajectory[batch_idx:batch_idx+1]  # [1, actual_seq_len, 4]
        
        # Get downsampled_indices for this batch element
        if downsampled_indices is not None and downsampled_indices.numel() > 0:
            # Handle batch dimension correctly
            if downsampled_indices.shape[0] > batch_idx:
                downsampled_indices_single = downsampled_indices[batch_idx:batch_idx+1]  # [1, seq_len_downsampled]
            elif downsampled_indices.shape[0] == 1:
                # Use the single batch element for all
                downsampled_indices_single = downsampled_indices  # [1, seq_len_downsampled]
            else:
                # Empty or invalid, create sequential indices
                actual_seq_len = actual_trajectory.shape[1]
                downsampled_indices_single = torch.arange(actual_seq_len, device=device, dtype=torch.float32).unsqueeze(0)
        else:
            # Create sequential indices as default
            actual_seq_len = actual_trajectory.shape[1]
            downsampled_indices_single = torch.arange(actual_seq_len, device=device, dtype=torch.float32).unsqueeze(0)
        
        # For this batch element, we only process 1 rollout, not batch_size rollouts
        single_batch_size = 1
        
        # Create ODE function for this batch element
        # Use finer control update interval for smoother integration
        control_update_dt = dt / 2  # Update controls twice per output interval
        ode_func = StateFeedbackODE(
            actor=actor,
            actual_trajectory=actual_traj_single,
            control_update_dt=control_update_dt,
            downsampled_indices=downsampled_indices_single
        )
        
        # Store total time in ODE function for time index conversion
        ode_func.total_time = total_time
        
        # Solve ODE continuously using adaptive solver
        # odeint returns [num_time_steps, 4] for single initial state
        solution_all_times = odeint(
            ode_func, 
            init_state_single, 
            integration_times, 
            method='rk4',
            options={'step_size': dt / 10}  # Fine step size for accuracy
        )  # [num_integration_steps, 4]
        
        # Vectorized interpolation at desired output times
        # Find indices for interpolation using searchsorted (vectorized)
        indices = torch.searchsorted(integration_times, output_times, right=False)  # [seq_len]
        indices = torch.clamp(indices - 1, 0, len(integration_times) - 2)  # Ensure valid range
        
        # Get interpolation points
        t0_vals = integration_times[indices]  # [seq_len]
        t1_vals = integration_times[indices + 1]  # [seq_len]
        alpha = (output_times - t0_vals) / (t1_vals - t0_vals + 1e-8)  # [seq_len]
        alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # Linear interpolation (vectorized)
        states_t0 = solution_all_times[indices]  # [seq_len, 4]
        states_t1 = solution_all_times[indices + 1]  # [seq_len, 4]
        states_interp = (1 - alpha.unsqueeze(-1)) * states_t0 + alpha.unsqueeze(-1) * states_t1  # [seq_len, 4]
        
        # Store predicted states (vectorized)
        predicted_states[batch_idx] = states_interp
        
        # Batch sample controls for all output times at once
        # Expand actual_traj and downsampled_indices for batch processing
        actual_traj_batch = actual_traj_single.expand(seq_len, -1, -1)  # [seq_len, actual_seq_len, 4]
        # Ensure downsampled_indices_single has correct shape before expanding
        if downsampled_indices_single is not None and downsampled_indices_single.numel() > 0:
            if downsampled_indices_single.shape[0] == 1:
                downsampled_indices_batch = downsampled_indices_single.expand(seq_len, -1)  # [seq_len, seq_len_downsampled]
            else:
                # If already has batch dimension, repeat it
                downsampled_indices_batch = downsampled_indices_single.repeat(seq_len, 1)
        else:
            # Create default sequential indices
            actual_seq_len = actual_trajectory.shape[1]
            downsampled_indices_batch = torch.arange(actual_seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(seq_len, -1)
        
        time_indices_batch = torch.arange(seq_len, device=device, dtype=torch.long)
        
        # Find next downsampled states for all time indices
        # actual_traj_batch is [seq_len, actual_seq_len, 4], we need to reshape for each time step
        # downsampled_indices_batch is [seq_len, seq_len_downsampled]
        # For this batch element, we only process 1 rollout (single_batch_size = 1)
        next_downsampled_states_batch = torch.zeros(single_batch_size, seq_len, 4, device=device)
        next_time_indices_batch = torch.zeros(single_batch_size, seq_len, device=device)
        for t in range(seq_len):
            time_idx_t = time_indices_batch[t:t+1].expand(single_batch_size)
            # actual_traj_batch[t] is [actual_seq_len, 4], expand to [single_batch_size, actual_seq_len, 4]
            actual_traj_t = actual_traj_batch[t:t+1].expand(single_batch_size, -1, -1)  # [single_batch_size, actual_seq_len, 4]
            # downsampled_indices_batch[t] is [seq_len_downsampled], expand to [single_batch_size, seq_len_downsampled]
            downsampled_indices_t = downsampled_indices_batch[t:t+1].expand(single_batch_size, -1)  # [single_batch_size, seq_len_downsampled]
            
            next_states_t, next_times_t = get_next_downsampled_state(
                time_idx_t, actual_traj_t, downsampled_indices_t
            )
            next_downsampled_states_batch[:, t, :] = next_states_t
            next_time_indices_batch[:, t] = next_times_t
        
        # Batch sample actions
        # states_interp is [seq_len, 4], we need to expand it to [single_batch_size, seq_len, 4]
        states_interp_expanded = states_interp.unsqueeze(0).expand(single_batch_size, -1, -1)  # [single_batch_size, seq_len, 4]
        
        sampled_controls_batch = torch.zeros(single_batch_size, seq_len, 2, device=device)
        log_probs_batch = torch.zeros(single_batch_size, seq_len, device=device)
        for t in range(seq_len):
            sampled_control_t, log_prob_t, _ = actor.sample(
                next_downsampled_states_batch[:, t, :],
                next_time_indices_batch[:, t],
                states_interp_expanded[:, t, :],
                current_time_index=time_indices_batch[t:t+1].expand(single_batch_size)
            )
            sampled_controls_batch[:, t, :] = sampled_control_t
            log_probs_batch[:, t] = log_prob_t
        
        # Store controls and log probs
        # sampled_controls_batch is [single_batch_size, seq_len, 2] = [1, seq_len, 2]
        sampled_controls[batch_idx] = sampled_controls_batch[0]  # [seq_len, 2]
        log_probs[batch_idx] = log_probs_batch[0]  # [seq_len]
    
    return predicted_states, sampled_controls, log_probs
