"""
Utility functions for reinforcement learning training.

This module provides shared utility functions used across different RL training scripts.
"""

import torch
import torch.nn as nn

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
        control_update_dt: float = DT
    ):
        """
        Initialize the ODE function.
        
        Args:
            actor: ActorNetwork policy
            actual_trajectory: [1, actual_seq_len, 4] actual trajectory states (single batch)
            control_update_dt: Time interval between control updates
        """
        super().__init__()
        self.actor = actor
        self.actual_trajectory = actual_trajectory  # [1, actual_seq_len, 4]
        self.control_update_dt = control_update_dt
        
        # Cache for controls (updated at discrete intervals)
        self.last_control_time = -float('inf')
        self.cached_control = None
        self.cached_state = None
    
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
            
            # Get control from actor using mean (deterministic for ODE integration)
            # We'll sample controls later at output times for log_probs
            # Note: action_history is not used in continuous ODE mode for simplicity
            mean, _, _ = self.actor.forward(self.actual_trajectory, state_batch, action_history=None)
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


def rollout_trajectory_state_feedback(
    actor: nn.Module,
    initial_state: torch.Tensor,
    actual_trajectory: torch.Tensor,
    seq_len: int,
    dt: float = DT,
    use_history: bool = True,
    history_length: int = 10,
    use_ode_solver: bool = True,
    use_continuous_ode: bool = True
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
    if use_continuous_ode and TORCHDIFFEQ_AVAILABLE:
        return rollout_trajectory_continuous_ode(
            actor=actor,
            initial_state=initial_state,
            actual_trajectory=actual_trajectory,
            seq_len=seq_len,
            dt=dt,
            use_history=use_history,
            history_length=history_length
        )
    
    # Otherwise, use discrete-time integration
    # Adjust sequence length based on dt ratio
    seq_len = int(seq_len * DT // dt)
    # Initialize trajectory and controls
    # predicted_states will match dataset format: [initial_state, state_after_1_step, ..., state_after_seq_len-1_steps]
    predicted_states = torch.zeros(batch_size, seq_len, 4, device=device)
    sampled_controls = torch.zeros(batch_size, seq_len, 2, device=device)
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    
    # Initialize current state and store initial state
    current_state = initial_state.clone()
    predicted_states[:, 0, :] = initial_state.clone()
    
    # Predicted state history buffer (for context if use_history is True)
    predicted_history = [initial_state.clone()]  # List of [batch_size, 4] tensors
    
    # Action history buffer for transformer input
    action_history_list = []  # List of [batch_size, 2] tensors
    
    for t in range(seq_len):
        # Prepare actual trajectory input for actor
        # If use_history, use recent predicted states combined with actual trajectory
        # Otherwise, just use the full actual trajectory
        if use_history and len(predicted_history) > 1:
            # Use sliding window of recent predicted states
            history_window = min(len(predicted_history), history_length)
            recent_predicted = torch.stack(predicted_history[-history_window:], dim=1)  # [batch, history_window, 4]
            # Combine with actual trajectory: concatenate along sequence dimension
            # Use actual trajectory as the main context
            actual_traj_input = actual_trajectory  # [batch, actual_seq_len, 4]
        else:
            # Use full actual trajectory
            actual_traj_input = actual_trajectory  # [batch, actual_seq_len, 4]
        
        # Prepare action history for transformer
        if len(action_history_list) > 0:
            # Stack recent actions (most recent last)
            action_history = torch.stack(action_history_list, dim=1)  # [batch, len(action_history_list), 2]
        else:
            # No previous actions
            action_history = None
        
        # Sample action from actor (stochastic policy)
        # actor.sample() returns (action, log_prob, mean)
        sampled_control, log_prob, _ = actor.sample(actual_traj_input, current_state, action_history)
        # sampled_control: [batch, 2]
        # log_prob: [batch] (already summed over action dimensions)
        
        # Store control and log prob
        sampled_controls[:, t, :] = sampled_control
        log_probs[:, t] = log_prob
        
        # Update action history
        action_history_list.append(sampled_control.clone())
        # Limit history size to action_history_len (defined in actor network)
        if hasattr(actor, 'action_history_len'):
            max_history = actor.action_history_len
        else:
            max_history = 10  # Default fallback
        if len(action_history_list) > max_history:
            action_history_list.pop(0)  # Remove oldest action
        
        # Apply dynamics to get next state (only if not at last step)
        if t < seq_len - 1:
            # Apply dynamics
            if use_ode_solver:
                # Use RK4 ODE solver for more accurate integration
                next_state = rk4_step(current_state, sampled_control, dt, simple_car_dynamics_torch)
            else:
                # Use Euler integration (simpler but less accurate)
                state_derivative = simple_car_dynamics_torch(current_state, sampled_control, dt)
                next_state = current_state + dt * state_derivative
            
            # Store the state AFTER applying dynamics at index t+1
            predicted_states[:, t + 1, :] = next_state
            
            # Update state and history
            current_state = next_state.clone()
            predicted_history.append(current_state)
            
            # Limit history size
            if len(predicted_history) > history_length:
                predicted_history.pop(0)
    
    return predicted_states, sampled_controls, log_probs


def rollout_trajectory_continuous_ode(
    actor: nn.Module,
    initial_state: torch.Tensor,
    actual_trajectory: torch.Tensor,
    seq_len: int,
    dt: float = DT,
    use_history: bool = True,
    history_length: int = 10
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
        
        # Create ODE function for this batch element
        # Use finer control update interval for smoother integration
        control_update_dt = dt / 2  # Update controls twice per output interval
        ode_func = StateFeedbackODE(
            actor=actor,
            actual_trajectory=actual_traj_single,
            control_update_dt=control_update_dt
        )
        
        # Solve ODE continuously using adaptive solver
        # odeint returns [num_time_steps, 4] for single initial state
        solution_all_times = odeint(
            ode_func, 
            init_state_single, 
            integration_times, 
            method='rk4',
            options={'step_size': dt / 10}  # Fine step size for accuracy
        )  # [num_integration_steps, 4]
        
        # Interpolate solution at desired output times
        # Simple linear interpolation (or we can use the closest integration points)
        for t_idx, t in enumerate(output_times):
            t_val = t.item()
            
            # Find closest integration time indices
            if t_val <= integration_times[0].item():
                state_at_t = solution_all_times[0]  # [4]
            elif t_val >= integration_times[-1].item():
                state_at_t = solution_all_times[-1]  # [4]
            else:
                # Linear interpolation
                idx = torch.searchsorted(integration_times, t_val) - 1
                idx = max(0, min(idx, len(integration_times) - 2))
                t0 = integration_times[idx].item()
                t1 = integration_times[idx + 1].item()
                alpha = (t_val - t0) / (t1 - t0) if t1 > t0 else 0.0
                state_at_t = (1 - alpha) * solution_all_times[idx] + alpha * solution_all_times[idx + 1]
            
            # Store predicted state
            predicted_states[batch_idx, t_idx] = state_at_t
            
            # Sample control and log prob at this output time (for stochastic policy)
            state_for_actor = state_at_t.unsqueeze(0)  # [1, 4]
            # Note: action_history is not used in continuous ODE mode for simplicity
            sampled_control, log_prob, _ = actor.sample(actual_traj_single, state_for_actor[0], action_history=None)
            # sampled_control: [2], log_prob: scalar
            
            sampled_controls[batch_idx, t_idx] = sampled_control
            log_probs[batch_idx, t_idx] = log_prob
    
    return predicted_states, sampled_controls, log_probs
