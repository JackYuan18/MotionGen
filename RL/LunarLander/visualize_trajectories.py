"""
Visualization utilities for comparing predicted and actual trajectories.

This module provides functions to generate comparison plots between actual trajectories
from the dataset and trajectories generated using learned control inputs.
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Optional

# Import ActorNetwork and utils for deterministic rollout
try:
    from model import ActorNetwork
    from utils import simple_car_dynamics_torch, DT, rk4_step
except ImportError:
    # Try alternative import path
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from learn_actions.model import ActorNetwork
        from learn_actions.utils import simple_car_dynamics_torch, DT, rk4_step
    except ImportError:
        ActorNetwork = None
        simple_car_dynamics_torch = None
        DT = 0.1
        rk4_step = None


def compute_constant_speed_trajectory(initial_state: np.ndarray, second_state: np.ndarray, 
                                      seq_len: int, dt: float = 0.1) -> np.ndarray:
    """
    Compute a trajectory with constant speed (no acceleration) and no steering (angular velocity = 0).
    
    The initial speed is calculated from the distance between the first two states.
    
    Args:
        initial_state: [4] initial state [x, y, theta, v]
        second_state: [4] second state to calculate initial velocity from
        seq_len: Number of time steps in the trajectory
        dt: Time step
    
    Returns:
        trajectory: [seq_len, 4] trajectory with constant speed and no steering
    """
    # Calculate initial velocity from first two states
    dx = second_state[0] - initial_state[0]
    dy = second_state[1] - initial_state[1]
    distance = np.sqrt(dx**2 + dy**2)
    initial_velocity = distance / dt if dt > 0 else 0.0
    
    # Initialize trajectory
    trajectory = np.zeros((seq_len, 4))
    trajectory[0] = initial_state.copy()
    
    # Get initial heading and velocity
    x, y, theta, v = initial_state
    v = initial_velocity  # Use calculated velocity
    
    # Rollout with constant speed (acceleration = 0) and no steering (angular_velocity = 0)
    for t in range(1, seq_len):
        # State derivatives: ẋ = v*cos(θ), ẏ = v*sin(θ), θ̇ = v*ω = 0, v̇ = a = 0
        # With constant speed and no steering: acceleration = 0, angular_velocity = 0
        dx_dt = v * np.cos(theta)
        dy_dt = v * np.sin(theta)
        dtheta_dt = 0.0  # No steering
        dv_dt = 0.0  # No acceleration
        
        # Euler integration
        x = x + dx_dt * dt
        y = y + dy_dt * dt
        theta = theta + dtheta_dt * dt
        v = v + dv_dt * dt  # Velocity stays constant
        
        trajectory[t] = [x, y, theta, v]
    
    return trajectory


def rollout_trajectory_deterministic_mean(
    actor: nn.Module,
    initial_state: torch.Tensor,
    actual_trajectory: torch.Tensor,
    seq_len: int,
    dt: float = 0.1,
    use_ode_solver: bool = True
) -> torch.Tensor:
    """
    Rollout trajectory using deterministic mean action from actor policy (no sampling).
    
    This function uses actor.forward() to get the mean action directly, without sampling.
    
    Args:
        actor: ActorNetwork policy
        initial_state: [batch_size, 4] initial state [x, y, θ, v]
        actual_trajectory: [batch_size, actual_seq_len, 4] actual trajectory states
        seq_len: Number of desired output time steps
        dt: Time step for desired output discretization
        use_ode_solver: If True, use RK4 ODE solver (default). If False, use Euler integration.
    
    Returns:
        predicted_states: [batch_size, seq_len, 4] deterministic trajectory
    """
    if simple_car_dynamics_torch is None or rk4_step is None:
        raise ImportError("Cannot import utils functions for deterministic rollout")
    
    batch_size = initial_state.shape[0]
    device = initial_state.device
    
    # Adjust sequence length based on dt ratio (DT is imported from utils)
    seq_len_adjusted = int(seq_len * DT // dt)
    seq_len = seq_len_adjusted
    
    # Initialize trajectory
    predicted_states = torch.zeros(batch_size, seq_len, 4, device=device)
    predicted_states[:, 0, :] = initial_state.clone()
    
    # Initialize current state
    current_state = initial_state.clone()
    
    for t in range(seq_len):
        # Get current time index
        current_time_idx = torch.tensor([float(t)], device=device, dtype=torch.float32)
        if batch_size > 1:
            current_time_idx = current_time_idx.expand(batch_size)

        # Get mean action from actor (deterministic, no sampling)
        # New signature: forward(current_state, current_time_index)
        mean, _ = actor.forward(current_state, current_time_index=current_time_idx)
        # mean: [batch, 2] or [2] if batch_size=1 and current_state is 1D
        
        # Ensure mean has batch dimension
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)  # [1, 2]
        
        # Use mean as the deterministic action
        deterministic_action = mean  # [batch, 2]
        
        # Apply dynamics to get next state (only if not at last step)
        if t < seq_len - 1:
            from utils import ensure_batch_dim, ensure_tensor_shape
            
            # Ensure both state and action have consistent 2D shapes with matching batch dimension
            state_for_dynamics = ensure_batch_dim(current_state, batch_size)
            action_for_dynamics = ensure_batch_dim(deterministic_action, batch_size)
            
            # Apply dynamics
            if use_ode_solver:
                next_state = rk4_step(state_for_dynamics, action_for_dynamics, dt, simple_car_dynamics_torch)
            else:
                state_derivative = simple_car_dynamics_torch(state_for_dynamics, action_for_dynamics, dt)
                next_state = state_for_dynamics + dt * state_derivative
            
            # Ensure next_state has correct shape [batch_size, 4]
            next_state = ensure_tensor_shape(next_state, (batch_size, 4))
            
            # Store the state AFTER applying dynamics at index t+1
            predicted_states[:, t + 1, :] = next_state
            
            # Update state
            current_state = next_state.clone()
  
    return predicted_states


def generate_trajectory_comparison_plots(
    dataset,
    output_dir: Path,
    num_epochs: int = 20,
    algorithm_name: str = "unknown",
    environment_name: str = "unknown",
    dt: float = 0.1,
    include_baseline: bool = True,
    include_deterministic_mean: bool = False,
    actual_states_downsampled: Optional[np.ndarray] = None
):
    """
    Generate comparison plots between actual and predicted trajectories across epochs.
    
    Uses saved rollout trajectories from training checkpoints (no trajectory generation).
    For each trajectory, creates a 3x3 grid showing comparisons at 9 evenly spaced epochs.
    
    Args:
        dataset: Dataset containing normalized states
        output_dir: Directory containing checkpoints and where to save plots
        num_epochs: Total number of training epochs
        algorithm_name: Name of the reinforcement learning algorithm (e.g., "PPO", "SAC", "REINFORCE")
        dt: Discrete time step value used for training
        include_baseline: If True, include a baseline trajectory with constant speed and no steering (default: True)
        include_deterministic_mean: If True, include a trajectory rolled out using deterministic mean action (default: True)
    """
    print("\nGenerating trajectory comparison plots across epochs...")
    comparison_dir = output_dir / 'trajectory_comparisons'
    comparison_dir.mkdir(exist_ok=True)
    
    # Find all available checkpoints
    checkpoint_files = sorted(output_dir.glob('checkpoint_epoch_*.pt'))
    
    # Extract epoch numbers from checkpoint filenames and create mapping
    available_checkpoints = {}
    for f in checkpoint_files:
        try:
            epoch_num = int(f.stem.split('_')[-1])
            available_checkpoints[epoch_num] = f
        except:
            continue
    
    # Always include final model
    if (output_dir / 'final_model.pt').exists():
        available_checkpoints[num_epochs] = output_dir / 'final_model.pt'
    
    if not available_checkpoints:
        print("Warning: No checkpoint files found. Cannot generate comparison plots.")
        return
    
    # Select 9 evenly spaced epochs across the full training range (1 to num_epochs)
    target_epochs = np.linspace(1, num_epochs, 9, dtype=int).tolist()
    
    # Find the closest available checkpoint for each target epoch
    epochs_to_plot = []
    checkpoint_files_to_use = []
    available_epochs = sorted(available_checkpoints.keys())
    
    # If we have fewer than 9 available epochs, use all of them and repeat the last one
    if len(available_epochs) < 9:
        # Use all available epochs
        for epoch in available_epochs:
            if epoch not in epochs_to_plot:
                epochs_to_plot.append(epoch)
                checkpoint_files_to_use.append(available_checkpoints[epoch])
        # Pad with the last available epoch to get 9 total
        while len(epochs_to_plot) < 9:
            epochs_to_plot.append(available_epochs[-1])
            checkpoint_files_to_use.append(available_checkpoints[available_epochs[-1]])
    else:
        # We have enough epochs - find closest for each target
        for target_epoch in target_epochs:
            # Find closest available epoch
            closest_epoch = min(available_epochs, key=lambda x: abs(x - target_epoch))
            if closest_epoch not in epochs_to_plot:  # Avoid duplicates
                epochs_to_plot.append(closest_epoch)
                checkpoint_files_to_use.append(available_checkpoints[closest_epoch])
        
        
    
    epochs_to_plot = epochs_to_plot[:9]
    checkpoint_files_to_use = checkpoint_files_to_use[:9]
    
    # Check if best_model.pt exists and replace 9th subplot with best model
    best_model_path = output_dir / 'best_model.pt'
    
    
    print(f"Plotting comparisons at epochs: {epochs_to_plot}")
    
    # Load rollout trajectories from checkpoints (if available)
    rollout_trajectories_by_epoch = {}
    for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
    
        checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load on CPU first
        
            
        rollout_trajectories_by_epoch[epoch] = checkpoint['rollout_trajectories']
        
    
    
    # If best model was used for 9th subplot, make sure it's loaded
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists() and best_model_path in checkpoint_files_to_use:
        best_epoch = epochs_to_plot[8]  # 9th subplot (index 8)
        if best_epoch not in rollout_trajectories_by_epoch:
          
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            
            rollout_trajectories_by_epoch[best_epoch] = best_checkpoint['rollout_trajectories']
                
    
    # Check if we have saved rollout trajectories
    
    num_trajs_per_epoch = rollout_trajectories_by_epoch[list(rollout_trajectories_by_epoch.keys())[0]].shape[0]
    print(f"Found saved rollout trajectories for {len(rollout_trajectories_by_epoch)} epochs. "
          f"Each epoch has {num_trajs_per_epoch} trajectories. Using saved trajectories.")
    
    # Load actor models for deterministic mean rollout if requested
    actor_models_by_epoch = {}
    if include_deterministic_mean and ActorNetwork is not None:
        print("Loading actor models for deterministic mean rollout...")
        for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
          
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Get model parameters from checkpoint
            hidden_dim = checkpoint.get('hidden_dim', 128)
            num_layers = checkpoint.get('num_layers', 2)       
            # Create actor model (action_history_len was removed)
            # For LunarLander: state_dim=8, num_actions=4 (discrete actions)
            # Try to get from checkpoint, otherwise use defaults
            state_dim = checkpoint.get('state_dim', 8)
            num_actions = checkpoint.get('num_actions', 4)
            actor = ActorNetwork(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                deterministic=True  # Use deterministic mode for mean action
            )
            # Be tolerant to architecture changes across experiments
           
            actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            
            actor.eval()  # Set to evaluation mode
            actor_models_by_epoch[epoch] = actor
            
    
    # If best model was used for 9th subplot, make sure actor model is loaded
    if include_deterministic_mean and ActorNetwork is not None and best_model_path.exists() and best_model_path in checkpoint_files_to_use:
        best_epoch = epochs_to_plot[8]  # 9th subplot (index 8)
        if best_epoch not in actor_models_by_epoch:
        
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
    
            hidden_dim = best_checkpoint.get('hidden_dim', 128)
            num_layers = best_checkpoint.get('num_layers', 2)
            
            # For LunarLander: state_dim=8, num_actions=4 (discrete actions)
            # Try to get from checkpoint, otherwise use defaults
            state_dim = best_checkpoint.get('state_dim', 8)
            num_actions = best_checkpoint.get('num_actions', 4)
            actor = ActorNetwork(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                deterministic=True
            )
            # try:
            actor.load_state_dict(best_checkpoint['actor_state_dict'], strict=False)
            #     if missing or unexpected:
            #         print(f"Warning: partial actor checkpoint load for best model. "
            #               f"missing={len(missing)}, unexpected={len(unexpected)}")
            # except Exception as e:
            #     print(f"Warning: could not load best actor model: {e}")
            #     actor = None
            
            actor.eval()
            actor_models_by_epoch[best_epoch] = actor
        
    # Generate plots for each trajectory showing evolution across epochs
    num_trajectories = len(dataset)
    for traj_idx in tqdm(range(num_trajectories), desc="Generating epoch comparison plots"):
        _create_epoch_comparison_plot_with_saved_trajectories(
            traj_idx,
            dataset,
            rollout_trajectories_by_epoch,
            epochs_to_plot,
            output_dir,
            algorithm_name,
            environment_name,
            dt,
            include_baseline,
            include_deterministic_mean,
            actor_models_by_epoch,
            actual_states_downsampled
        )
    
    print(f"Trajectory comparison plots saved to: {comparison_dir}")
    
  
    
    # Generate best rollout vs actual trajectory comparison plots
    print("\nGenerating best rollout vs actual trajectory comparison plots...")
    _generate_best_rollout_vs_actual_comparison_plots(
        dataset=dataset,
        output_dir=output_dir,
        epochs_to_plot=epochs_to_plot,
        checkpoint_files_to_use=checkpoint_files_to_use,
        algorithm_name=algorithm_name,
        dt=dt,
        actual_states_downsampled=actual_states_downsampled
    )


def _create_epoch_comparison_plot_with_saved_trajectories(
    traj_idx: int,
    dataset,
    rollout_trajectories_by_epoch: Dict[int, torch.Tensor],
    epochs_to_plot: list,
    output_dir: Path,
    algorithm_name: str = "unknown",
    environment_name: str = "unknown",
    dt: float = 0.1,
    include_baseline: bool = True,
    include_deterministic_mean: bool = True,
    actor_models_by_epoch: Optional[Dict[int, nn.Module]] = None,
    actual_states_downsampled: Optional[np.ndarray] = None
):
    """
    Create a 3x3 grid plot showing trajectory comparison using saved rollout trajectories.
    Plots up to 10 rollout trajectories per epoch.
    
    Args:
        traj_idx: Index of trajectory to plot
        dataset: Dataset containing normalized states
        rollout_trajectories_by_epoch: Dictionary mapping epoch numbers to saved rollout trajectories (normalized)
                                        Each value is [num_trajectories, seq_len, 4]
        epochs_to_plot: List of epoch numbers to plot
        output_dir: Directory to save plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
        include_baseline: If True, include a baseline trajectory with constant speed and no steering (default: True)
        include_deterministic_mean: If True, include a trajectory rolled out using deterministic mean action (default: True)
        actor_models_by_epoch: Dictionary mapping epoch numbers to actor models for deterministic rollout
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    
    # Get actual trajectory data
    states_norm = dataset[traj_idx]  # [seq_len, 4] - normalized
    states_norm_np = states_norm.numpy() if isinstance(states_norm, torch.Tensor) else states_norm
    
    # Denormalize actual states
    # Get normalization stats for this specific trajectory
   # norm_stats = dataset.get_normalization_stats(traj_idx)
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    if isinstance(state_mean, torch.Tensor):
        state_mean = state_mean.numpy()
    if isinstance(state_std, torch.Tensor):
        state_std = state_std.numpy()
    
    actual_states = states_norm_np * state_std + state_mean
    actual_states_np = actual_states  # [seq_len, 4]
    
    # Denormalize downsampled states if provided
    actual_states_downsampled_np = None
    if actual_states_downsampled is not None:
        # actual_states_downsampled is already normalized, so denormalize it
        actual_states_downsampled_np = actual_states_downsampled * state_std + state_mean
    
    # Normalize actual trajectory for actor input
    actual_trajectory_norm = torch.from_numpy(states_norm_np).float().unsqueeze(0)  # [1, seq_len, 4]
    initial_state_norm = torch.from_numpy(states_norm_np[0:1]).float()  # [1, 4]
    
    # Compute baseline trajectory if requested (only for 4D vehicle states, not for LunarLander)
    baseline_trajectory = None
    if include_baseline and actual_states_np.shape[0] >= 2:
        state_dim = actual_states_np.shape[1] if len(actual_states_np.shape) > 1 else len(actual_states_np[0])
        # Baseline trajectory only makes sense for 4D vehicle states [x, y, theta, v]
        if state_dim == 4:
            initial_state = actual_states_np[0]  # [4]
            second_state = actual_states_np[1]  # [4]
            seq_len = actual_states_np.shape[0]
            baseline_trajectory = compute_constant_speed_trajectory(
                initial_state, second_state, seq_len, dt
            )
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for plot_idx, epoch in enumerate(epochs_to_plot[:9]):  # Limit to 9 plots
        if epoch not in rollout_trajectories_by_epoch:
            continue
        
        ax = axes[plot_idx]
        
        # Get saved rollout trajectories (normalized)
        rollout_trajs_norm = rollout_trajectories_by_epoch[epoch]
        if isinstance(rollout_trajs_norm, torch.Tensor):
            rollout_trajs_norm = rollout_trajs_norm.numpy()
        
        # Handle both old format [seq_len, 4] and new format [num_trajectories, seq_len, 4]
        if len(rollout_trajs_norm.shape) == 2:
            # Old format: single trajectory, add batch dimension
            rollout_trajs_norm = rollout_trajs_norm[np.newaxis, :, :]  # [1, seq_len, 4]
        
        # Denormalize rollout trajectories
        # rollout_trajs_norm: [num_trajectories, seq_len, 4]
        # state_std: [4], state_mean: [4]
        predicted_states_np = rollout_trajs_norm * state_std[np.newaxis, np.newaxis, :] + state_mean[np.newaxis, np.newaxis, :]
        # predicted_states_np: [num_trajectories, seq_len, 4]
        num_trajectories = min(10, predicted_states_np.shape[0])
        
        # Plot complete actual trajectory
        ax.plot(actual_states_np[:, 0], actual_states_np[:, 1], 'b--', 
               markersize=2, label='Actual (complete)', linewidth=1, alpha=0.8, zorder=10)
        
        # Plot downsampled actual trajectory if provided
        if actual_states_downsampled_np is not None:
            ax.plot(actual_states_downsampled_np[:, 0], actual_states_downsampled_np[:, 1], 'b.', 
                   markersize=4, label='Actual (downsampled)', linewidth=0, alpha=0.6, zorder=11)
        
        # Plot baseline trajectory if available
        if baseline_trajectory is not None:
            ax.plot(baseline_trajectory[:, 0], baseline_trajectory[:, 1], 'm:', 
                   markersize=2, label='Baseline (const speed, no steering)', 
                   linewidth=2, alpha=0.7, zorder=8)
        
        # Plot deterministic mean rollout if available
        if include_deterministic_mean and actor_models_by_epoch is not None and epoch in actor_models_by_epoch:
            try:
                actor = actor_models_by_epoch[epoch]
                with torch.no_grad():
                    # Rollout using deterministic mean action
                    deterministic_traj_norm = rollout_trajectory_deterministic_mean(
                        actor=actor,
                        initial_state=initial_state_norm,
                        actual_trajectory=actual_trajectory_norm,
                        seq_len=actual_states_np.shape[0],
                        dt=dt,
                        use_ode_solver=True
                    )  # [1, seq_len, 4]
                    
                    # Denormalize deterministic trajectory
                    deterministic_traj_np = deterministic_traj_norm[0].cpu().numpy()  # [seq_len, 4]
                    deterministic_traj_denorm = deterministic_traj_np * state_std + state_mean  # [seq_len, 4]
                    
                    # Plot deterministic mean trajectory
                    ax.plot(deterministic_traj_denorm[:, 0], deterministic_traj_denorm[:, 1], 'c-', 
                           markersize=2, label='Deterministic mean', 
                           linewidth=2, alpha=0.8, zorder=9)
            except Exception as e:
                print(f"Warning: Could not rollout deterministic mean trajectory for epoch {epoch}: {e}")
        
        # Plot start point
        ax.plot(actual_states_np[0, 0], actual_states_np[0, 1], 'go', 
               markersize=6, label='Start', zorder=15)
        
        # Mark best model in title
        best_model_path = output_dir / 'best_model.pt'
        if best_model_path.exists() and plot_idx == 8:  # 9th subplot (index 8)
            try:
                best_checkpoint = torch.load(best_model_path, map_location='cpu')
                best_loss = best_checkpoint.get('train_loss', 'N/A')
                if isinstance(best_loss, (int, float)):
                    ax.set_title(f'Epoch {epoch} (BEST, loss={best_loss:.6f}, {num_trajectories} rollouts)', 
                                fontweight='bold', color='green', fontsize=25)
                else:
                    ax.set_title(f'Epoch {epoch} (BEST, {num_trajectories} rollouts)', 
                                fontweight='bold', color='green', fontsize=25)
            except:
                ax.set_title(f'Epoch {epoch} (BEST, {num_trajectories} rollouts)', 
                            fontweight='bold', color='green', fontsize=25)
        else:
            ax.set_title(f'Epoch {epoch} ({num_trajectories} rollouts)', fontsize=25)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=25)
        
        # Plot all rollout trajectories (up to 10)
        for traj_idx_plot in range(num_trajectories):
            traj = predicted_states_np[traj_idx_plot]  # [seq_len, 4]
            ax.plot(traj[:, 0], traj[:, 1], 'r-o', 
                   markersize=1, linewidth=2, alpha=0.4, zorder=5)
    
    plt.suptitle(f'Trajectory {traj_idx}: Evolution Across Training Epochs (10 Rollout Trajectories)', 
                 fontsize=25, y=0.995)
    plt.tight_layout()
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'{environment_name}_trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_epochs.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()


def _generate_mean_vs_actual_comparison_plots(
    dataset,
    output_dir: Path,
    actor_models_by_epoch: Dict[int, nn.Module],
    epochs_to_plot: list,
    algorithm_name: str = "unknown",
    dt: float = 0.1,
    actual_states_downsampled: Optional[np.ndarray] = None
):
    """
    Generate separate comparison plots between deterministic mean trajectories and actual trajectories.
    
    Creates a separate figure for each trajectory showing mean vs actual comparison across epochs.
    
    Args:
        dataset: Dataset containing normalized states
        output_dir: Directory containing checkpoints and where to save plots
        actor_models_by_epoch: Dictionary mapping epoch numbers to actor models
        epochs_to_plot: List of epoch numbers to plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    comparison_dir.mkdir(exist_ok=True)
    
    if not actor_models_by_epoch:
        print("Warning: No actor models available for mean trajectory comparison.")
        return
    
    # Generate plots for each trajectory
    num_trajectories = len(dataset)
    for traj_idx in tqdm(range(num_trajectories), desc="Generating mean vs actual comparison plots"):
        _create_mean_vs_actual_plot(
            traj_idx=traj_idx,
            dataset=dataset,
            actor_models_by_epoch=actor_models_by_epoch,
            epochs_to_plot=epochs_to_plot,
            output_dir=output_dir,
            algorithm_name=algorithm_name,
            dt=dt,
            actual_states_downsampled=actual_states_downsampled
        )
    
    print(f"Mean vs actual trajectory comparison plots saved to: {comparison_dir}")


def _create_mean_vs_actual_plot(
    traj_idx: int,
    dataset,
    actor_models_by_epoch: Dict[int, nn.Module],
    epochs_to_plot: list,
    output_dir: Path,
    algorithm_name: str = "unknown",
    dt: float = 0.1,
    actual_states_downsampled: Optional[np.ndarray] = None
):
    """
    Create a 3x3 grid plot showing mean trajectory vs actual trajectory comparison across epochs.
    
    Args:
        traj_idx: Index of trajectory to plot
        dataset: Dataset containing normalized states
        actor_models_by_epoch: Dictionary mapping epoch numbers to actor models
        epochs_to_plot: List of epoch numbers to plot
        output_dir: Directory to save plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    
    # Get actual trajectory data
    states_norm = dataset[traj_idx]  # [seq_len, 4] - normalized
    states_norm_np = states_norm.numpy() if isinstance(states_norm, torch.Tensor) else states_norm
    
    # Denormalize actual states
    # Get normalization stats for this specific trajectory
    
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    if isinstance(state_mean, torch.Tensor):
        state_mean = state_mean.numpy()
    if isinstance(state_std, torch.Tensor):
        state_std = state_std.numpy()
    
    actual_states = states_norm_np * state_std + state_mean
    actual_states_np = actual_states  # [seq_len, 4]

    # Denormalize downsampled states if provided (they are passed in normalized space)
    actual_states_downsampled_np = None
    if actual_states_downsampled is not None:
        actual_states_downsampled_np = actual_states_downsampled * state_std + state_mean
    
    # Normalize actual trajectory for actor input
    actual_trajectory_norm = torch.from_numpy(states_norm_np).float().unsqueeze(0)  # [1, seq_len, 4]
    initial_state_norm = torch.from_numpy(states_norm_np[0:1]).float()  # [1, 4]
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for plot_idx, epoch in enumerate(epochs_to_plot[:9]):  # Limit to 9 plots
        if epoch not in actor_models_by_epoch:
            # If no actor model for this epoch, skip or show empty plot
            ax = axes[plot_idx]
            ax.text(0.5, 0.5, f'No model for epoch {epoch}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=25)
            ax.set_title(f'Epoch {epoch}', fontsize=25)
            continue
        
        ax = axes[plot_idx]
        
        # Plot complete actual trajectory
        ax.plot(actual_states_np[:, 0], actual_states_np[:, 1], 'b--', 
               markersize=2, label='Actual trajectory', linewidth=2, alpha=0.8, zorder=10)

        # Plot downsampled actual states if provided
        if actual_states_downsampled_np is not None:
            ax.plot(
                actual_states_downsampled_np[:, 0],
                actual_states_downsampled_np[:, 1],
                'b.',
                markersize=25,
                label='Waypoints',
                linewidth=0,
                alpha=0.7,
                zorder=11
            )
        
        # Plot deterministic mean rollout if available
        try:
            actor = actor_models_by_epoch[epoch]
            with torch.no_grad():
                # Rollout using deterministic mean action
                deterministic_traj_norm = rollout_trajectory_deterministic_mean(
                    actor=actor,
                    initial_state=initial_state_norm,
                    actual_trajectory=actual_trajectory_norm,
                    seq_len=actual_states_np.shape[0],
                    dt=dt,
                    use_ode_solver=True
                )  # [1, seq_len, 4]
                
                # Denormalize deterministic trajectory
                deterministic_traj_np = deterministic_traj_norm[0].cpu().numpy()  # [seq_len, 4]
                deterministic_traj_denorm = deterministic_traj_np * state_std + state_mean  # [seq_len, 4]
                
                # Plot deterministic mean trajectory
                ax.plot(deterministic_traj_denorm[:, 0], deterministic_traj_denorm[:, 1], 'r-', 
                       markersize=2, label='Generated trajectory', 
                       linewidth=2, alpha=0.8, zorder=9)
        except Exception as e:
            print(f"Warning: Could not rollout deterministic mean trajectory for epoch {epoch}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=25)
        
        # Plot start point
        ax.plot(actual_states_np[0, 0], actual_states_np[0, 1], 'go', 
               markersize=8, label='Start', zorder=15)
        
        ax.set_title(f'Iteration {epoch}', fontsize=25)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=25, loc='upper right')
    
    plt.suptitle(f'Trajectory {traj_idx}: Mean vs Actual Comparison Across Training Epochs', 
                 fontsize=25, y=0.995)
    plt.tight_layout()
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_mean_vs_actual.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()


def _generate_best_rollout_vs_actual_comparison_plots(
    dataset,
    output_dir: Path,
    epochs_to_plot: list,
    checkpoint_files_to_use: list,
    algorithm_name: str = "unknown",
    dt: float = 0.1,
    actual_states_downsampled: Optional[np.ndarray] = None
):
    """
    Generate separate comparison plots between best rollout trajectories and actual trajectories.
    
    Creates a separate figure for each trajectory showing best rollout vs actual comparison across epochs.
    
    Args:
        dataset: Dataset containing normalized states
        output_dir: Directory containing checkpoints and where to save plots
        epochs_to_plot: List of epoch numbers to plot
        checkpoint_files_to_use: List of checkpoint file paths corresponding to epochs_to_plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
        actual_states_downsampled: Optional downsampled actual states (normalized)
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    comparison_dir.mkdir(exist_ok=True)
    
    # Load best rollout trajectories and reward sums from checkpoints
    best_rollout_trajectories_by_epoch = {}
    best_rollout_reward_sums_by_epoch = {}
    for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'best_rollout_trajectory' in checkpoint:
                best_rollout_trajectories_by_epoch[epoch] = checkpoint['best_rollout_trajectory']
            if 'best_rollout_reward_sum' in checkpoint:
                best_rollout_reward_sums_by_epoch[epoch] = checkpoint['best_rollout_reward_sum']
        except Exception as e:
            print(f"Warning: Could not load best rollout trajectory for epoch {epoch}: {e}")
    
    # Also check best_model.pt if it exists
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        try:
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            if 'best_rollout_trajectory' in best_checkpoint:
                # Find the epoch number from the checkpoint
                best_epoch = best_checkpoint.get('epoch', None)
                if best_epoch is not None and best_epoch not in best_rollout_trajectories_by_epoch:
                    best_rollout_trajectories_by_epoch[best_epoch] = best_checkpoint['best_rollout_trajectory']
                if 'best_rollout_reward_sum' in best_checkpoint and best_epoch is not None:
                    best_rollout_reward_sums_by_epoch[best_epoch] = best_checkpoint['best_rollout_reward_sum']
        except Exception as e:
            print(f"Warning: Could not load best rollout trajectory from best_model.pt: {e}")
    
    # Find the overall best rollout across all epochs (highest reward sum)
    overall_best_epoch = None
    overall_best_trajectory = None
    if best_rollout_reward_sums_by_epoch:
        overall_best_epoch = max(best_rollout_reward_sums_by_epoch, key=best_rollout_reward_sums_by_epoch.get)
        overall_best_trajectory = best_rollout_trajectories_by_epoch.get(overall_best_epoch, None)
        print(f"Overall best rollout found at epoch {overall_best_epoch} with reward sum {best_rollout_reward_sums_by_epoch[overall_best_epoch]:.6f}")
    
    if not best_rollout_trajectories_by_epoch:
        print("Warning: No best rollout trajectories found in checkpoints. Skipping best rollout plots.")
        return
    
    # Generate plots for each trajectory
    num_trajectories = len(dataset)
    for traj_idx in tqdm(range(num_trajectories), desc="Generating best rollout vs actual comparison plots"):
        _create_best_rollout_vs_actual_plot(
            traj_idx=traj_idx,
            dataset=dataset,
            best_rollout_trajectories_by_epoch=best_rollout_trajectories_by_epoch,
            epochs_to_plot=epochs_to_plot,
            output_dir=output_dir,
            algorithm_name=algorithm_name,
            dt=dt,
            actual_states_downsampled=actual_states_downsampled,
            overall_best_trajectory=overall_best_trajectory,
            overall_best_epoch=overall_best_epoch
        )
    
    print(f"Best rollout vs actual trajectory comparison plots saved to: {comparison_dir}")


def _create_best_rollout_vs_actual_plot(
    traj_idx: int,
    dataset,
    best_rollout_trajectories_by_epoch: Dict[int, torch.Tensor],
    epochs_to_plot: list,
    output_dir: Path,
    algorithm_name: str = "unknown",
    dt: float = 0.1,
    actual_states_downsampled: Optional[np.ndarray] = None,
    overall_best_trajectory: Optional[torch.Tensor] = None,
    overall_best_epoch: Optional[int] = None
):
    """
    Create a 3x3 grid plot showing best rollout trajectory vs actual trajectory comparison across epochs.
    
    Args:
        traj_idx: Index of trajectory to plot
        dataset: Dataset containing normalized states
        best_rollout_trajectories_by_epoch: Dictionary mapping epoch numbers to best rollout trajectories (normalized)
        epochs_to_plot: List of epoch numbers to plot
        output_dir: Directory to save plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
        actual_states_downsampled: Optional downsampled actual states (normalized)
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    # Get actual trajectory data
    states_norm = dataset[traj_idx]  # [seq_len, 4] - normalized
    states_norm_np = states_norm.numpy() if isinstance(states_norm, torch.Tensor) else states_norm
    
    # Denormalize actual states
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    if isinstance(state_mean, torch.Tensor):
        state_mean = state_mean.numpy()
    if isinstance(state_std, torch.Tensor):
        state_std = state_std.numpy()
    
    actual_states = states_norm_np * state_std + state_mean
    actual_states_np = actual_states  # [seq_len, 4]

    # Denormalize downsampled states if provided (they are passed in normalized space)
    actual_states_downsampled_np = None
    if actual_states_downsampled is not None:
        actual_states_downsampled_np = actual_states_downsampled * state_std + state_mean
    
    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()
    
    for plot_idx, epoch in enumerate(epochs_to_plot[:9]):  # Limit to 9 plots
        ax = axes[plot_idx]
        
        # For the last plot (9th subplot, index 8), use the overall best rollout across all epochs
        if plot_idx == 8 and overall_best_trajectory is not None:
            # Plot complete actual trajectory
            ax.plot(actual_states_np[:, 0], actual_states_np[:, 1], 'b--', 
                   markersize=2, label='Actual trajectory', linewidth=2, alpha=0.8, zorder=10)

            # Plot downsampled actual states if provided
            if actual_states_downsampled_np is not None:
                ax.plot(
                    actual_states_downsampled_np[:, 0],
                    actual_states_downsampled_np[:, 1],
                    'b.',
                    markersize=25,
                    label='Waypoints',
                    linewidth=0,
                    alpha=0.7,
                    zorder=11
                )
            
            # Plot overall best rollout trajectory
            try:
                best_rollout_norm = overall_best_trajectory
                if isinstance(best_rollout_norm, torch.Tensor):
                    best_rollout_norm = best_rollout_norm.numpy()
                
                # Denormalize best rollout trajectory
                best_rollout_denorm = best_rollout_norm * state_std + state_mean  # [seq_len, 4]
                
                # Plot overall best rollout trajectory
                ax.plot(best_rollout_denorm[:, 0], best_rollout_denorm[:, 1], 'r-', 
                       markersize=2, label='Best rollout (all epochs)', 
                       linewidth=2, alpha=0.8, zorder=9)
                
                # Plot start point
                ax.plot(actual_states_np[0, 0], actual_states_np[0, 1], 'go', 
                       markersize=8, label='Start', zorder=15)
                
                ax.set_title(f'Best Overall (Epoch {overall_best_epoch})', fontweight='bold', color='green', fontsize=25)
                print(f"Plotted overall best rollout found at epoch {overall_best_epoch}")
            except Exception as e:
                print(f"Warning: Could not plot overall best rollout trajectory: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=25)
        else:
            # For other plots, use epoch-specific best rollout
            if epoch not in best_rollout_trajectories_by_epoch:
                # If no best rollout for this epoch, show empty plot
                ax.text(0.5, 0.5, f'No best rollout for epoch {epoch}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=25)
                ax.set_title(f'Epoch {epoch}', fontsize=25)
                continue
            
            # Plot complete actual trajectory
            ax.plot(actual_states_np[:, 0], actual_states_np[:, 1], 'b--', 
                   markersize=2, label='Actual trajectory', linewidth=2, alpha=0.8, zorder=10)

            # Plot downsampled actual states if provided
            if actual_states_downsampled_np is not None:
                ax.plot(
                    actual_states_downsampled_np[:, 0],
                    actual_states_downsampled_np[:, 1],
                    'b.',
                    markersize=25,
                    label='Waypoints',
                    linewidth=0,
                    alpha=0.7,
                    zorder=11
                )
            
            # Plot best rollout trajectory for this epoch
            try:
                best_rollout_norm = best_rollout_trajectories_by_epoch[epoch]
                if isinstance(best_rollout_norm, torch.Tensor):
                    best_rollout_norm = best_rollout_norm.numpy()
                
                # Denormalize best rollout trajectory
                best_rollout_denorm = best_rollout_norm * state_std + state_mean  # [seq_len, 4]
                
                # Plot best rollout trajectory
                ax.plot(best_rollout_denorm[:, 0], best_rollout_denorm[:, 1], 'r-', 
                       markersize=2, label='Generated trajectory', 
                       linewidth=2, alpha=0.8, zorder=9)
            except Exception as e:
                print(f"Warning: Could not plot best rollout trajectory for epoch {epoch}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=25)
            
            # Plot start point
            ax.plot(actual_states_np[0, 0], actual_states_np[0, 1], 'go', 
                   markersize=8, label='Start', zorder=15)
            
            ax.set_title(f'Iteration {epoch}', fontsize=25)
        
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=25, loc='upper right')
    
    plt.suptitle(f'Trajectory {traj_idx}: Best Rollout vs Actual Comparison Across Training Epochs', 
                 fontsize=25, y=0.995)
    plt.tight_layout()
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_best_rollout_vs_actual.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()
