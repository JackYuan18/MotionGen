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
    
    # Action history buffer for transformer input
    action_history_list = []  # List of [batch_size, 2] tensors
    
    for t in range(seq_len):
        # Prepare action history for transformer
        if len(action_history_list) > 0:
            # Stack recent actions (most recent last)
            action_history = torch.stack(action_history_list, dim=1)  # [batch, len(action_history_list), 2]
        else:
            # No previous actions
            action_history = None
        
        # Get mean action from actor (deterministic, no sampling)
        mean, _ = actor.forward(actual_trajectory, current_state, action_history)
        # mean: [batch, 2]
        
        # Use mean as the deterministic action
        deterministic_action = mean
        
        # Update action history
        action_history_list.append(deterministic_action.clone())
        # Limit history size
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
                next_state = rk4_step(current_state, deterministic_action, dt, simple_car_dynamics_torch)
            else:
                # Use Euler integration
                state_derivative = simple_car_dynamics_torch(current_state, deterministic_action, dt)
                next_state = current_state + dt * state_derivative
            
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
    dt: float = 0.1,
    include_baseline: bool = True,
    include_deterministic_mean: bool = True,
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
        
        # If we still don't have 9 unique epochs, fill with evenly spaced from available
        if len(epochs_to_plot) < 9:
            remaining_slots = 9 - len(epochs_to_plot)
            # Get epochs we haven't used yet
            unused_epochs = [e for e in available_epochs if e not in epochs_to_plot]
            if unused_epochs:
                # Add evenly spaced from unused epochs
                indices = np.linspace(0, len(unused_epochs) - 1, min(remaining_slots, len(unused_epochs)), dtype=int)
                for idx in indices:
                    epoch = unused_epochs[idx]
                    epochs_to_plot.append(epoch)
                    checkpoint_files_to_use.append(available_checkpoints[epoch])
            # If still not enough, pad with last epoch
            while len(epochs_to_plot) < 9:
                epochs_to_plot.append(epochs_to_plot[-1])
                checkpoint_files_to_use.append(checkpoint_files_to_use[-1])
    
    epochs_to_plot = epochs_to_plot[:9]
    checkpoint_files_to_use = checkpoint_files_to_use[:9]
    
    # Check if best_model.pt exists and replace 9th subplot with best model
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists():
        try:
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            best_epoch = best_checkpoint.get('epoch', num_epochs)
            # Replace 9th subplot (index 8) with best model
            epochs_to_plot[8] = best_epoch
            checkpoint_files_to_use[8] = best_model_path
            print(f"Using best model (epoch {best_epoch}) for 9th subplot")
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
    
    print(f"Plotting comparisons at epochs: {epochs_to_plot}")
    
    # Load rollout trajectories from checkpoints (if available)
    rollout_trajectories_by_epoch = {}
    for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load on CPU first
            # Check for both old format (single trajectory) and new format (multiple trajectories)
            if 'rollout_trajectories' in checkpoint:
                # New format: multiple rollout trajectories [num_trajectories, seq_len, 4]
                rollout_trajectories_by_epoch[epoch] = checkpoint['rollout_trajectories']
            elif 'rollout_trajectory' in checkpoint:
                # Old format: single rollout trajectory [seq_len, 4] - convert to [1, seq_len, 4]
                single_traj = checkpoint['rollout_trajectory']
                if len(single_traj.shape) == 2:
                    single_traj = single_traj.unsqueeze(0)  # Add batch dimension
                rollout_trajectories_by_epoch[epoch] = single_traj
        except Exception as e:
            print(f"Warning: Could not load rollout trajectories from epoch {epoch}: {e}")
            continue
    
    # If best model was used for 9th subplot, make sure it's loaded
    best_model_path = output_dir / 'best_model.pt'
    if best_model_path.exists() and best_model_path in checkpoint_files_to_use:
        best_epoch = epochs_to_plot[8]  # 9th subplot (index 8)
        if best_epoch not in rollout_trajectories_by_epoch:
            try:
                best_checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'rollout_trajectories' in best_checkpoint:
                    rollout_trajectories_by_epoch[best_epoch] = best_checkpoint['rollout_trajectories']
                elif 'rollout_trajectory' in best_checkpoint:
                    single_traj = best_checkpoint['rollout_trajectory']
                    if len(single_traj.shape) == 2:
                        single_traj = single_traj.unsqueeze(0)
                    rollout_trajectories_by_epoch[best_epoch] = single_traj
            except Exception as e:
                print(f"Warning: Could not load best model rollout trajectories: {e}")
    
    # Check if we have saved rollout trajectories
    if len(rollout_trajectories_by_epoch) == 0:
        print("Error: No saved rollout trajectories found in checkpoints.")
        print("Cannot generate comparison plots without saved trajectories.")
        print("Make sure training was run with checkpoint saving enabled.")
        return
    
    num_trajs_per_epoch = rollout_trajectories_by_epoch[list(rollout_trajectories_by_epoch.keys())[0]].shape[0]
    print(f"Found saved rollout trajectories for {len(rollout_trajectories_by_epoch)} epochs. "
          f"Each epoch has {num_trajs_per_epoch} trajectories. Using saved trajectories.")
    
    # Load actor models for deterministic mean rollout if requested
    actor_models_by_epoch = {}
    if include_deterministic_mean and ActorNetwork is not None:
        print("Loading actor models for deterministic mean rollout...")
        for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'actor_state_dict' in checkpoint:
                    # Get model parameters from checkpoint
                    hidden_dim = checkpoint.get('hidden_dim', 128)
                    num_layers = checkpoint.get('num_layers', 2)
                    # action_history_len defaults to 20 in ActorNetwork, but check checkpoint first
                    action_history_len = checkpoint.get('action_history_len', 20)
                    
                    # Create actor model
                    actor = ActorNetwork(
                        state_dim=4,
                        action_dim=2,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        deterministic=True,  # Use deterministic mode for mean action
                        action_history_len=action_history_len
                    )
                    actor.load_state_dict(checkpoint['actor_state_dict'])
                    actor.eval()  # Set to evaluation mode
                    actor_models_by_epoch[epoch] = actor
            except Exception as e:
                print(f"Warning: Could not load actor model from epoch {epoch}: {e}")
                continue
    
    # If best model was used for 9th subplot, make sure actor model is loaded
    if include_deterministic_mean and ActorNetwork is not None and best_model_path.exists() and best_model_path in checkpoint_files_to_use:
        best_epoch = epochs_to_plot[8]  # 9th subplot (index 8)
        if best_epoch not in actor_models_by_epoch:
            try:
                best_checkpoint = torch.load(best_model_path, map_location='cpu')
                if 'actor_state_dict' in best_checkpoint:
                    hidden_dim = best_checkpoint.get('hidden_dim', 128)
                    num_layers = best_checkpoint.get('num_layers', 2)
                    action_history_len = best_checkpoint.get('action_history_len', 20)
                    
                    actor = ActorNetwork(
                        state_dim=4,
                        action_dim=2,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        deterministic=True,
                        action_history_len=action_history_len
                    )
                    actor.load_state_dict(best_checkpoint['actor_state_dict'])
                    actor.eval()
                    actor_models_by_epoch[best_epoch] = actor
            except Exception as e:
                print(f"Warning: Could not load best model actor: {e}")
    
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
            dt,
            include_baseline,
            include_deterministic_mean,
            actor_models_by_epoch,
            actual_states_downsampled
        )
    
    print(f"Trajectory comparison plots saved to: {comparison_dir}")
    
    # Generate separate mean vs actual trajectory comparison plots
    if include_deterministic_mean and len(actor_models_by_epoch) > 0:
        print("\nGenerating mean trajectory vs actual trajectory comparison plots...")
        _generate_mean_vs_actual_comparison_plots(
            dataset=dataset,
            output_dir=output_dir,
            actor_models_by_epoch=actor_models_by_epoch,
            epochs_to_plot=epochs_to_plot,
            algorithm_name=algorithm_name,
            dt=dt
        )


def _create_epoch_comparison_plot_with_saved_trajectories(
    traj_idx: int,
    dataset,
    rollout_trajectories_by_epoch: Dict[int, torch.Tensor],
    epochs_to_plot: list,
    output_dir: Path,
    algorithm_name: str = "unknown",
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
    
    # Compute baseline trajectory if requested
    baseline_trajectory = None
    if include_baseline and actual_states_np.shape[0] >= 2:
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
                                fontweight='bold', color='green')
                else:
                    ax.set_title(f'Epoch {epoch} (BEST, {num_trajectories} rollouts)', 
                                fontweight='bold', color='green')
            except:
                ax.set_title(f'Epoch {epoch} (BEST, {num_trajectories} rollouts)', 
                            fontweight='bold', color='green')
        else:
            ax.set_title(f'Epoch {epoch} ({num_trajectories} rollouts)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=8)
        
        # Plot all rollout trajectories (up to 10)
        for traj_idx_plot in range(num_trajectories):
            traj = predicted_states_np[traj_idx_plot]  # [seq_len, 4]
            ax.plot(traj[:, 0], traj[:, 1], 'r-o', 
                   markersize=1, linewidth=2, alpha=0.4, zorder=5)
    
    plt.suptitle(f'Trajectory {traj_idx}: Evolution Across Training Epochs (10 Rollout Trajectories)', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_epochs.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()


def _generate_mean_vs_actual_comparison_plots(
    dataset,
    output_dir: Path,
    actor_models_by_epoch: Dict[int, nn.Module],
    epochs_to_plot: list,
    algorithm_name: str = "unknown",
    dt: float = 0.1
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
            dt=dt
        )
    
    print(f"Mean vs actual trajectory comparison plots saved to: {comparison_dir}")


def _create_mean_vs_actual_plot(
    traj_idx: int,
    dataset,
    actor_models_by_epoch: Dict[int, nn.Module],
    epochs_to_plot: list,
    output_dir: Path,
    algorithm_name: str = "unknown",
    dt: float = 0.1
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
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Epoch {epoch}')
            continue
        
        ax = axes[plot_idx]
        
        # Plot complete actual trajectory
        ax.plot(actual_states_np[:, 0], actual_states_np[:, 1], 'b--', 
               markersize=2, label='Actual (complete)', linewidth=2, alpha=0.8, zorder=10)
        
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
                       markersize=2, label='Deterministic mean', 
                       linewidth=2, alpha=0.8, zorder=9)
        except Exception as e:
            print(f"Warning: Could not rollout deterministic mean trajectory for epoch {epoch}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        # Plot start point
        ax.plot(actual_states_np[0, 0], actual_states_np[0, 1], 'go', 
               markersize=8, label='Start', zorder=15)
        
        ax.set_title(f'Epoch {epoch}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=10, loc='upper right')
    
    plt.suptitle(f'Trajectory {traj_idx}: Mean vs Actual Comparison Across Training Epochs', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_mean_vs_actual.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()
