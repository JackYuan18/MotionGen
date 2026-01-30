"""
Script to plot best rollout vs actual comparison by reading checkpoint files directly.

This script reads checkpoint files from Training_Results/rl_checkpoints and creates
comparison plots showing best rollout trajectories vs actual trajectories across epochs.
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from tqdm import tqdm

from data_loader import (
    TrajectoryDataset,
    load_trajectories_from_selected_file,
    prepare_training_data
)


def load_normalization_stats(checkpoint_dir: Path) -> tuple:
    """
    Load normalization statistics from checkpoint directory.
    
    Args:
        checkpoint_dir: Directory containing normalization_stats.json
        
    Returns:
        Tuple of (state_mean, state_std) as numpy arrays
    """
    norm_stats_path = checkpoint_dir / 'normalization_stats.json'
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")
    
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    state_mean = np.array(norm_stats['state_mean'])
    state_std = np.array(norm_stats['state_std'])
    
    return state_mean, state_std


def find_checkpoint_files(checkpoint_dir: Path, num_epochs: int) -> Dict[int, Path]:
    """
    Find all checkpoint files and map them to epoch numbers.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        num_epochs: Total number of training epochs
        
    Returns:
        Dictionary mapping epoch numbers to checkpoint file paths
    """
    available_checkpoints = {}
    
    # Find regular checkpoints
    checkpoint_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    for f in checkpoint_files:
        try:
            epoch_num = int(f.stem.split('_')[-1])
            available_checkpoints[epoch_num] = f
        except:
            continue
    
    # Always include final model
    final_model_path = checkpoint_dir / 'final_model.pt'
    if final_model_path.exists():
        available_checkpoints[num_epochs] = final_model_path
    
    return available_checkpoints


def select_epochs_to_plot(
    available_checkpoints: Dict[int, Path],
    num_epochs: int,
    num_plots: int = 9
) -> tuple:
    """
    Select epochs and corresponding checkpoint files to plot.
    
    Args:
        available_checkpoints: Dictionary mapping epoch numbers to checkpoint paths
        num_epochs: Total number of training epochs
        num_plots: Number of plots to generate (default: 9)
        
    Returns:
        Tuple of (epochs_to_plot, checkpoint_files_to_use)
    """
    epochs_to_plot = []
    checkpoint_files_to_use = []
    available_epochs = sorted(available_checkpoints.keys())
    
    if len(available_epochs) < num_plots:
        # Use all available epochs and pad with the last one
        for epoch in available_epochs:
            if epoch not in epochs_to_plot:
                epochs_to_plot.append(epoch)
                checkpoint_files_to_use.append(available_checkpoints[epoch])
        # Pad with the last available epoch
        while len(epochs_to_plot) < num_plots:
            epochs_to_plot.append(available_epochs[-1])
            checkpoint_files_to_use.append(available_checkpoints[available_epochs[-1]])
    else:
        # Select evenly spaced epochs
        target_epochs = np.linspace(1, num_epochs, num_plots, dtype=int).tolist()
        for target_epoch in target_epochs:
            closest_epoch = min(available_epochs, key=lambda x: abs(x - target_epoch))
            if closest_epoch not in epochs_to_plot:
                epochs_to_plot.append(closest_epoch)
                checkpoint_files_to_use.append(available_checkpoints[closest_epoch])
    
    return epochs_to_plot[:num_plots], checkpoint_files_to_use[:num_plots]


def create_best_rollout_vs_actual_plot(
    traj_idx: int,
    actual_states_np: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    best_rollout_trajectories_by_epoch: Dict[int, torch.Tensor],
    epochs_to_plot: List[int],
    output_dir: Path,
    algorithm_name: str = "PPO",
    dt: float = 0.1,
    actual_states_downsampled_np: Optional[np.ndarray] = None,
    overall_best_trajectory: Optional[torch.Tensor] = None,
    overall_best_epoch: Optional[int] = None
):
    """
    Create a 3x3 grid plot showing best rollout trajectory vs actual trajectory comparison across epochs.
    
    Args:
        traj_idx: Index of trajectory to plot
        actual_states_np: [seq_len, 4] actual trajectory states (denormalized)
        state_mean: [4] normalization mean
        state_std: [4] normalization std
        best_rollout_trajectories_by_epoch: Dictionary mapping epoch numbers to best rollout trajectories (normalized)
        epochs_to_plot: List of epoch numbers to plot
        output_dir: Directory to save plot
        algorithm_name: Name of the reinforcement learning algorithm
        dt: Discrete time step value
        actual_states_downsampled_np: Optional [downsampled_seq_len, 4] downsampled waypoints (denormalized)
        overall_best_trajectory: Optional overall best rollout trajectory across all epochs (normalized)
        overall_best_epoch: Optional epoch number for overall best trajectory
    """
    comparison_dir = output_dir / 'trajectory_comparisons'
    comparison_dir.mkdir(exist_ok=True)
    
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
    
    # Create a single legend below all subplots with 4 columns
    # Get handles and labels from the first subplot that has data
    handles, labels = None, None
    for plot_idx in range(9):
        if plot_idx < len(epochs_to_plot):
            epoch = epochs_to_plot[plot_idx]
            if epoch in best_rollout_trajectories_by_epoch or (plot_idx == 8 and overall_best_trajectory is not None):
                ax = axes[plot_idx]
                handles, labels = ax.get_legend_handles_labels()
                if handles:  # Only use if we got handles
                    break
    
    # If we have handles, create figure-level legend below all subplots
    if handles and labels:
        # Remove duplicate labels while preserving order
        seen = set()
        unique_handles = []
        unique_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        
        fig.legend(unique_handles, unique_labels, 
                   loc='lower center', ncol=4, fontsize=25, 
                   framealpha=0.95, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(f'Trajectory {traj_idx}: Best Rollout vs Actual Comparison Across Training Epochs', 
                 fontsize=25, y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.995])  # Leave space at bottom for legend
    
    # Format dt value for filename (replace decimal point with underscore)
    dt_str = f"{dt:.3f}".replace('.', '_')
    filename = f'trajectory_{traj_idx:04d}_{algorithm_name}_dt{dt_str}_best_rollout_vs_actual.png'
    plt.savefig(comparison_dir / filename, 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot best rollout vs actual comparison from checkpoint files'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='Training_Results/rl_checkpoints',
        help='Directory containing checkpoint files (default: Training_Results/rl_checkpoints)'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        required=True,
        help='Total number of training epochs'
    )
    
    parser.add_argument(
        '--trajectory_file',
        type=str,
        default=None,
        help='Path to selected trajectories JSON file (if not provided, will try to infer from checkpoint data)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as checkpoint_dir)'
    )
    
    parser.add_argument(
        '--algorithm_name',
        type=str,
        default='PPO',
        help='Name of the algorithm (default: PPO)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='Time step value (default: 0.1)'
    )
    
    parser.add_argument(
        '--downsample_ratio',
        type=int,
        default=None,
        help='Downsample ratio used during training (for plotting waypoints)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = script_dir / checkpoint_dir
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load normalization statistics
    print("Loading normalization statistics...")
    try:
        state_mean, state_std = load_normalization_stats(checkpoint_dir)
        print(f"Loaded normalization stats: mean={state_mean}, std={state_std}")
    except Exception as e:
        print(f"Error loading normalization stats: {e}")
        return
    
    # Find checkpoint files
    print("Finding checkpoint files...")
    available_checkpoints = find_checkpoint_files(checkpoint_dir, args.num_epochs)
    if not available_checkpoints:
        print("Error: No checkpoint files found!")
        return
    
    print(f"Found {len(available_checkpoints)} checkpoint files")
    
    # Select epochs to plot
    epochs_to_plot, checkpoint_files_to_use = select_epochs_to_plot(available_checkpoints, args.num_epochs)
    print(f"Selected epochs to plot: {epochs_to_plot}")
    
    # Load best rollout trajectories from checkpoints
    print("Loading best rollout trajectories from checkpoints...")
    best_rollout_trajectories_by_epoch = {}
    best_rollout_reward_sums_by_epoch = {}
    downsampled_indices = None
    
    for epoch, checkpoint_path in zip(epochs_to_plot, checkpoint_files_to_use):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'best_rollout_trajectory' in checkpoint:
                best_rollout_trajectories_by_epoch[epoch] = checkpoint['best_rollout_trajectory']
            if 'best_rollout_reward_sum' in checkpoint:
                best_rollout_reward_sums_by_epoch[epoch] = checkpoint['best_rollout_reward_sum']
            # Load downsampled_indices from first checkpoint that has it
            if downsampled_indices is None and 'downsampled_indices' in checkpoint:
                downsampled_indices = checkpoint['downsampled_indices']
        except Exception as e:
            print(f"Warning: Could not load best rollout trajectory for epoch {epoch}: {e}")
    
    # Also check best_model.pt if it exists
    best_model_path = checkpoint_dir / 'best_model.pt'
    if best_model_path.exists():
        try:
            best_checkpoint = torch.load(best_model_path, map_location='cpu')
            if 'best_rollout_trajectory' in best_checkpoint:
                best_epoch = best_checkpoint.get('epoch', None)
                if best_epoch is not None and best_epoch not in best_rollout_trajectories_by_epoch:
                    best_rollout_trajectories_by_epoch[best_epoch] = best_checkpoint['best_rollout_trajectory']
                if 'best_rollout_reward_sum' in best_checkpoint and best_epoch is not None:
                    best_rollout_reward_sums_by_epoch[best_epoch] = best_checkpoint['best_rollout_reward_sum']
            # Load downsampled_indices if not already loaded
            if downsampled_indices is None and 'downsampled_indices' in best_checkpoint:
                downsampled_indices = best_checkpoint['downsampled_indices']
        except Exception as e:
            print(f"Warning: Could not load best rollout trajectory from best_model.pt: {e}")
    
    # Extract waypoints using downsampled_indices if available
    if downsampled_indices is not None:
        # Convert to numpy if tensor
        if isinstance(downsampled_indices, torch.Tensor):
            downsampled_indices_np = downsampled_indices.cpu().numpy()
        else:
            downsampled_indices_np = downsampled_indices
        
        # Handle shape: could be [1, num_indices] or [num_indices]
        if len(downsampled_indices_np.shape) == 2:
            downsampled_indices_np = downsampled_indices_np[0]  # Take first row if batch dimension exists
        
        print(f"Found downsampled_indices with {len(downsampled_indices_np)} waypoints")
    else:
        print("Warning: No downsampled_indices found in checkpoints. Waypoints will not be plotted.")
        downsampled_indices_np = None
    
    # Find the overall best rollout across all epochs (highest reward sum)
    overall_best_epoch = None
    overall_best_trajectory = None
    if best_rollout_reward_sums_by_epoch:
        overall_best_epoch = max(best_rollout_reward_sums_by_epoch, key=best_rollout_reward_sums_by_epoch.get)
        overall_best_trajectory = best_rollout_trajectories_by_epoch.get(overall_best_epoch, None)
        print(f"Overall best rollout found at epoch {overall_best_epoch} with reward sum {best_rollout_reward_sums_by_epoch[overall_best_epoch]:.6f}")
    
    if not best_rollout_trajectories_by_epoch:
        print("Warning: No best rollout trajectories found in checkpoints. Cannot generate plots.")
        return
    
    # Load actual trajectory data
    print("Loading actual trajectory data...")
    if args.trajectory_file is not None:
        trajectory_file = Path(args.trajectory_file)
        if not trajectory_file.is_absolute():
            if (script_dir / trajectory_file).exists():
                trajectory_file = script_dir / trajectory_file
            elif (script_dir / trajectory_file.name).exists():
                trajectory_file = script_dir / trajectory_file.name
            else:
                trajectory_file = trajectory_file.resolve()
        
        print(f"Loading trajectories from: {trajectory_file}")
        trajectories = load_trajectories_from_selected_file(
            file_path=str(trajectory_file),
            object_type=None,
            num_training_trajectories=1,
            min_length=10
        )
        
        if len(trajectories) == 0:
            print("Error: No trajectories found in file!")
            return
        
        # Prepare training data
        states = prepare_training_data(trajectories, state_dim=4, dt=0.1)
        
        # Create dataset for normalization
        train_dataset = TrajectoryDataset(states, normalize=True)
        
        # Get normalized states
        normalized_states = train_dataset[0]  # [seq_len, 4]
        normalized_states_np = normalized_states.numpy() if isinstance(normalized_states, torch.Tensor) else normalized_states
        
        # Denormalize actual states
        actual_states_np = normalized_states_np * state_std + state_mean  # [seq_len, 4]
        
        # Extract waypoints using downsampled_indices if available
        actual_states_downsampled_np = None
        if downsampled_indices_np is not None:
            downsampled_indices_np = np.concatenate([downsampled_indices_np, [len(actual_states_np)-1]])

            # Use downsampled_indices to extract waypoints from complete trajectory
            actual_states_downsampled_np = actual_states_np[downsampled_indices_np]  # [num_waypoints, 4]
            print(f"Extracted {len(actual_states_downsampled_np)} waypoints using downsampled_indices")
        
    else:
        # Try to get actual trajectory from first checkpoint (if it has rollout_trajectories)
        print("No trajectory file provided. Attempting to infer from checkpoint data...")
        try:
            first_checkpoint = torch.load(checkpoint_files_to_use[0], map_location='cpu')
            if 'rollout_trajectories' in first_checkpoint:
                # Use the first rollout trajectory as reference (it should be close to actual)
                # This is a fallback - ideally we should have the actual trajectory
                rollout_trajs = first_checkpoint['rollout_trajectories']
                if isinstance(rollout_trajs, torch.Tensor):
                    rollout_trajs = rollout_trajs.numpy()
                if len(rollout_trajs.shape) == 3:
                    # [num_trajectories, seq_len, 4]
                    normalized_states_np = rollout_trajs[0]  # Use first rollout
                else:
                    normalized_states_np = rollout_trajs
                
                actual_states_np = normalized_states_np * state_std + state_mean
                
                # Extract waypoints using downsampled_indices if available
                actual_states_downsampled_np = None
                if downsampled_indices_np is not None:
                    actual_states_downsampled_np = actual_states_np[downsampled_indices_np]  # [num_waypoints, 4]
                    print(f"Extracted {len(actual_states_downsampled_np)} waypoints using downsampled_indices")
                
                print("Warning: Using rollout trajectory as reference (not actual trajectory). Provide --trajectory_file for accurate plots.")
            else:
                print("Error: Cannot infer actual trajectory from checkpoints. Please provide --trajectory_file")
                return
        except Exception as e:
            print(f"Error inferring trajectory from checkpoint: {e}")
            print("Please provide --trajectory_file argument")
            return
    
    print(f"Actual trajectory shape: {actual_states_np.shape}")
    
    # Generate plots
    print("\nGenerating best rollout vs actual comparison plots...")
    create_best_rollout_vs_actual_plot(
        traj_idx=0,
        actual_states_np=actual_states_np,
        state_mean=state_mean,
        state_std=state_std,
        best_rollout_trajectories_by_epoch=best_rollout_trajectories_by_epoch,
        epochs_to_plot=epochs_to_plot,
        output_dir=output_dir,
        algorithm_name=args.algorithm_name,
        dt=args.dt,
        actual_states_downsampled_np=actual_states_downsampled_np,
        overall_best_trajectory=overall_best_trajectory,
        overall_best_epoch=overall_best_epoch
    )
    
    comparison_dir = output_dir / 'trajectory_comparisons'
    print(f"\nBest rollout vs actual trajectory comparison plots saved to: {comparison_dir}")


if __name__ == '__main__':
    main()
