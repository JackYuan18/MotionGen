"""
Test script for evaluating PPO policy performance on multiple trajectories.

This script loads trajectories from selected_trajectory826.json, trains a PPO policy
for each trajectory, and records the trajectory matching error using the best policy
(lowest training loss).
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
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ActorNetwork, ValueNetwork
from data_loader import (
    TrajectoryDataset,
    load_trajectories_from_selected_file,
    prepare_training_data
)
from rewards import compute_reward
from utils import rollout_trajectory_state_feedback, DT
from train_reinforcement_ppo_vehicle import train_epoch_ppo, downsample_states
from ppo_args import add_ppo_common_args




def compute_trajectory_matching_error(
    predicted_states: torch.Tensor,
    actual_states: torch.Tensor,
    device: torch.device
) -> float:
    """
    Compute trajectory matching error between predicted and actual trajectories.
    
    Args:
        predicted_states: [seq_len_pred, 4] predicted trajectory
        actual_states: [seq_len_actual, 4] actual trajectory
    
    Returns:
        Average matching error (scalar)
    """
    predicted_states = predicted_states.to(device)
    actual_states = actual_states.to(device)
    
    # Use the curve fitting error from rewards function
    
    mse = ((predicted_states[:,:2]  - actual_states[:,:2]) ** 2).mean(dim=-1)  # [1, seq_len_pred]
    avg_error = mse.mean().item()  # Convert to scalar float
    
    return avg_error


def train_and_evaluate_trajectory(
    trajectory_states: np.ndarray,
    traj_idx: int,
    device: torch.device,
    downsample_ratio: int,
    args: argparse.Namespace
) -> Dict:
    """
    Train a PPO policy for a single trajectory and return the best matching error.
    
    Args:
        trajectory_states: [seq_len, 4] trajectory states
        traj_idx: Index of trajectory
        device: Device to run on
        args: Training arguments
    
    Returns:
        Dictionary with trajectory index, best loss, best epoch, and matching error
    """
    print(f"\n{'='*60}")
    print(f"Training trajectory {traj_idx + 1}")
    print(f"{'='*60}")
    
    # Create dataset for normalization
    states = trajectory_states.reshape(1, -1, 4)  # [1, seq_len, 4]
    train_dataset = TrajectoryDataset(states, normalize=True)
    
    # Get normalized states
    normalized_states = train_dataset[0]  # [seq_len, 4]
    normalized_states = normalized_states.unsqueeze(0)  # [1, seq_len, 4]
    initial_state = normalized_states[:, 0, :]  # [1, 4]
    
    # Keep complete states for evaluation
    actual_states_complete = normalized_states  # [1, seq_len, 4]
    
    # Downsample actual_states for training if specified
    # The reward function can handle different sequence lengths without time alignment
    if downsample_ratio > 1:
        actual_states, downsampled_indices = downsample_states(actual_states_complete, downsample_ratio)
        print(f"  Downsampling actual states: {actual_states_complete.shape[1]} -> {actual_states.shape[1]} (ratio={downsample_ratio})")
    else:
        actual_states, downsampled_indices = downsample_states(actual_states_complete, downsample_ratio)
        
    # Move to device
    initial_state = initial_state.to(device)
    actual_states = actual_states.to(device)
    actual_states_complete = actual_states_complete.to(device)
    downsampled_indices = downsampled_indices.to(device)
    
    # Create actor and value networks
    actor = ActorNetwork(
        state_dim=4,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        deterministic=False
    ).to(device)
    
    value_net = ValueNetwork(
        state_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.learning_rate)
    
    # Track best model
    best_loss = float('inf')
    best_epoch = 0
    best_rollout_reward = float('-inf')
    best_rollout_trajectory_saved = None
    train_losses = []
    
    # Training loop
    for epoch in range(args.num_epochs):
        train_loss, actor_loss, value_loss, rollout_trajectory, best_rollout_trajectory, best_rollout_reward_sum = train_epoch_ppo(
            actor=actor,
            value_net=value_net,
            initial_state=initial_state,
            actual_states=actual_states,
            downsampled_indices=downsampled_indices,
            actor_optimizer=actor_optimizer,
            value_optimizer=value_optimizer,
            device=device,
            num_rollouts=args.num_rollouts,
            dt=args.dt,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            complete_seq_len=actual_states_complete.shape[1]
        )
        
        train_losses.append(train_loss)
        
        # Track best model based on training loss
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
        
        # Track best rollout trajectory based on reward sum
        if best_rollout_reward_sum > best_rollout_reward:
            best_rollout_reward = best_rollout_reward_sum
            best_rollout_trajectory_saved = best_rollout_trajectory.clone() if isinstance(best_rollout_trajectory, torch.Tensor) else best_rollout_trajectory
        
    print(f"  Best loss: {best_loss:.6f} at epoch {best_epoch}")
    print(f"  Best rollout reward: {best_rollout_reward:.6f}")
    
    # Use the best rollout trajectory for evaluation
    if best_rollout_trajectory_saved is None:
        raise ValueError("No best rollout trajectory found. This should not happen.")
    
    # Handle tensor shape: best_rollout_trajectory_saved is [seq_len, 4] (already on CPU from train_epoch_ppo)
    if isinstance(best_rollout_trajectory_saved, torch.Tensor):
        predicted_states = best_rollout_trajectory_saved.cpu()  # [seq_len, 4]
    else:
        predicted_states = torch.from_numpy(best_rollout_trajectory_saved).cpu()  # [seq_len, 4]
    actual_states_cpu = actual_states_complete[0].cpu()  # [seq_len, 4]
    
    # Restore original deterministic setting
  
        
    # Compute matching error using complete trajectory
    matching_error = compute_trajectory_matching_error(
        predicted_states,
        actual_states_cpu,
        device
    )
    
    print(f"  Matching error: {matching_error:.6f}")
    
    # Prepare trajectories for saving (convert to numpy and then to list for JSON serialization)
    # Denormalize trajectories for saving
    state_mean = train_dataset.state_mean
    state_std = train_dataset.state_std
    if isinstance(state_mean, torch.Tensor):
        state_mean = state_mean.cpu().numpy()
    if isinstance(state_std, torch.Tensor):
        state_std = state_std.cpu().numpy()
    
    # Denormalize complete actual trajectory
    actual_states_complete_np = actual_states_complete[0].cpu().numpy()  # [seq_len, 4]
    actual_states_complete_denorm = actual_states_complete_np * state_std + state_mean
    actual_trajectory_complete = actual_states_complete_denorm.tolist()
    
    # Denormalize downsampled trajectory if it exists
    actual_trajectory_downsampled = None
    if downsample_ratio is not None and downsample_ratio > 1:
        actual_states_downsampled_np = actual_states[0].cpu().numpy()  # [downsampled_seq_len, 4]
        actual_states_downsampled_denorm = actual_states_downsampled_np * state_std + state_mean
        actual_trajectory_downsampled = actual_states_downsampled_denorm.tolist()
    
    # Denormalize predicted rollout trajectory
    predicted_states_np = predicted_states.numpy()  # [seq_len, 4]
    predicted_trajectory_denorm = predicted_states_np * state_std + state_mean
    predicted_trajectory = predicted_trajectory_denorm.tolist()
    
    return {
        'trajectory_index': traj_idx,
        'best_loss': best_loss,
        'best_rollout_reward': best_rollout_reward,
        'best_epoch': best_epoch,
        'matching_error': matching_error,
        'final_loss': train_losses[-1] if train_losses else float('inf'),
        'actual_trajectory_complete': actual_trajectory_complete,
        'actual_trajectory_downsampled': actual_trajectory_downsampled,
        'predicted_trajectory': predicted_trajectory,
        'downsample_ratio': downsample_ratio
    }


def main():
    parser = argparse.ArgumentParser(
        description='Test PPO policy on multiple trajectories and record matching errors'
    )
    
    # Add common PPO arguments
    add_ppo_common_args(parser)
    
    # Override default for num_epochs (test script uses more epochs per trajectory)
  
    
    # Test-specific arguments
    parser.add_argument(
        '--selected_trajectories_file',
        type=str,
        default='selected_trajectories826.json',
        help='Path to selected trajectories JSON file'
    )
    parser.add_argument(
        '--downsample_ratio',
        type=int,
        nargs='+',
        default=[5,10,15,20],
        help='Downsample ratio(s) for actual states during training. Can be a single value or a list. '
             'If multiple values provided, test will run for each ratio separately.'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Loading trajectories from {args.selected_trajectories_file}")
    
    # Load trajectories
    script_dir = Path(__file__).parent
    trajectories_file = Path(args.selected_trajectories_file)
    
    if not trajectories_file.is_absolute():
        if (script_dir / trajectories_file).exists():
            trajectories_file = script_dir / trajectories_file
        elif (script_dir / trajectories_file.name).exists():
            trajectories_file = script_dir / trajectories_file.name
        else:
            trajectories_file = trajectories_file.resolve()
    
    print(f"Using file path: {trajectories_file}")
    
    trajectories = load_trajectories_from_selected_file(
        file_path=str(trajectories_file),
        object_type=None,
        num_training_trajectories=None,
        min_length=10
    )
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    if len(trajectories) == 0:
        print("No trajectories found! Exiting.")
        return
    
    # Prepare training data
    print("Preparing training data...")
    states = prepare_training_data(
        trajectories,
        state_dim=4,
        dt=0.1
    )

    print(f"States shape: {states.shape}")
    
    # Train and evaluate each trajectory
   
    for downsample_ratio in args.downsample_ratio:
        results = []
        for traj_idx in range(len(states)):
            trajectory_states = states[traj_idx]  # [seq_len, 4]
            
            result = train_and_evaluate_trajectory(
                trajectory_states,
                traj_idx,
                device,
                downsample_ratio,
                args
            )
            results.append(result)
         
    # Create test_results directory
        test_results_dir = 'vehicle_test_results'
        os.makedirs(test_results_dir, exist_ok=True)
        # Generate output filename based on downsample ratio

        output_filename = f'vehicle_ppo_trajectories_{downsample_ratio}.json'
        output_path = test_results_dir + '/' + output_filename
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total trajectories tested: {len(results)}")
        
        if results:
            errors = [r['matching_error'] for r in results]
            losses = [r['best_loss'] for r in results]
            
            print(f"\nMatching Errors:")
            print(f"  Mean: {np.mean(errors):.6f}")
            print(f"  Std:  {np.std(errors):.6f}")
            print(f"  Min:  {np.min(errors):.6f}")
            print(f"  Max:  {np.max(errors):.6f}")
            
            print(f"\nBest Training Losses:")
            print(f"  Mean: {np.mean(losses):.6f}")
            print(f"  Std:  {np.std(losses):.6f}")
            print(f"  Min:  {np.min(losses):.6f}")
            print(f"  Max:  {np.max(losses):.6f}")
            
            print(f"\nResults saved to: {output_path}")

    if len(args.downsample_ratio) > 1:
        print(f"\n{'#'*60}")
        print("OVERALL SUMMARY")
        print(f"{'#'*60}")
        print(f"Tested {len(args.downsample_ratio)} downsample ratios: {args.downsample_ratio}")
        print(f"All results saved to: {test_results_dir}")
if __name__ == '__main__':
    main()
