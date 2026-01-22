"""
Reinforcement learning training script for trajectory to control neural network.

This script trains a neural network control policy using policy gradient methods
(REINFORCE) to minimize the difference between generated and actual trajectories.
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
from tqdm import tqdm
from typing import Callable, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ActorNetwork, ValueNetwork
from data_loader import (
    TrajectoryDataset,
    load_trajectories_from_directory,
    load_trajectories_from_selected_file,
    prepare_training_data
)
from visualize_trajectories import generate_trajectory_comparison_plots
from rewards import compute_reward
from utils import rollout_trajectory_state_feedback, simple_car_dynamics_torch, DT
from ppo_args import add_ppo_common_args, add_trajectory_data_args


def downsample_states(states: torch.Tensor, ratio: int) -> torch.Tensor:
    """
    Downsample states by keeping every Nth state.
    
    Args:
        states: [batch_size, seq_len, state_dim] or [seq_len, state_dim] tensor
        ratio: Downsample ratio (keep every Nth state, starting from index 0)
    
    Returns:
        Downsampled states with shape [batch_size, new_seq_len, state_dim] or [new_seq_len, state_dim]
    """
    if ratio <= 1:
        return states
    
    if len(states.shape) == 2:
        # [seq_len, state_dim]
        return states[::ratio]
    else:
        # [batch_size, seq_len, state_dim]
        return states[:, ::ratio, :]


def train_epoch_ppo(
    actor: nn.Module,
    value_net: nn.Module,
    initial_state: torch.Tensor,
    actual_states: torch.Tensor,
    actor_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    device: torch.device,
    num_rollouts: int = 10,
    dt: float = DT,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.002,
    gamma: float = 0.99,
    use_history: bool = True,
    history_length: int = 10,
    complete_seq_len: Optional[int] = None
) -> Tuple[float, torch.Tensor]:
    """
    Train for one epoch using Proximal Policy Optimization (PPO) with state-feedback control.
    
    For a given control policy, rollout multiple trajectories from the same initial state,
    then perform multiple PPO update epochs on the collected data.
    
    Args:
        actor: ActorNetwork policy (stochastic policy)
        value_net: ValueNetwork for state value estimation
        initial_state: [1, 4] initial state (single trajectory)
        actual_states: [1, seq_len, 4] actual trajectory states (single trajectory)
        actor_optimizer: Optimizer for actor
        value_optimizer: Optimizer for value network
        device: Device to run on
        num_rollouts: Number of rollouts to perform per training step
        dt: Time step
        ppo_epochs: Number of PPO update epochs on collected data
        clip_epsilon: Clipping parameter for PPO (default: 0.2)
        value_coef: Coefficient for value loss (default: 0.5)
        entropy_coef: Coefficient for entropy bonus (default: 0.01)
        gamma: Discount factor for computing returns (default: 0.99)
        use_history: If True, use sliding window of recent states for context
        history_length: Length of history window when use_history=True
    
    Returns:
        Tuple of (average_loss, rollout_trajectories)
    """
    actor.train()
    value_net.train()
    # Use complete_seq_len if provided (before downsampling), otherwise use actual_states length
    if complete_seq_len is not None:
        seq_len = complete_seq_len
    else:
        seq_len = actual_states.shape[1]
    
    # Expand initial_state to batch_size = num_rollouts for parallel rollouts
    initial_state_batch = initial_state.repeat(num_rollouts, 1)  # [num_rollouts, 4]
    
    # Expand actual_states to match batch size for rollout
    actual_states_batch = actual_states.repeat(num_rollouts, 1, 1)  # [num_rollouts, seq_len, 4]
    
    # Rollout multiple trajectories using state-feedback control policy
    with torch.no_grad():
        predicted_states, sampled_controls, old_log_probs = rollout_trajectory_state_feedback(
            actor=actor,
            initial_state=initial_state_batch,
            actual_trajectory=actual_states_batch,
            seq_len=seq_len,
            dt=dt,
            use_history=use_history,
            history_length=history_length
        )
        # predicted_states: [num_rollouts, predicted_seq_len, 4]
        # sampled_controls: [num_rollouts, predicted_seq_len, 2]
        # old_log_probs: [num_rollouts, predicted_seq_len]
    
    # Get the actual sequence length from predicted_states (may differ from actual_states due to dt)
    predicted_seq_len = predicted_states.shape[1]
    actual_seq_len = actual_states_batch.shape[1]
    
    with torch.no_grad():
        # Compute step-wise rewards for each rollout
        # Note: compute_reward can handle different sequence lengths between predicted and actual
        step_rewards = compute_reward(predicted_states, actual_states_batch, actions=sampled_controls)  # [num_rollouts, predicted_seq_len]
        
        # Compute values for all states in trajectories
        # Flatten for batch processing: [num_rollouts * predicted_seq_len]
        states_flat = predicted_states.reshape(-1, 4)  # [num_rollouts * predicted_seq_len, 4]
        actual_traj_flat = actual_states_batch.repeat_interleave(predicted_seq_len, dim=0)  # [num_rollouts * predicted_seq_len, actual_seq_len, 4]
        
        old_values = value_net(actual_traj_flat, states_flat).squeeze(-1)  # [num_rollouts * predicted_seq_len]
        old_values = old_values.reshape(num_rollouts, predicted_seq_len)  # [num_rollouts, predicted_seq_len]
        
        # Compute returns using discounted cumulative rewards (backward from end)
        # For PPO, we compute returns as discounted sum of future rewards
        returns = torch.zeros_like(step_rewards)  # [num_rollouts, predicted_seq_len]
        returns[:, -1] = step_rewards[:, -1]  # Last step return is just its reward
        
        # Compute discounted returns backwards
        for t in range(predicted_seq_len - 2, -1, -1):
            returns[:, t] = step_rewards[:, t] + gamma * returns[:, t + 1]
        
        # Compute advantages (returns - values)
        advantages = returns - old_values  # [num_rollouts, predicted_seq_len]
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old log probs (sum over time steps for each rollout)
        old_log_probs_sum = old_log_probs.sum(dim=1)  # [num_rollouts]
    
    # Prepare data for PPO updates
    # Flatten all trajectories into a single batch
    total_samples = num_rollouts * predicted_seq_len
    
    # Repeat advantages and returns for each time step
    advantages_flat = advantages.reshape(-1)  # [num_rollouts * predicted_seq_len]
    returns_flat = returns.reshape(-1)  # [num_rollouts * predicted_seq_len]
    old_log_probs_flat = old_log_probs.reshape(-1)  # [num_rollouts * predicted_seq_len]
    
    # Store rollout trajectories for visualization
    num_trajectories_to_save = min(10, num_rollouts)
    rollout_trajectories = predicted_states[:num_trajectories_to_save].detach().cpu()
    
    # PPO update epochs
    total_actor_loss = 0.0
    total_value_loss = 0.0
    
    for ppo_epoch in range(ppo_epochs):
        # Shuffle data
        indices = torch.randperm(total_samples, device=device)
        
        # Process in mini-batches (optional, but can help with stability)
        batch_size = min(64, total_samples)
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch data
            batch_states = states_flat[batch_indices]  # [batch_size, 4]
            batch_actual_traj = actual_traj_flat[batch_indices]  # [batch_size, actual_seq_len, 4]
            batch_actions = sampled_controls.reshape(-1, 2)[batch_indices]  # [batch_size, 2]
            batch_advantages = advantages_flat[batch_indices]  # [batch_size]
            batch_returns = returns_flat[batch_indices]  # [batch_size]
            batch_old_log_probs = old_log_probs_flat[batch_indices]  # [batch_size]
            
            # Get current policy log probs and values
            # Need to recompute log probs for the same actions
            # Note: We don't have action_history here, so pass None (model will use zeros)
            mean, log_std = actor.forward(batch_actual_traj, batch_states, action_history=None)
            std = torch.exp(log_std)
            std = torch.clamp(std, min=1e-6, max=10.0)
            
            # Compute log prob for the stored actions
            normal = Normal(mean, std)
            new_log_probs = normal.log_prob(batch_actions).sum(dim=1)  # [batch_size]
            
            # Compute value estimates
            new_values = value_net(batch_actual_traj, batch_states).squeeze(-1)  # [batch_size]
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - batch_old_log_probs)  # [batch_size]
            
            # Clipped surrogate objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(new_values, batch_returns)
            
            # Entropy bonus (encourage exploration)
            entropy = normal.entropy().sum(dim=1).mean()  # [batch_size] -> scalar
            entropy_bonus = -entropy_coef * entropy
            
            # Total actor loss
            total_actor_loss_batch = actor_loss + entropy_bonus
            
            # Update actor
            actor_optimizer.zero_grad()
            total_actor_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_optimizer.step()
            
            # Update value network
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            value_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
    
    # Average losses
    num_batches = ppo_epochs * (total_samples // batch_size + (1 if total_samples % batch_size > 0 else 0))
    avg_actor_loss = total_actor_loss / num_batches if num_batches > 0 else 0.0
    avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
    
    # Normalize losses by sequence length to make them per-timestep
    # This makes losses comparable across different sequence lengths
    if predicted_seq_len > 0:
        normalized_actor_loss = avg_actor_loss / predicted_seq_len
        normalized_value_loss = avg_value_loss / predicted_seq_len
    else:
        normalized_actor_loss = avg_actor_loss
        normalized_value_loss = avg_value_loss
    
    avg_loss = normalized_actor_loss + value_coef * normalized_value_loss
    
    return avg_loss, rollout_trajectories


def main():
    parser = argparse.ArgumentParser(
        description='Train neural network control policy using reinforcement learning (PPO)'
    )
    # Add common PPO arguments
    add_ppo_common_args(parser)
    
    # Add trajectory data arguments
    add_trajectory_data_args(parser)
    
    # Override defaults for training script
    parser.set_defaults(
        selected_trajectories_file='selected_trajectories1.json'
    )
    
    # Training-specific arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Training_Results/rl_checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("Training with Proximal Policy Optimization (PPO)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectories from selected_trajectories.json
    print(f"Loading trajectories from {args.selected_trajectories_file}...")
    selected_file_path = Path(args.selected_trajectories_file)
    
    # Resolve path relative to script directory (where the file should be)
    script_dir = Path(__file__).parent
    
    if not selected_file_path.is_absolute():
        # Try relative to script directory first
        if (script_dir / selected_file_path).exists():
            selected_file_path = script_dir / selected_file_path
        # If that doesn't work, try just the filename in script directory
        elif (script_dir / selected_file_path.name).exists():
            selected_file_path = script_dir / selected_file_path.name
        # Otherwise, try as-is (might be relative to current working directory)
        else:
            selected_file_path = selected_file_path.resolve()
    else:
        selected_file_path = selected_file_path.resolve()
    
    print(f"Using file path: {selected_file_path}")
    print(f"File exists: {selected_file_path.exists()}")
    
    trajectories = load_trajectories_from_selected_file(
        file_path=str(selected_file_path),
        object_type=args.object_type,
        num_training_trajectories=1,
        min_length=args.min_length
    )
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    if len(trajectories) == 0:
        print("No trajectories found! Exiting.")
        return
    
    # Prepare training data (states only, no controls)
    print("Preparing training data...")
    states = prepare_training_data(
        trajectories,
        state_dim=4,
        dt=0.1
    )
    
    print(f"States shape: {states.shape}")
    print(f"Initial States before normalization: {states[0,0,:]}")
    
    # For reinforcement learning, use all loaded trajectories
    # If multiple trajectories, we'll train on each one sequentially
    print(f"Using {len(states)} trajectory(ies) for RL training...")
    
    # Create dataset for normalization using all trajectories
    train_dataset = TrajectoryDataset(states, normalize=True)
    
    # Save normalization statistics
    norm_stats = train_dataset.get_normalization_stats()
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump({
            'state_mean': norm_stats['state_mean'].tolist(),
            'state_std': norm_stats['state_std'].tolist()
        }, f, indent=2)
    
    # Get normalized states for training
    # Use the first trajectory for training (can be extended to train on multiple)
    normalized_states = train_dataset[0]  # [seq_len, 4]
    normalized_states = normalized_states.unsqueeze(0)  # [1, seq_len, 4]
    initial_state = normalized_states[:, 0, :]  # [1, 4]
    
    # Keep complete states for visualization
    actual_states_complete = normalized_states  # [1, seq_len, 4]
    
    # Downsample actual_states for training if specified
    # The reward function can handle different sequence lengths without time alignment
    if args.downsample_ratio > 1:
        actual_states = downsample_states(actual_states_complete, args.downsample_ratio)
        print(f"Downsampling actual states: {actual_states_complete.shape[1]} -> {actual_states.shape[1]} (ratio={args.downsample_ratio})")
    else:
        actual_states = actual_states_complete
    
    print(f"Initial State after normalization: {initial_state[0,:]}")
    if len(states) > 1:
        print(f"Note: {len(states)} trajectories loaded. Training on first trajectory only.")
        print("To train on multiple trajectories, modify the training loop to iterate over them.")
    
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Actual states shape (for training): {actual_states.shape}")
    print(f"Actual states shape (complete): {actual_states_complete.shape}")
    
    # Create actor network (deterministic by default for PPO)
    print("Creating ActorNetwork...")
    actor = ActorNetwork(
        state_dim=4,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        deterministic=False  # Use deterministic policy by default for PPO
    )
    actor = actor.to(device)
    
    # When using deterministic policy, use only 1 rollout (all would be identical)
    if actor.deterministic:
        if args.num_rollouts > 1:
            print(f"Note: Deterministic policy enabled. Setting num_rollouts to 1 (was {args.num_rollouts})")
            args.num_rollouts = 1
    
    # Create value network
    print("Creating ValueNetwork...")
    value_net = ValueNetwork(
        state_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    value_net = value_net.to(device)
    
    # Count parameters
    num_params_actor = sum(p.numel() for p in actor.parameters())
    num_params_value = sum(p.numel() for p in value_net.parameters())
    print(f"ActorNetwork has {num_params_actor:,} parameters")
    print(f"ValueNetwork has {num_params_value:,} parameters")
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    best_loss = float('inf')
    best_epoch = 0
    
    print("\nStarting reinforcement learning training...")
    print("Using Proximal Policy Optimization (PPO) with step-wise rewards")
    print(f"Policy mode: {'Deterministic' if actor.deterministic else 'Stochastic'}")
    print(f"Performing {args.num_rollouts} rollouts per training step")
    print(f"PPO epochs: {args.ppo_epochs}, Clip epsilon: {args.clip_epsilon}, Gamma: {args.gamma}")
    
    # Move data to device
    initial_state = initial_state.to(device)
    actual_states = actual_states.to(device)
    
    # Determine checkpoint saving strategy: save every 10 epochs OR 9 evenly spaced, whichever is fewer
    # Option 1: Save every 10 epochs
    save_every_10_epochs = list(range(10, args.num_epochs + 1, 10))
    
    # Option 2: 9 evenly spaced epochs (including first and last)
    if args.num_epochs >= 9:
        evenly_spaced_epochs = np.linspace(1, args.num_epochs, 9, dtype=int).tolist()
        # Remove duplicates and ensure sorted
        evenly_spaced_epochs = sorted(list(set(evenly_spaced_epochs)))
    else:
        # If fewer than 9 epochs, save all epochs
        evenly_spaced_epochs = list(range(1, args.num_epochs + 1))
    
    # Choose the option with fewer checkpoints
    if len(save_every_10_epochs) >= len(evenly_spaced_epochs):
        epochs_to_save = set(save_every_10_epochs)
        save_strategy = "every_10_epochs"
    else:
        epochs_to_save = set(evenly_spaced_epochs)
        save_strategy = "evenly_spaced"
    
    # Always save final epoch
    epochs_to_save.add(args.num_epochs)
    
    print(f"\nCheckpoint saving strategy: {save_strategy}")
    print(f"Will save checkpoints at epochs: {sorted(epochs_to_save)}")
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train with PPO
        train_loss, rollout_trajectory = train_epoch_ppo(
            actor=actor,
            value_net=value_net,
            initial_state=initial_state,
            actual_states=actual_states,
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
        
        print(f"Train Loss: {train_loss:.6f}")
        
        # Track best model (minimum training loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            # Save best model checkpoint
            best_checkpoint = {
                'epoch': best_epoch,
                'actor_state_dict': actor.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'train_loss': best_loss,
                'rollout_trajectories': rollout_trajectory,  # [num_trajectories, seq_len, 4] - normalized states
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }
            torch.save(best_checkpoint, output_dir / 'best_model.pt')
            print(f"New best model saved at epoch {best_epoch} with loss {best_loss:.6f}")
        
        # Save checkpoint with rollout trajectories (up to 10) if this epoch is in the save set
        if (epoch + 1) in epochs_to_save:
            checkpoint = {
                'epoch': epoch + 1,
                'actor_state_dict': actor.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'train_loss': train_loss,
                'rollout_trajectories': rollout_trajectory,  # [num_trajectories, seq_len, 4] - normalized states
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
            print(f"Saved checkpoint for epoch {epoch + 1} with {rollout_trajectory.shape[0]} rollout trajectories")
    
    # Save final rollout trajectories (generate 10 stochastic rollouts)
    # Use stochastic sampling to get diverse trajectories
    actor.eval()
    with torch.no_grad():
        # Generate 10 rollouts for final model
        initial_state_batch = initial_state.repeat(10, 1)  # [10, 4]
        actual_states_batch = actual_states.repeat(10, 1, 1)  # [10, seq_len, 4]
        
        # Use stochastic policy (sampling) for final rollouts to get diverse trajectories
        final_rollout_trajectories, _, _ = rollout_trajectory_state_feedback(
            actor=actor,
            initial_state=initial_state_batch,
            actual_trajectory=actual_states_batch,
            seq_len=actual_states_complete.shape[1],
            dt=args.dt
        )
        final_rollout_trajectories = final_rollout_trajectories.detach().cpu()  # [10, seq_len, 4]
    
    # Save final model with rollout trajectories
    final_checkpoint = {
        'epoch': args.num_epochs,
        'actor_state_dict': actor.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'value_optimizer_state_dict': value_optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'rollout_trajectories': final_rollout_trajectories,  # [10, seq_len, 4] - normalized states
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pt')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (PPO)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Proximal Policy Optimization (PPO) Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()
    
    print(f"\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")
    
    # Create a simple dataset wrapper for visualization (only contains the first trajectory)
    # The visualization function expects a dataset with state_mean and state_std as numpy arrays
    class SingleTrajectoryDataset:
        def __init__(self, states, state_mean, state_std):
            # states should be [seq_len, 4] tensor - convert to numpy if needed
            if isinstance(states, torch.Tensor):
                states = states.cpu().numpy()
            self.states = states.reshape(1, -1, 4) if len(states.shape) == 2 else states  # Ensure [1, seq_len, 4]
            # Ensure state_mean and state_std are numpy arrays
            if isinstance(state_mean, torch.Tensor):
                state_mean = state_mean.cpu().numpy()
            if isinstance(state_std, torch.Tensor):
                state_std = state_std.cpu().numpy()
            self.state_mean = state_mean
            self.state_std = state_std
        
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            # Return [seq_len, 4] as torch tensor (visualization converts it)
            return torch.from_numpy(self.states[0]).float()
    
    # Get normalized states as numpy for visualization (use complete trajectory)
    normalized_states_np = actual_states_complete[0].cpu().numpy()  # [seq_len, 4]
    viz_dataset = SingleTrajectoryDataset(
        normalized_states_np,
        train_dataset.state_mean,
        train_dataset.state_std
    )
    
    # Generate comparison plots for training trajectory using saved rollout trajectories
    # Pass downsampled states if downsampling was used
    actual_states_downsampled_np = None
    if args.downsample_ratio > 1:
        actual_states_downsampled_np = actual_states[0].cpu().numpy()  # [downsampled_seq_len, 4]
    
    generate_trajectory_comparison_plots(
        dataset=viz_dataset,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        algorithm_name="PPO",
        dt=args.dt,
        actual_states_downsampled=actual_states_downsampled_np
    )


if __name__ == '__main__':
    main()
