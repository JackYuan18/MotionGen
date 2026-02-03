"""
Reinforcement learning training script for trajectory to control neural network.

This script trains a neural network control policy using policy gradient methods
(REINFORCE) to minimize the difference between generated and actual trajectories.
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import gym

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
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
from rewards import compute_reward_downsampled_next
from utils import DT
from ppo_args import add_ppo_common_args, add_trajectory_data_args
# NOTE: Actor/Value now only depend on (current_state, current_time_index). No next-downsampled lookup needed.

def downsample_states(states: torch.Tensor, ratio: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample states by keeping every Nth state.
    
    Args:
        states: [batch_size, seq_len, state_dim] or [seq_len, state_dim] tensor
        ratio: Downsample ratio (keep every Nth state, starting from index 0)
    
    Returns:
        Tuple of:
        - Downsampled states with shape [batch_size, new_seq_len, state_dim] or [new_seq_len, state_dim]
        - Indices tensor with shape [new_seq_len] or [batch_size, new_seq_len] containing the original indices
    """
    if ratio <= 1:
        seq_len = states.shape[-2] if len(states.shape) == 3 else states.shape[0]
        if len(states.shape) == 2:
            indices = torch.arange(seq_len, device=states.device, dtype=torch.long)
        else:
            batch_size = states.shape[0]
            indices = torch.arange(seq_len, device=states.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return states, indices
    
    if len(states.shape) == 2:
        # [seq_len, state_dim]
        seq_len = states.shape[0]
        indices = torch.arange(0, seq_len-1, ratio, device=states.device, dtype=torch.long)
        return torch.cat([states[::ratio], states[-1]], dim=0), indices
    else:
        # [batch_size, seq_len, state_dim]
        seq_len = states.shape[1]
        indices = torch.arange(0, seq_len-1, ratio, device=states.device, dtype=torch.long)
        indices = torch.cat([indices, torch.tensor([seq_len-1], device=states.device, dtype=torch.long)])
        # Expand indices for batch dimension: [batch_size, new_seq_len]
        batch_size = states.shape[0]
        indices = indices.unsqueeze(0).expand(batch_size, -1)
        return torch.cat([states[:, ::ratio, :], states[:, -1, :].unsqueeze(1)], dim=1), indices

def rollout_lunar_lander(
    actor: nn.Module,
    initial_state: torch.Tensor,
    seq_len: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    env = gym.make("LunarLander-v2")
    device = initial_state.device   
    batch_size = initial_state.shape[0]
    state_dim = initial_state.shape[1]
    predicted_states = torch.zeros(batch_size, seq_len, state_dim, device=device)
    sampled_controls = torch.zeros(batch_size, seq_len, 1, device=device, dtype=torch.long)
    log_probs = torch.zeros(batch_size, seq_len, device=device)
    
    # Initialize current state from initial_state (first state of batch)
    # For batch processing, we'll use the first state and process sequentially
    current_state_np, _ = env.reset(seed=42)
    current_state = torch.tensor(current_state_np, device=device, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
    
    # Store initial state
    predicted_states[:, 0, :] = current_state.expand(batch_size, -1)
    # Expand current_state to match batch_size for actor.sample()
    current_state_batch = current_state.expand(batch_size, -1)  # [batch_size, state_dim]
        
    # Pre-allocate time indices tensor
    time_indices = torch.arange(seq_len - 1, device=device, dtype=torch.long)
    
    # Track actual trajectory length (may terminate early)
    actual_length = seq_len
    
    for t_idx, t in enumerate(range(seq_len - 1)):
        # Use pre-allocated time index
        current_time_idx = time_indices[t_idx:t_idx+1].expand(batch_size)
        
        
        # Sample action from actor (stochastic policy) - expects tensor input
        sampled_control, log_prob, _ = actor.sample(
            current_state_batch, 
            current_time_index=current_time_idx
        )
        
        # sampled_control is [batch_size], take first action for environment
        action_np = sampled_control[0].item()  # Convert to Python int
        
        # Store control and log prob for all batch elements (same action for all)
        sampled_controls[:, t, 0] = sampled_control
        log_probs[:, t] = log_prob
        
        # Apply dynamics - step environment
        next_state_np, _, terminated, truncated, info = env.step(action_np)
        
        # Convert next state to tensor
        next_state = torch.tensor(next_state_np, device=device, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
        
        # Store next state and update current state
        predicted_states[:, t + 1, :] = next_state.expand(batch_size, -1)
        current_state = next_state  # Update reference
        
        # Reset if episode terminated or truncated
        if terminated or truncated:
            actual_length = t + 2  # t+2 because we've stored state at t+1, and we started at t=0
            break
    
    # Return only the actual generated states (truncate if episode ended early)
    if actual_length < seq_len:
        predicted_states = predicted_states[:, :actual_length, :]
        sampled_controls = sampled_controls[:, :actual_length, :]
        log_probs = log_probs[:, :actual_length]

    return predicted_states, sampled_controls, log_probs


def train_epoch_ppo(
    actor: nn.Module,
    value_net: nn.Module,
    initial_state: torch.Tensor,
    actual_states: torch.Tensor,
    downsampled_indices: torch.Tensor,
    actor_optimizer: optim.Optimizer,
    value_optimizer: optim.Optimizer,
    device: torch.device,
    num_rollouts: int = 10,
    dt: float = DT,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,  # Increased to encourage more exploration
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
        initial_state: [1, 8] initial state (single trajectory) for LunarLander
        actual_states: [1, seq_len, 8] actual trajectory states (single trajectory) for LunarLander
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
    initial_state_batch = initial_state.repeat(num_rollouts, 1)  # [num_rollouts, 8] for LunarLander
    
    # Expand actual_states to match batch size for rollout
    actual_states_batch = actual_states.repeat(num_rollouts, 1, 1)  # [num_rollouts, seq_len, 8] for LunarLander
    
    # Rollout multiple trajectories using state-feedback control policy
    with torch.no_grad():
        predicted_states, sampled_controls, old_log_probs = rollout_lunar_lander(
            actor=actor,
            initial_state=initial_state_batch,
            seq_len=seq_len,
            device=device
        )

    
    # Get the actual sequence length from predicted_states (may differ from actual_states due to dt)
    predicted_seq_len = predicted_states.shape[1]

    
    with torch.no_grad():
        # Compute step-wise rewards for each rollout
        # Note: compute_reward can handle different sequence lengths between predicted and actual
        step_rewards = compute_reward_downsampled_next(predicted_states, actual_states_batch, downsampled_indices,  actions=sampled_controls)  # [num_rollouts, predicted_seq_len]
        # step_rewards = torch.zeros(num_rollouts, predicted_seq_len, device=device)

        
        # Find the rollout with the best sum of step rewards
        total_rewards_per_rollout = step_rewards.sum(dim=1)  # [num_rollouts] - sum of rewards for each rollout
        best_rollout_idx = total_rewards_per_rollout.argmax().item()  # Index of rollout with highest total reward
        best_rollout_trajectory = predicted_states[best_rollout_idx].detach().cpu()  # [predicted_seq_len, 8] - best rollout trajectory (moved to CPU)
        best_rollout_reward_sum = total_rewards_per_rollout[best_rollout_idx].item()
        best_rollout_controls = sampled_controls[best_rollout_idx].detach().cpu()  # [predicted_seq_len, 2] - best rollout controls (moved to CPU)
        
        # Compute values for all states in trajectories - batched for GPU efficiency
        # Flatten for batch processing: [num_rollouts * predicted_seq_len, state_dim]
        state_dim = predicted_states.shape[2]
        states_flat = predicted_states.reshape(-1, state_dim)  # [num_rollouts * predicted_seq_len, state_dim]
        # Prepare time indices for value network evaluation
        time_indices_flat = torch.arange(predicted_seq_len, device=device, dtype=torch.long).repeat(num_rollouts)  # [num_rollouts * predicted_seq_len]
        
        # Batch value network evaluation with adaptive batch size to avoid OOM
        total_states = num_rollouts * predicted_seq_len
        value_batch_size = min(512, total_states)  # Larger batch size for value network
        old_values_list = []
      
        for batch_start in range(0, total_states, value_batch_size):
            batch_end = min(batch_start + value_batch_size, total_states)
            batch_slice = slice(batch_start, batch_end)
            batch_values = value_net(
                states_flat[batch_slice],
                current_time_index=time_indices_flat[batch_slice]
            ).squeeze(-1)  # [batch_size]
            old_values_list.append(batch_values)
        
        old_values = torch.cat(old_values_list, dim=0).reshape(num_rollouts, predicted_seq_len)  # [num_rollouts, predicted_seq_len]
        
        # Compute returns using vectorized approach: Returns[t] = reward[t] + gamma * Returns[t+1]
        # Work backwards from the end, but vectorize across rollouts
        returns = torch.zeros_like(step_rewards)  # [num_rollouts, predicted_seq_len]
        returns[:, -1] = step_rewards[:, -1]  # Last step return is just its reward
        
        # Vectorized backward pass: for each time step, compute in parallel across rollouts
        # We still need to iterate backwards, but each iteration processes all rollouts at once
        for t in range(predicted_seq_len - 2, -1, -1):
            returns[:, t] = step_rewards[:, t] + gamma * returns[:, t + 1]
        
        # Compute advantages (returns - values)
        advantages = returns - old_values  # [num_rollouts, predicted_seq_len]
        # Normalize advantages to stabilize training and ensure consistent gradient signals
        # This is critical for PPO to work effectively
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-6:  # Only normalize if std is significant
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        # If std is too small, advantages are likely uniform - keep as is
        
    
    # Prepare data for PPO updates
    # Flatten all trajectories into a single batch
    total_samples = num_rollouts * predicted_seq_len
    
    # Repeat advantages and returns for each time step
    advantages_flat = advantages.reshape(-1)  # [num_rollouts * predicted_seq_len]
    returns_flat = returns.reshape(-1)  # [num_rollouts * predicted_seq_len]
    old_log_probs_flat = old_log_probs.reshape(-1)  # [num_rollouts * predicted_seq_len]
    
    # Store rollout trajectories for visualization (move to CPU asynchronously)
    num_trajectories_to_save = min(10, num_rollouts)
    rollout_trajectories = predicted_states[:num_trajectories_to_save].detach()  # Keep on GPU until needed
    
    # PPO update epochs
    total_actor_loss = 0.0
    total_value_loss = 0.0
    
    for ppo_epoch in range(ppo_epochs):
        # Shuffle data
        indices = torch.randperm(total_samples, device=device)
        
        # Process in mini-batches - use larger batch size for better GPU utilization
        batch_size = min(256, total_samples)  # Increased from 64 to 256 for better GPU utilization
        
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch data
            batch_states = states_flat[batch_indices]  # [batch_size, 8] for LunarLander
            batch_actions = sampled_controls.reshape(-1, 1)[batch_indices]  # Correct - discrete actions are 1D
            batch_advantages = advantages_flat[batch_indices]  # [batch_size]
            batch_returns = returns_flat[batch_indices]  # [batch_size]
            batch_old_log_probs = old_log_probs_flat[batch_indices]  # [batch_size]
            
            # Get current policy log probs and values
            # Need to recompute log probs for the same actions
            # Get time indices for batch states (from the flattened indices)
            # batch_indices correspond to positions in the flattened [num_rollouts * predicted_seq_len] array
            # Convert to (rollout_idx, time_idx) pairs
            time_indices = batch_indices % predicted_seq_len  # Which time step in that rollout
            batch_time_indices = time_indices.long()  # [batch_size]

            # For discrete actions (LunarLander), use log_prob method
            batch_actions_squeezed = batch_actions.squeeze(-1)  # [batch_size] - remove last dimension
            new_log_probs = actor.log_prob(batch_states, batch_actions_squeezed, current_time_index=batch_time_indices)  # [batch_size]
            
            # Compute value estimates
            new_values = value_net(batch_states, current_time_index=batch_time_indices).squeeze(-1)  # [batch_size]
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - batch_old_log_probs)  # [batch_size]
            
            # Clipped surrogate objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss: use MSE loss to learn the actual return scale
            # DO NOT normalize by returns variance - this prevents learning the true reward scale
            # The value function should predict actual returns (around -22.6), not normalized returns
            # Normalizing by variance makes the loss scale-invariant, which can cause the value function
            # to converge to a suboptimal solution where rewards don't improve
            # Weight the MSE loss by the (shifted) returns so higher returns have higher weight
            # Normalize weights to sum to 1 for stability
            mean_return = batch_returns.mean()
            std_return = batch_returns.std()
            weights = batch_returns - mean_return + 1e-4  # shift so weights are >= 0
            weights = weights / std_return  # normalize
            weights = torch.softmax(weights, dim=0)  # normalize
            value_loss = torch.nn.functional.mse_loss(new_values, batch_returns, reduction='none')
            value_loss = (value_loss * weights).sum()
            
            # Entropy bonus (encourage exploration) - for discrete actions
            # Get logits to compute entropy
            logits = actor.forward(batch_states, current_time_index=batch_time_indices)  # [batch_size, num_actions]
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()  # [batch_size] -> scalar
            
            # Minimum entropy constraint: ensure policy maintains some exploration
            # For discrete actions, minimum entropy is based on number of actions
            num_actions = logits.shape[1]  # Number of discrete actions
            min_entropy = torch.tensor(0.5 * np.log(num_actions), device=device, dtype=torch.float32)  # Minimum entropy (roughly uniform distribution)
            entropy_penalty = torch.clamp(min_entropy - entropy, min=0.0) * 0.1  # Penalty if entropy too low
            
            entropy_bonus = -entropy_coef * entropy - entropy_penalty
            
            # Total actor loss
            total_actor_loss_batch = actor_loss + entropy_bonus
            
            # Compute gradients (update after each batch for stability, but reduce optimizer overhead)
            total_actor_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            actor_optimizer.step()
            actor_optimizer.zero_grad()
            
            # Update value network
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
            value_optimizer.step()
            value_optimizer.zero_grad()
            
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
    
    return avg_loss, normalized_actor_loss, normalized_value_loss, rollout_trajectories,  best_rollout_trajectory, best_rollout_reward_sum, best_rollout_controls


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
        default='Lunar_Training_Results/rl_checkpoints',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--downsample_ratio',
        type=int,
        default=30,
        help='Downsample ratio for actual states during training (default: 1, no downsampling). '
             'If > 1, every Nth state is kept. Complete trajectory is used for evaluation and visualization.'
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
    
 

    # Load the first trajectory from LunarLander.json
    lunar_json_path = script_dir / "LunarLander.json"
    with open(lunar_json_path, "r") as f:
        lunar_data = json.load(f)
    # Get the first key (should be "0") and extract the trajectory dict
    trajectory = lunar_data["0"]["Trajectory"]
    # all_trajectories should be a list of trajectory dicts as expected by prepare_training_data
    
    
    # Prepare training data (states only, no controls)
    print("Preparing training data...")
    states = torch.tensor(trajectory['obs'], device=device)
    
    # Add batch dimension if states is 2D: [seq_len, state_dim] -> [1, seq_len, state_dim]
    if states.dim() == 2:
        states = states.unsqueeze(0)  # Add batch dimension
    
    print(f"States shape: {states.shape}")
    print(f"Initial States: {states[0,0,:]}")
    
    # For reinforcement learning, use all loaded trajectories
    # If multiple trajectories, we'll train on each one sequentially
    print(f"Using {states.shape[0]} trajectory(ies) for RL training...")
    
    # Compute state statistics for visualization (states are NOT normalized)
    # These are used by visualization code but don't affect training
    state_mean = states.mean(dim=(0, 1), keepdim=False)  # [state_dim]
    state_std = states.std(dim=(0, 1), keepdim=False)  # [state_dim]
    # Avoid division by zero
    state_std = torch.where(state_std < 1e-6, torch.ones_like(state_std), state_std)
    
    initial_state = states[:, 0, :]  # [1, state_dim]
    
    # Keep complete states for visualization
    actual_states_complete = states  # [1, seq_len, 8] for LunarLander
    
    # Downsample actual_states for training if specified
    # The reward function can handle different sequence lengths without time alignment
    if args.downsample_ratio > 1:
        actual_states, downsampled_indices = downsample_states(actual_states_complete, args.downsample_ratio)
        print(f"Downsampling actual states: {actual_states_complete.shape[1]} -> {actual_states.shape[1]} (ratio={args.downsample_ratio})")
    else:
        actual_states = actual_states_complete
        seq_len = actual_states_complete.shape[1]
        batch_size = actual_states_complete.shape[0]
        downsampled_indices = torch.arange(seq_len, device=actual_states_complete.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Initial State: {initial_state[0,:]}")
    if len(states) > 1:
        print(f"Note: {len(states)} trajectories loaded. Training on first trajectory only.")
        print("To train on multiple trajectories, modify the training loop to iterate over them.")
    
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Actual states shape (for training): {actual_states.shape}")
    print(f"Actual states shape (complete): {actual_states_complete.shape}")
    
    # Create actor network (deterministic by default for PPO)
    print("Creating ActorNetwork...")
    actor = ActorNetwork(
        state_dim=states.shape[-1],
        num_actions=4,
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
        state_dim=states.shape[-1],
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
    best_rollout = float('-inf')
    
    print("\nStarting reinforcement learning training...")
    print("Using Proximal Policy Optimization (PPO) with step-wise rewards")
    print(f"Policy mode: {'Deterministic' if actor.deterministic else 'Stochastic'}")
    print(f"Performing {args.num_rollouts} rollouts per training step")
    print(f"PPO epochs: {args.ppo_epochs}, Clip epsilon: {args.clip_epsilon}, Gamma: {args.gamma}")
    
    # Move data to device
    initial_state = initial_state.to(device)
    actual_states = actual_states.to(device)
    downsampled_indices = downsampled_indices.to(device)
    
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
        train_loss, actor_loss, value_loss, rollout_trajectory, best_rollout_trajectory, best_rollout_reward_sum, best_rollout_controls = train_epoch_ppo(
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
        
        print(f"Train Loss: {train_loss:.6f}, Actor Loss: {actor_loss:.6f}, Value Loss: {value_loss:.6f}, "
              f"Best Rollout Reward Sum: {best_rollout_reward_sum:.6f}")
        
        # Track best model (minimum training loss)
        if best_rollout_reward_sum > best_rollout:
            best_rollout = best_rollout_reward_sum
            
            best_epoch = epoch + 1
            # Save best model checkpoint
            best_checkpoint = {
                'epoch': best_epoch,
                'actor_state_dict': actor.state_dict(),
                'downsampled_indices': downsampled_indices,
                'value_net_state_dict': value_net.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'train_loss': best_loss,
                'rollout_trajectories': rollout_trajectory,  # [num_trajectories, seq_len, 8] - states (not normalized)
                'best_rollout_trajectory': best_rollout_trajectory,  # [seq_len, 8] - states (not normalized)
                'best_rollout_reward_sum': best_rollout_reward_sum,
                'best_rollout_controls': best_rollout_controls,
                'state_dim': states.shape[-1],  # Save state dimension for model loading
                'num_actions': 4,  # LunarLander has 4 discrete actions
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
                'downsampled_indices': downsampled_indices,
                'value_net_state_dict': value_net.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
                'train_loss': train_loss,
                'rollout_trajectories': rollout_trajectory,  # [num_trajectories, seq_len, 8] - states (not normalized)
                'best_rollout_trajectory': best_rollout_trajectory,  # [seq_len, 8] - states (not normalized)
                'best_rollout_reward_sum': best_rollout_reward_sum,
                'best_rollout_controls': best_rollout_controls,
                'state_dim': states.shape[-1],  # Save state dimension for model loading
                'num_actions': 4,  # LunarLander has 4 discrete actions
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
        initial_state_batch = initial_state.repeat(10, 1)  # [10, 8] for LunarLander
        actual_states_batch = actual_states_complete.repeat(10, 1, 1)  # [10, seq_len, 8]
        
        # Expand downsampled_indices for batch
        if downsampled_indices.shape[0] == 1:
            downsampled_indices_batch = downsampled_indices.expand(10, -1)
        else:
            downsampled_indices_batch = downsampled_indices[:10]  # Take first 10 if multiple
        
        # Use rollout_lunar_lander for LunarLander environment
        final_predicted_states, final_sampled_controls, _ = rollout_lunar_lander(
            actor=actor,
            initial_state=initial_state_batch,
            seq_len=actual_states_complete.shape[1],
            device=device
        )
        final_rollout_trajectories = final_predicted_states.detach().cpu()  # [10, seq_len, 8]
        
        # Compute step rewards for final rollouts to find the best one
        final_step_rewards = compute_reward_downsampled_next(
            final_predicted_states, actual_states_batch, downsampled_indices_batch, actions=final_sampled_controls
        )  # [10, predicted_seq_len]
        
        # Find best rollout
        final_total_rewards = final_step_rewards.sum(dim=1)  # [10]
        final_best_rollout_idx = final_total_rewards.argmax().item()
        final_best_rollout_trajectory = final_rollout_trajectories[final_best_rollout_idx]  # [seq_len, 8] (already on CPU)
        final_best_rollout_reward_sum = final_total_rewards[final_best_rollout_idx].item()
    
    # Save final model with rollout trajectories
    final_checkpoint = {
        'epoch': args.num_epochs,
        'actor_state_dict': actor.state_dict(),
        'downsampled_indices': downsampled_indices,
        'value_net_state_dict': value_net.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'value_optimizer_state_dict': value_optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'rollout_trajectories': final_rollout_trajectories,  # [10, seq_len, 8] - states (not normalized)
        'best_rollout_trajectory': final_best_rollout_trajectory,  # [seq_len, 8] - states (not normalized)
        'best_rollout_reward_sum': final_best_rollout_reward_sum,
        'state_dim': states.shape[-1],  # Save state dimension for model loading
        'num_actions': 4,  # LunarLander has 4 discrete actions
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
            # states should be [seq_len, 8] tensor for LunarLander - convert to numpy if needed
            if isinstance(states, torch.Tensor):
                states = states.cpu().numpy()
            self.states = states.reshape(1, -1, states.shape[-1]) if len(states.shape) == 2 else states  # Ensure [1, seq_len, state_dim]
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
            # Return [seq_len, state_dim] as torch tensor (visualization converts it)
            return torch.from_numpy(self.states[0]).float()
    
    # Get states as numpy for visualization (use complete trajectory)
    # Note: states are NOT normalized, but visualization code expects state_mean and state_std
    states_np = actual_states_complete[0].cpu().numpy()  # [seq_len, 8] for LunarLander
    viz_dataset = SingleTrajectoryDataset(
        states_np,
        state_mean.cpu().numpy(),
        state_std.cpu().numpy()
    )
    
    # Generate comparison plots for training trajectory using saved rollout trajectories
    # Pass downsampled states if downsampling was used
    actual_states_downsampled_np = None
    if args.downsample_ratio > 1:
        actual_states_downsampled_np = actual_states[0].cpu().numpy()  # [downsampled_seq_len, 8] for LunarLander
    
    generate_trajectory_comparison_plots(
        dataset=viz_dataset,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        algorithm_name="PPO",
        environment_name="LunarLander",
        dt=args.dt,
        actual_states_downsampled=actual_states_downsampled_np
    )


if __name__ == '__main__':
    main()