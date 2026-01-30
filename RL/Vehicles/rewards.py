"""
Reward computation functions for reinforcement learning.

This module provides reward functions for training control policies.
"""

import torch


def compute_reward(
    predicted_states: torch.Tensor,
    actual_states: torch.Tensor,
    actions: torch.Tensor = None,
    action_smoothness_weight: float = 0.01,
    action_norm_weight: float = 0.01,
    progression_weight: float = 0.1,
    coverage_weight: float = 5,
    end_state_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute step-wise rewards based on distances from predicted states to actual trajectory.
    
    For each predicted state, the reward consists of:
    1. Coverage error - penalizes unmatched actual states (encourages covering all actual states)
    2. Progression error - penalizes if predicted trajectory goes backwards along actual trajectory
    3. End state error - penalizes mismatch between predicted and actual end states
    4. (Optional) Action smoothness penalty - difference between consecutive actions
    5. (Optional) Action norm penalty - magnitude of actions
    
    Returns step-wise rewards (reward at each predicted state).
    (Negative because we want to maximize reward, which means minimize distances/errors).
    
    Args:
        predicted_states: [batch_size, seq_len_pred, 4] predicted trajectory states
        actual_states: [batch_size, seq_len_actual, 4] actual trajectory states
                      (can have different sequence length than predicted_states)
        actions: [batch_size, seq_len_pred, 2] actions/controls applied at each time step
                 If None, action smoothness penalty is not included
        action_smoothness_weight: Weight for action smoothness penalty (default: 0.01)
        action_norm_weight: Weight for action norm penalty (default: 0.01)
        progression_weight: Weight for progression error penalty (default: 0.1)
        coverage_weight: Weight for coverage error penalty (default: 0.1)
        end_state_weight: Weight for end state matching penalty (default: 1.0)
    Returns:
        Step-wise rewards [batch_size, seq_len_pred] (higher is better)
    """
    batch_size, seq_len_pred, state_dim = predicted_states.shape
    _, seq_len_actual, _ = actual_states.shape
    
    # Part 1: Coverage error - penalize unmatched actual states
    # Expand dimensions for broadcasting to compute all pairwise distances
    pred_expanded = predicted_states.unsqueeze(2)  # [batch_size, seq_len_pred, 1, 4]
    actual_expanded = actual_states.unsqueeze(1)    # [batch_size, 1, seq_len_actual, 4]
    
    # Compute state differences for all pairs
    state_diff = pred_expanded - actual_expanded  # [batch_size, seq_len_pred, seq_len_actual, 4]
    
    # Wrap heading difference to (-pi, pi]
    heading_diff = state_diff[:, :, :, 2]
    heading_diff_wrapped = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))
    state_diff[:, :, :, 2] = heading_diff_wrapped
    
    # Compute Euclidean distances
    state_dists = torch.norm(state_diff, dim=3)  # [batch_size, seq_len_pred, seq_len_actual]
    
    # Coverage error - penalize unmatched actual states
    # For each actual state, find the closest predicted state
    dists_from_actual_to_closest_pred = torch.min(state_dists, dim=1)[0]  # [batch_size, seq_len_actual]
    
    # Coverage error: average distance from actual states to their closest predicted states
    # This encourages the predicted trajectory to cover all actual states
    coverage_error = dists_from_actual_to_closest_pred.mean(dim=1)  # [batch_size]
    # Expand to match predicted sequence length for per-step reward
    coverage_error_expanded = coverage_error.unsqueeze(1).expand(-1, seq_len_pred)  # [batch_size, seq_len_pred]
    
    # Part 2: Progression error - measure forward progress along actual trajectory
    # Get indices of closest actual states
    closest_indices = torch.argmin(state_dists, dim=2)  # [batch_size, seq_len_pred]
    
    # Gather closest actual states for each predicted state
    batch_idx = torch.arange(batch_size, device=predicted_states.device).unsqueeze(1).expand(-1, seq_len_pred)
    closest_actual_states = actual_states[batch_idx, closest_indices, :]  # [batch_size, seq_len_pred, 4]
    
    # Get previous predicted states (shifted by one time step)
    prev_predicted_states = torch.zeros_like(predicted_states)
    prev_predicted_states[:, 1:, :] = predicted_states[:, :-1, :]
    
    # Compute distance from previous predicted state to current closest actual state
    state_diff_progression = closest_actual_states - prev_predicted_states  # [batch_size, seq_len_pred, 4]
    
    # Wrap heading difference
    heading_diff_prog = state_diff_progression[:, :, 2]
    heading_diff_prog_wrapped = torch.atan2(torch.sin(heading_diff_prog), torch.cos(heading_diff_prog))
    state_diff_progression[:, :, 2] = heading_diff_prog_wrapped
    
    # Compute progression distance
    progression_distance = torch.norm(state_diff_progression, dim=2)  # [batch_size, seq_len_pred]
    
    # Determine direction of progress using index differences
    closest_indices_float = closest_indices.float()
    prev_indices = torch.zeros_like(closest_indices_float)
    prev_indices[:, 1:] = closest_indices_float[:, :-1]
    index_diff = closest_indices_float - prev_indices  # [batch_size, seq_len_pred]
    
    # Compute progression error: reward forward progress, penalize backward/no progress
    progression_error = torch.zeros_like(progression_distance)
    forward_mask = index_diff > 0
    backward_mask = index_diff < 0
    no_progress_mask = index_diff == 0
    
    progression_error[forward_mask] = 1.0 / (progression_distance[forward_mask] + 1e-6)
    progression_error[backward_mask] = progression_distance[backward_mask]
    progression_error[no_progress_mask] = 0.1
    progression_error[:, 0] = 0.0  # No penalty for first step
    
    # Part 2b: End state matching error - penalize mismatch between predicted and actual end states
    # Get last states
    predicted_end_states = predicted_states[:, -1, :]  # [batch_size, 4]
    actual_end_states = actual_states[:, -1, :]  # [batch_size, 4]
    
    # Compute state difference
    end_state_diff = predicted_end_states - actual_end_states  # [batch_size, 4]
    
    # Wrap heading difference to (-pi, pi]
    end_heading_diff = end_state_diff[:, 2]
    end_heading_diff_wrapped = torch.atan2(torch.sin(end_heading_diff), torch.cos(end_heading_diff))
    end_state_diff[:, 2] = end_heading_diff_wrapped
    
    # Compute Euclidean distance between end states
    end_state_error = torch.norm(end_state_diff, dim=1)  # [batch_size]
    
    # Expand to match predicted sequence length for per-step reward
    # Apply end state error to all steps to encourage overall trajectory matching
    end_state_error_expanded = end_state_error.unsqueeze(1).expand(-1, seq_len_pred)  # [batch_size, seq_len_pred]
    
    # Part 3: Action penalties
    action_smoothness_penalty = torch.zeros(batch_size, seq_len_pred, device=predicted_states.device)
    action_norm = torch.zeros(batch_size, seq_len_pred, device=predicted_states.device)
    
    if actions is not None:
        # Action smoothness: difference between consecutive actions
        prev_actions = torch.zeros_like(actions)
        prev_actions[:, 1:, :] = actions[:, :-1, :]
        action_diffs = actions - prev_actions
        action_smoothness_penalty = torch.norm(action_diffs, dim=2)  # [batch_size, seq_len_pred]
        
        # Action norm: magnitude of actions (reduced penalty for first action)
        action_norm = torch.norm(actions, dim=2)  # [batch_size, seq_len_pred]
        action_norm[:, 0] = action_norm[:, 0] * 0.1
    
    # Combine all components (negative because we maximize reward = minimize errors)
    state_rewards = -(
        coverage_weight * coverage_error_expanded
        + progression_weight * progression_error
        + end_state_weight * end_state_error_expanded
        + action_smoothness_weight * action_smoothness_penalty 
        + action_norm_weight * action_norm
    )  # [batch_size, seq_len_pred]
    
    return state_rewards


def compute_reward_downsampled_next(
    predicted_states: torch.Tensor,
    downsampled_states: torch.Tensor,
    downsampled_indices: torch.Tensor,
    actions: torch.Tensor = None,
    action_norm_weight: float = 0.001,
    distance_weight: float = 10.0
) -> torch.Tensor:
    """
    Compute step-wise rewards based on distance toward the next downsampled state.
    
    For each predicted state at time index t:
    1. Find the downsampled state with the smallest time index larger than t
    2. Compute the distance toward that next downsampled state
    3. Reward is negative distance (closer to next target = higher reward)
    
    This encourages the policy to progress forward through the downsampled waypoints.
    
    Args:
        predicted_states: [batch_size, seq_len_pred, 4] predicted trajectory states
                         Each state at position i has time index i (0, 1, 2, ...)
        downsampled_states: [batch_size, seq_len_downsampled, 4] downsampled actual states
        downsampled_indices: [batch_size, seq_len_downsampled] original time indices of downsampled states
        actions: [batch_size, seq_len_pred, 2] actions/controls applied at each time step
                 If None, action penalties are not included
        action_smoothness_weight: Weight for action smoothness penalty (default: 0.01)
        action_norm_weight: Weight for action norm penalty (default: 0.01)
        distance_weight: Weight for distance to next downsampled state (default: 10.0)
    
    Returns:
        Step-wise rewards [batch_size, seq_len_pred] (higher is better)
    """
    batch_size, seq_len_pred, state_dim = predicted_states.shape
    _, seq_len_downsampled, _ = downsampled_states.shape
    
    # Expand downsampled_indices and downsampled_states to match batch_size if needed
    if downsampled_indices.shape[0] == 1 and batch_size > 1:
        downsampled_indices = downsampled_indices.expand(batch_size, -1)
    if downsampled_states.shape[0] == 1 and batch_size > 1:
        downsampled_states = downsampled_states.expand(batch_size, -1, -1)
    
    # Vectorized computation: For each predicted state at time index t, find the next downsampled state
    # Create time indices for all predicted states: [batch_size, seq_len_pred]
    time_indices = torch.arange(seq_len_pred, device=predicted_states.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len_pred]
    
    # Expand downsampled_indices for broadcasting: [batch_size, seq_len_pred, seq_len_downsampled]
    # For each predicted time t, compare with all downsampled time indices
    downsampled_indices_expanded = downsampled_indices.unsqueeze(1)  # [batch_size, 1, seq_len_downsampled]
    time_indices_expanded = time_indices.unsqueeze(2)  # [batch_size, seq_len_pred, 1]
    
    # Find next downsampled states: mask where downsampled_indices > t
    next_mask = downsampled_indices_expanded > time_indices_expanded  # [batch_size, seq_len_pred, seq_len_downsampled]
    
    # Get next time indices (smallest time index > t for each (b, t))
    # Set invalid (no next state) to a very large value (use max value of dtype instead of inf)
    # Handle both integer and float types
    if downsampled_indices_expanded.dtype.is_floating_point:
        max_value = torch.finfo(downsampled_indices_expanded.dtype).max
    else:
        max_value = torch.iinfo(downsampled_indices_expanded.dtype).max
    next_time_indices = torch.where(next_mask, downsampled_indices_expanded, torch.full_like(downsampled_indices_expanded, max_value))
    next_time_idx_values, next_time_idx_positions = next_time_indices.min(dim=2)  # [batch_size, seq_len_pred]
    
    # Check if there's a valid next state (not max_value)
    has_next = next_time_idx_values < max_value
    
    # Get next target states using advanced indexing
    # For positions where has_next is True, use next_time_idx_positions; otherwise use -1 (last state)
    target_indices = torch.where(has_next, next_time_idx_positions, torch.full_like(next_time_idx_positions, seq_len_downsampled - 1, dtype=torch.long))
    
    # Use batch indexing to get target states: [batch_size, seq_len_pred, 4]
    batch_indices = torch.arange(batch_size, device=predicted_states.device).unsqueeze(1).expand(-1, seq_len_pred)
    next_target_states = downsampled_states[batch_indices, target_indices]  # [batch_size, seq_len_pred, 4]
    
    # Compute state differences: [batch_size, seq_len_pred, 4]
    state_diff = predicted_states - next_target_states
    
    # Wrap heading difference (index 2) for all states at once
    heading_diff = state_diff[:, :, 2]
    state_diff[:, :, 2] = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff))*2
    
    # Compute distances: [batch_size, seq_len_pred]
    next_target_distances = torch.norm(state_diff, dim=2)
    
    # Part 2: Action penalties (optional)
    action_norm = torch.zeros(batch_size, seq_len_pred, device=predicted_states.device)
    
    if actions is not None:
        # Action smoothness: difference between consecutive actions
        prev_actions = torch.zeros_like(actions)
        prev_actions[:, 1:, :] = actions[:, :-1, :]
        
        # Action norm: magnitude of actions (reduced penalty for first action)
        action_norm = torch.norm(actions, dim=2)  # [batch_size, seq_len_pred]
        action_norm[:, 0] = action_norm[:, 0] * 0.1
    
    # Combine all components (negative distance because we maximize reward = minimize distance)
    state_rewards = -(
        distance_weight * next_target_distances
        + action_norm_weight * action_norm
    )  # [batch_size, seq_len_pred]
    
    return state_rewards