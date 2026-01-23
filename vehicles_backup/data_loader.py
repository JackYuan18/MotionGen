"""
Data loading utilities for trajectory to control learning.

This module provides functions to load trajectory data and prepare it for training.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    """
    Dataset class for vehicle trajectories (states only).
    
    Each sample contains:
        - Input: Sequence of vehicle states [x, y, heading, velocity]
    """
    
    def __init__(
        self,
        states: np.ndarray,
        normalize: bool = True,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            states: Array of state sequences [N, seq_len, state_dim]
            normalize: Whether to normalize the data
            state_mean: Mean for state normalization (computed if None)
            state_std: Std for state normalization (computed if None)
        """
        self.states = states
        self.normalize = normalize
        
        # Compute normalization statistics if not provided
        if normalize:
            if state_mean is None or state_std is None:
                # Compute over all samples and time steps
                state_flat = states.reshape(-1, states.shape[-1])
                self.state_mean = np.mean(state_flat, axis=0)
                per_dim_std = np.std(state_flat, axis=0) + 1e-8  # Add small epsilon
                
                # Use a single scale factor to preserve relative scales between dimensions
                # Use the maximum std across all dimensions (excluding heading) as the global scale
                # This preserves the relative magnitudes between x, y, velocity, etc.
                if state_flat.shape[1] > 2:  # Ensure heading dimension exists
                    # Exclude heading (index 2) from scale computation
                    scale_dims = np.concatenate([per_dim_std[:2], per_dim_std[3:]]) if state_flat.shape[1] > 3 else per_dim_std[:2]
                    global_scale = np.max(scale_dims) if len(scale_dims) > 0 else 1.0
                    
                    # Apply same scale to all dimensions except heading
                    self.state_std = np.full_like(per_dim_std, global_scale)
                    self.state_std[2] = 1.0  # Heading std = 1 (no scaling)
                    
                    # Heading mean = 0 (no shift)
                    self.state_mean[2] = 0.0
                else:
                    # If no heading dimension, use max std as global scale
                    global_scale = np.max(per_dim_std)
                    self.state_std = np.full_like(per_dim_std, global_scale)
            else:
                self.state_mean = state_mean.copy() if isinstance(state_mean, np.ndarray) else state_mean
                self.state_std = state_std.copy() if isinstance(state_std, np.ndarray) else state_std
                
                # Ensure heading (index 2) is not normalized even if provided
                if isinstance(self.state_mean, np.ndarray) and self.state_mean.shape[0] > 2:
                    self.state_mean[2] = 0.0
                if isinstance(self.state_std, np.ndarray) and self.state_std.shape[0] > 2:
                    self.state_std[2] = 1.0
        else:
            self.state_mean = None
            self.state_std = None
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            States as torch tensor
        """
        states = self.states[idx].astype(np.float32)
        
        if self.normalize:
            states = (states - self.state_mean) / self.state_std
        
        return torch.from_numpy(states)
    
    def get_normalization_stats(self) -> Dict:
        """Get normalization statistics."""
        return {
            'state_mean': self.state_mean,
            'state_std': self.state_std
        }


def extract_states_from_trajectory(
    trajectory: Dict,
    state_dim: int = 4
) -> np.ndarray:
    """
    Extract state sequence from trajectory data.
    
    For vehicles: state = [x, y, heading, velocity]
    
    Args:
        trajectory: Trajectory dictionary with positions, velocities, headings
        state_dim: Dimension of state vector (default: 4)
    
    Returns:
        States array [seq_len, state_dim]
    """
    positions = np.array(trajectory.get('positions', []))
    velocities = np.array(trajectory.get('velocities', []))
    headings = np.array(trajectory.get('headings', []))
    
    if len(positions) == 0:
        return np.array([])
    
    num_points = len(positions)
    
    # For vehicle model: [x, y, θ, v]
    if len(headings) == num_points and len(velocities) == num_points:
        # Compute velocity magnitude
        v = np.linalg.norm(velocities, axis=1)
        states = np.column_stack([
            positions[:, 0],
            positions[:, 1],
            headings,
            v
        ])
    elif len(velocities) == num_points:
        # Compute heading from velocity if not available
        v = np.linalg.norm(velocities, axis=1)
        headings_computed = np.arctan2(velocities[:, 1], velocities[:, 0])
        states = np.column_stack([
            positions[:, 0],
            positions[:, 1],
            headings_computed,
            v
        ])
    else:
        # Compute from positions only
        vx = np.gradient(positions[:, 0]) / 0.1  # Assuming DT = 0.1
        vy = np.gradient(positions[:, 1]) / 0.1
        v = np.sqrt(vx**2 + vy**2)
        headings_computed = np.arctan2(vy, vx)
        states = np.column_stack([
            positions[:, 0],
            positions[:, 1],
            headings_computed,
            v
        ])
    
    return states.astype(np.float32)


def compute_controls_from_states(
    states: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Compute control inputs from state sequence using finite differences.
    
    For vehicle model: control = [acceleration, angular_velocity]
    
    Args:
        states: State sequence [seq_len, state_dim] where state_dim = [x, y, θ, v]
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Control sequence [seq_len, control_dim] where control_dim = [a, ω]
    """
    if len(states) < 2:
        return np.zeros((len(states), 2), dtype=np.float32)
    
    seq_len = len(states)
    controls = np.zeros((seq_len, 2), dtype=np.float32)
    
    # Extract components
    x = states[:, 0]
    y = states[:, 1]
    theta = states[:, 2]
    v = states[:, 3]
    
    # Compute acceleration: a = dv/dt
    # Use forward difference for first point, backward for last, central for others
    dv = np.gradient(v) / dt
    controls[:, 0] = dv  # Acceleration
    
    # Compute angular velocity: ω = dθ/dt
    # Handle angle wrapping
    dtheta = np.diff(theta)
    # Wrap angles to [-π, π]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    dtheta = np.concatenate([[dtheta[0]], dtheta])  # Repeat first for first point
    controls[:, 1] = dtheta / dt  # Angular velocity
    
    return controls


def load_trajectories_from_selected_file(
    file_path: str = 'Learn_actions/selected_trajectories.json',
    object_type: Optional[str] = None,
    num_training_trajectories: Optional[int] = None,
    min_length: int = 10
) -> List[Dict]:
    """
    Load trajectories from selected_trajectories.json file.
    
    Args:
        file_path: Path to selected_trajectories.json file
        object_type: Filter by object type (e.g., 'vehicle', 'pedestrian'). If None, loads all.
        num_training_trajectories: Maximum number of trajectories to load
        min_length: Minimum trajectory length to include
    
    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    file_path = Path(file_path)
    
    # Resolve to absolute path for better error messages
    if not file_path.is_absolute():
        file_path = file_path.resolve()
    
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist. Returning empty list.")
        print(f"  Current working directory: {Path.cwd()}")
        return trajectories
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle empty file
        if not data or len(data) == 0:
            print(f"Warning: {file_path} is empty. Returning empty list.")
            return trajectories
        
        # Data structure: { "filename:track_id": trajectory_dict, ... }
        for key, traj in data.items():
            if not isinstance(traj, dict):
                continue
            
            # Filter by object type if specified
            if object_type:
                traj_obj_type = traj.get('object_type', '').lower()
                if traj_obj_type != object_type.lower():
                    continue
            
            # Filter by minimum length
            positions = traj.get('positions', [])
            if len(positions) < min_length:
                continue
            
            trajectories.append(traj)
            
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from {file_path}: {e}")
        return trajectories
    except Exception as e:
        print(f"Error loading trajectories from {file_path}: {e}")
        return trajectories
    
    if num_training_trajectories is not None:
        if num_training_trajectories > len(trajectories):
            print(f"Warning: Requested {num_training_trajectories} training trajectories, "
                  f"but only {len(trajectories)} available. Using all {len(trajectories)}.")
        else:
            trajectories = trajectories[:num_training_trajectories]
            print(f"Using {len(trajectories)} trajectories for training (limited by --num_training_trajectories)")
    
    return trajectories


def load_trajectories_from_directory(
    trajectories_dir: str,
    format: str = 'json',
    object_type: Optional[str] = 'vehicle',
    max_files: Optional[int] = None,
    num_training_trajectories: Optional[int] = None,
    min_length: int = 10
) -> List[Dict]:
    """
    Load trajectories from directory.
    
    Args:
        trajectories_dir: Directory containing trajectory files
        format: File format ('json' or 'pkl')
        object_type: Filter by object type (e.g., 'vehicle', 'pedestrian')
        num_training_trajectories: Maximum number of trajectories to load
        min_length: Minimum trajectory length to include
    
    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    trajectories_dir = Path(trajectories_dir)
    
    if format == 'json':
        pattern = '*.json'
    elif format == 'pkl':
        pattern = '*.pkl'
    else:
        raise ValueError(f"Unknown format: {format}")
    
    files = list(trajectories_dir.glob(pattern))
    
    if max_files:
        files = files[:max_files]
    
    for file_path in files:
        try:
            if format == 'json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Handle both single trajectory and dictionary of trajectories
            if isinstance(data, dict):
                if 'object_type' in data:
                    # Single trajectory
                    traj_list = [data]
                else:
                    # Dictionary of trajectories
                    traj_list = list(data.values())
            elif isinstance(data, list):
                traj_list = data
            else:
                continue
            
            for traj in traj_list:
                if not isinstance(traj, dict):
                    continue
                
                # Filter by object type
                if object_type:
                    traj_obj_type = traj.get('object_type', '').lower()
                    if traj_obj_type != object_type.lower():
                        continue
                
                # Filter by minimum length
                positions = traj.get('positions', [])
                if len(positions) < min_length:
                    continue
                
                trajectories.append(traj)
        except Exception as e:
            # Skip files that can't be loaded
            continue
    
    if num_training_trajectories is not None:
        if num_training_trajectories > len(trajectories):
            print(f"Warning: Requested {num_training_trajectories} training trajectories, "
                  f"but only {len(trajectories)} available. Using all {len(trajectories)}.")
        else:
            trajectories = trajectories[:num_training_trajectories]
            print(f"Using {len(trajectories)} trajectories for training (limited by --num_training_trajectories)")
        
    return trajectories


def prepare_training_data(
    trajectories: List[Dict],
    state_dim: int = 4,
    dt: float = 0.1
) -> np.ndarray:
    """
    Prepare training data from trajectories (states only, no controls).
    
    Args:
        trajectories: List of trajectory dictionaries
        state_dim: Dimension of state vector
        dt: Time step (unused, kept for interface consistency)
    
    Returns:
        States array [N, seq_len, state_dim]
    """
    states_list = []
    
    for traj in trajectories:
        # Extract states
        states = extract_states_from_trajectory(traj, state_dim)
        if len(states) == 0:
            continue
        
        states_list.append(states)
    
    # Pad sequences to same length (or handle variable length with DataLoader)
    # For simplicity, we'll use the maximum length
    if states_list:
        max_len = max(len(s) for s in states_list)
        
        # Pad sequences
        states_padded = []
        
        for states in states_list:
            seq_len = len(states)
            if seq_len < max_len:
                # Pad with last state
                padding_states = np.tile(states[-1:], (max_len - seq_len, 1))
                states = np.vstack([states, padding_states])
            
            states_padded.append(states)
        
        states_array = np.array(states_padded)
    else:
        states_array = np.array([])
    
    return states_array
