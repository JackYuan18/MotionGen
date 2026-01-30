"""
Shared argument parser configuration for PPO training scripts.

This module provides common argument definitions to ensure consistency
across different PPO training and testing scripts.
"""

import argparse
from utils import DT


def add_ppo_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common PPO training arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    
    Returns:
        The same parser instance (for chaining)
    """
    # Network architecture arguments
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=256,
        help='Hidden dimension for LSTM and networks'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of LSTM layers'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=80,
        help='Number of rollouts per training step'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    # parser.add_argument(
    #     '--downsample_ratio',
    #     type=int,
    #     default=4,
    #     help='Downsample ratio for actual states during training (default: 1, no downsampling). '
    #          'If > 1, every Nth state is kept. Complete trajectory is used for evaluation and visualization.'
    # )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0003,
        help='Learning rate'
    )
    parser.add_argument(
        '--ppo_epochs',
        type=int,
        default=4,
        help='Number of PPO update epochs per training step'
    )
    parser.add_argument(
        '--clip_epsilon',
        type=float,
        default=0.2,
        help='PPO clipping parameter'
    )
    parser.add_argument(
        '--value_coef',
        type=float,
        default=0.5,
        help='Coefficient for value loss'
    )
    parser.add_argument(
        '--entropy_coef',
        type=float,
        default=0.085,
        help='Coefficient for entropy bonus'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor for computing returns'
    )
    
    
    # System arguments
    parser.add_argument(
        '--dt',
        type=float,
        default=DT,
        help='Discrete time step'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (auto, cpu, or cuda)'
    )
    
    return parser


def add_trajectory_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add trajectory data loading arguments to an argument parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    
    Returns:
        The same parser instance (for chaining)
    """
    parser.add_argument(
        '--selected_trajectories_file',
        type=str,
        default='selected_trajectories.json',
        help='Path to selected trajectories JSON file'
    )
    parser.add_argument(
        '--object_type',
        type=str,
        default=None,
        help='Object type to filter (e.g., vehicle, pedestrian). If None, loads all types.'
    )
    parser.add_argument(
        '--num_training_trajectories',
        type=int,
        default=None,
        help='Number of trajectories to use for training'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=10,
        help='Minimum trajectory length'
    )
    
    
    return parser
