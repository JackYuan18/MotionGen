"""
Script to plot trajectories from trajectory_matching_results.json.

This script loads the results from test_trajectories_ppo.py and creates
visualization plots showing the predicted trajectory, downsampled actual trajectory,
and complete actual trajectory for each tested trajectory.
"""

import os
# Fix OpenMP library conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_trajectory_comparison(
    result: dict,
    output_dir: Path,
    traj_idx: int
):
    """
    Plot comparison of predicted, downsampled actual, and complete actual trajectories.
    
    Args:
        result: Dictionary containing trajectory data from results JSON
        output_dir: Directory to save the plot
        traj_idx: Trajectory index for filename
    """
    # Extract trajectories
    predicted_trajectory = np.array(result['predicted_trajectory'])  # [seq_len, 4]
    actual_trajectory_complete = np.array(result['actual_trajectory_complete'])  # [seq_len, 4]
    actual_trajectory_downsampled = result.get('actual_trajectory_downsampled')
    
    if actual_trajectory_downsampled is not None:
        actual_trajectory_downsampled = np.array(actual_trajectory_downsampled)  # [downsampled_seq_len, 4]
    
    # Extract metadata
    matching_error = result.get('matching_error', 0.0)
    best_loss = result.get('best_loss', 0.0)
    best_epoch = result.get('best_epoch', 0)
    downsample_ratio = result.get('downsample_ratio')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot complete actual trajectory
    ax.plot(
        actual_trajectory_complete[:, 0],
        actual_trajectory_complete[:, 1],
        'b-',
        linewidth=2,
        label='Actual (complete)',
        alpha=0.8,
        zorder=10
    )
    
    # Plot downsampled actual trajectory if available
    if actual_trajectory_downsampled is not None:
        ax.plot(
            actual_trajectory_downsampled[:, 0],
            actual_trajectory_downsampled[:, 1],
            'b.',
            markersize=25,
            label=f'Waypoints (downsampled, ratio={downsample_ratio})',
            alpha=0.7,
            zorder=11
        )
    
    # Plot predicted trajectory
    ax.plot(
        predicted_trajectory[:, 0],
        predicted_trajectory[:, 1],
        'r-',
        linewidth=2,
        label='Generated Trajectory',
        alpha=0.8,
        zorder=9
    )
    
    # Plot start point
    ax.plot(
        actual_trajectory_complete[0, 0],
        actual_trajectory_complete[0, 1],
        'go',
        markersize=10,
        label='Start',
        zorder=15
    )
    
  
    
    # Set labels and title
    ax.set_xlabel('X Position (m)', fontsize=25)
    ax.set_ylabel('Y Position (m)', fontsize=25)
    ax.set_title(
        f'Trajectory {traj_idx + 1}: Best Policy (Epoch {best_epoch},  Error={matching_error:.6f})',
        fontsize=25,
        fontweight='bold'
    )
    ax.legend(loc='best', fontsize=25)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add text box with statistics
    stats_text = (
        f'Matching Error: {matching_error:.6f}\n'
        f'Best Epoch: {best_epoch}'
    )
    if downsample_ratio is not None:
        stats_text += f'\nDownsample Ratio: {downsample_ratio}'
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=25,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'trajectory_{traj_idx + 1}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for trajectory {traj_idx + 1} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot trajectories from all ppo_trajectories JSON files in test_results folder'
    )
    
    parser.add_argument(
        '--results_file',
        type=str,
        default=None,
        help='Path to specific trajectory matching results JSON file (if not provided, uses all JSON files in test_results)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='vehicle_trajectory_plots',
        help='Output directory for plots'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    test_results_dir = script_dir / 'vehicle_test_results'
    
    # Find all JSON files to process
    if args.results_file is not None:
        # Use explicitly provided file path
        results_file = Path(args.results_file)
        if not results_file.is_absolute():
            if (script_dir / results_file).exists():
                results_file = script_dir / results_file
            elif (test_results_dir / results_file).exists():
                results_file = test_results_dir / results_file
            elif (script_dir / results_file.name).exists():
                results_file = script_dir / results_file.name
            else:
                results_file = results_file.resolve()
        
        if not results_file.exists():
            print(f"Error: Results file not found: {results_file}")
            return
        
        json_files = [results_file]
    else:
        # Find all JSON files in test_results directory
        if not test_results_dir.exists():
            print(f"Error: Test results directory not found: {test_results_dir}")
            return
        
        json_files = sorted(test_results_dir.glob('vehicle_ppo_trajectories_*.json'))
        
        if not json_files:
            print(f"No JSON files found in {test_results_dir}")
            print("Expected files matching pattern: ppo_trajectories_*.json")
            return
        
        print(f"Found {len(json_files)} JSON file(s) in test_results directory:")
        for f in json_files:
            print(f"  - {f.name}")
    
    # Process each JSON file
    for results_file in json_files:
        print(f"\n{'='*60}")
        print(f"Processing: {results_file.name}")
        print(f"{'='*60}")
        
        # Create output directory for this file (based on filename)
        file_stem = results_file.stem  # e.g., 'ppo_trajectories_4'
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
        
        # Create subdirectory based on filename
        file_output_dir = output_dir / file_stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading results from: {results_file}")
        print(f"Output directory: {file_output_dir}")
        
        # Load results
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
        
        if not results:
            print(f"Warning: No results found in {results_file.name}")
            continue
        
        print(f"Found {len(results)} trajectories to plot")
        
        # Plot each trajectory
        for idx, result in enumerate(results):
            try:
                plot_trajectory_comparison(result, file_output_dir, idx)
            except Exception as e:
                print(f"Error plotting trajectory {idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"All plots saved to: {file_output_dir}")
        
        # Print summary statistics for this file
        if results:
            errors = [r.get('matching_error', 0.0) for r in results]
            losses = [r.get('best_loss', 0.0) for r in results]
            
            print(f"\nSummary Statistics for {results_file.name}:")
            print(f"  Matching Errors - Mean: {np.mean(errors):.6f}, Std: {np.std(errors):.6f}")
            print(f"  Best Losses - Mean: {np.mean(losses):.6f}, Std: {np.std(losses):.6f}")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
