"""
Script to analyze test results from vehicle_test_results folder.

This script reads JSON files containing trajectory test results and calculates:
1. Mean square error between generated trajectory and actual complete trajectory
2. Average waypoint error (error between generated states and downsampled states at waypoint indices)
3. Average squared jerk of the generated states
4. Mean action magnitude
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def compute_mse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute mean square error between predicted and actual trajectories.
    
    Args:
        predicted: [seq_len, 4] predicted trajectory states
        actual: [seq_len, 4] actual trajectory states
    
    Returns:
        MSE (scalar)
    """
    # Use position (x, y) for MSE calculation
    mse = np.mean((predicted[:, :2] - actual[:, :2]) ** 2)
    return float(mse)


def compute_waypoint_error(
    predicted: np.ndarray,
    actual_downsampled: np.ndarray,
    downsample_ratio: int
) -> float:
    """
    Compute average waypoint error between generated states and downsampled states
    at the time indices of the downsampled states.
    
    Args:
        predicted: [seq_len, 4] predicted trajectory states
        actual_downsampled: [downsampled_seq_len, 4] downsampled actual trajectory states
        downsample_ratio: Ratio used for downsampling
    
    Returns:
        Average waypoint error (scalar)
    """
    if actual_downsampled is None or len(actual_downsampled) == 0:
        return float('nan')
    
    num_waypoints = len(actual_downsampled)
    
    # Get indices where waypoints are located
    # If downsample_ratio is R, waypoints are at indices: 0, R, 2R, 3R, ...
    # But we need to handle cases where predicted trajectory might be shorter or longer
    if downsample_ratio > 1:
        waypoint_indices = np.arange(0, min(len(predicted), num_waypoints * downsample_ratio), downsample_ratio)
    else:
        # No downsampling, compare all points
        waypoint_indices = np.arange(0, min(len(predicted), num_waypoints))
    
    # Limit to available waypoints
    waypoint_indices = waypoint_indices[:num_waypoints]
    num_valid_waypoints = len(waypoint_indices)
    
    if num_valid_waypoints == 0:
        return float('nan')
    
    # Extract predicted states at waypoint indices
    predicted_at_waypoints = predicted[waypoint_indices]
    
    # Compare with actual downsampled states (only up to num_valid_waypoints)
    actual_at_waypoints = actual_downsampled[:num_valid_waypoints]
    
    # Compute position error (x, y) at waypoints
    errors = np.linalg.norm(predicted_at_waypoints[:, :2] - actual_at_waypoints[:, :2], axis=1)
    avg_error = np.mean(errors)
    
    return float(avg_error)


def compute_jerk_magnitude(states: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute average square root of jerk (jerk magnitude) of the trajectory states.
    
    Jerk is the third derivative of position: j = d³x/dt³
    
    We compute jerk directly from position:
    - Velocity: v = dx/dt
    - Acceleration: a = dv/dt = d²x/dt²
    - Jerk: j = da/dt = d³x/dt³
    
    Then compute the magnitude: |j| = sqrt(jx² + jy²)
    
    Args:
        states: [seq_len, 4] trajectory states [x, y, θ, v]
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Average jerk magnitude (scalar)
    """
    if len(states) < 3:
        return 0.0
    
    # Extract position components
    x = states[:, 0]
    y = states[:, 1]
    
    # Compute velocity (first derivative of position)
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    
    # Compute acceleration (second derivative of position)
    ax = np.gradient(vx) / dt
    ay = np.gradient(vy) / dt
    
    # Compute jerk (third derivative of position)
    jx = np.gradient(ax) / dt
    jy = np.gradient(ay) / dt
    
    # Compute jerk magnitude: sqrt(jx² + jy²)
    jerk_magnitude = np.sqrt(jx**2 + jy**2)
    
    # Return average jerk magnitude
    avg_jerk_magnitude = np.mean(jerk_magnitude)
    
    return float(avg_jerk_magnitude)


def compute_actions_from_states(states: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Compute control actions from state sequence.
    
    For vehicle model: control = [acceleration, angular_velocity]
    
    Args:
        states: [seq_len, 4] trajectory states [x, y, θ, v]
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Actions [seq_len, 2] where actions = [a, ω]
    """
    if len(states) < 2:
        return np.zeros((len(states), 2), dtype=np.float32)
    
    # Extract components
    theta = states[:, 2]
    v = states[:, 3]
    
    # Compute acceleration: a = dv/dt
    dv = np.gradient(v) / dt
    acceleration = dv
    
    # Compute angular velocity: ω = dθ/dt
    # Handle angle wrapping
    dtheta = np.diff(theta)
    # Wrap angles to [-π, π]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    dtheta = np.concatenate([[dtheta[0]], dtheta])  # Repeat first for first point
    angular_velocity = dtheta / dt
    
    actions = np.column_stack([acceleration, angular_velocity])
    
    return actions


def compute_mean_action_magnitude(states: np.ndarray, dt: float = 0.1) -> float:
    """
    Compute mean action magnitude from trajectory states.
    
    Args:
        states: [seq_len, 4] trajectory states [x, y, θ, v]
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Mean action magnitude (scalar)
    """
    actions = compute_actions_from_states(states, dt)
    
    # Compute magnitude: ||action|| = sqrt(a² + ω²)
    action_magnitudes = np.linalg.norm(actions, axis=1)
    
    # Return mean magnitude
    mean_magnitude = np.mean(action_magnitudes)
    
    return float(mean_magnitude)


def analyze_trajectory(trajectory_data: Dict, dt: float = 0.1) -> Dict:
    """
    Analyze a single trajectory and compute all metrics.
    
    Args:
        trajectory_data: Dictionary containing trajectory data from JSON
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Dictionary with computed metrics
    """
    # Extract data
    traj_idx = trajectory_data.get('trajectory_index', -1)
    predicted_trajectory = np.array(trajectory_data.get('predicted_trajectory', []))
    actual_trajectory_complete = np.array(trajectory_data.get('actual_trajectory_complete', []))
    actual_trajectory_downsampled = trajectory_data.get('actual_trajectory_downsampled')
    downsample_ratio = trajectory_data.get('downsample_ratio', 1)
    
    if actual_trajectory_downsampled is not None:
        actual_trajectory_downsampled = np.array(actual_trajectory_downsampled)
    
    # Check if we have valid data
    if len(predicted_trajectory) == 0 or len(actual_trajectory_complete) == 0:
        return {
            'trajectory_index': traj_idx,
            'mse': float('nan'),
            'waypoint_error': float('nan'),
            'avg_jerk_magnitude': float('nan'),
            'avg_jerk_magnitude_actual': float('nan'),
            'mean_action_magnitude': float('nan')
        }
    
    # Compute metrics
    mse = compute_mse(predicted_trajectory, actual_trajectory_complete)
    waypoint_error = compute_waypoint_error(
        predicted_trajectory,
        actual_trajectory_downsampled,
        downsample_ratio
    )
    avg_jerk_magnitude = compute_jerk_magnitude(predicted_trajectory, dt)
    avg_jerk_magnitude_actual = compute_jerk_magnitude(actual_trajectory_complete, dt)
    mean_action_magnitude = compute_mean_action_magnitude(predicted_trajectory, dt)
    
    return {
        'trajectory_index': traj_idx,
        'mse': mse,
        'waypoint_error': waypoint_error,
        'avg_jerk_magnitude': avg_jerk_magnitude,
        'avg_jerk_magnitude_actual': avg_jerk_magnitude_actual,
        'mean_action_magnitude': mean_action_magnitude
    }


def analyze_json_file(json_path: Path, dt: float = 0.1) -> List[Dict]:
    """
    Analyze all trajectories in a JSON file.
    
    Args:
        json_path: Path to JSON file
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        List of metric dictionaries, one per trajectory
    """
    print(f"\nAnalyzing: {json_path.name}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    results = []
    for trajectory_data in data:
        metrics = analyze_trajectory(trajectory_data, dt)
        results.append(metrics)
    
    return results


def print_summary(all_results: Dict[str, List[Dict]]):
    """
    Print summary statistics across all files.
    
    Args:
        all_results: Dictionary mapping filename to list of metric dictionaries
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Collect all metrics across all files
    all_mse = []
    all_waypoint_error = []
    all_jerk_magnitude = []
    all_jerk_magnitude_actual = []
    all_action_magnitude = []
    
    for filename, results in all_results.items():
        for result in results:
            if not np.isnan(result['mse']):
                all_mse.append(result['mse'])
            if not np.isnan(result['waypoint_error']):
                all_waypoint_error.append(result['waypoint_error'])
            if not np.isnan(result['avg_jerk_magnitude']):
                all_jerk_magnitude.append(result['avg_jerk_magnitude'])
            if not np.isnan(result['avg_jerk_magnitude_actual']):
                all_jerk_magnitude_actual.append(result['avg_jerk_magnitude_actual'])
            if not np.isnan(result['mean_action_magnitude']):
                all_action_magnitude.append(result['mean_action_magnitude'])
    
    # Print statistics for each metric
    print(f"\nMean Square Error (MSE):")
    if all_mse:
        print(f"  Count: {len(all_mse)}")
        print(f"  Mean:  {np.mean(all_mse):.6f}")
        print(f"  Std:   {np.std(all_mse):.6f}")
        print(f"  Min:   {np.min(all_mse):.6f}")
        print(f"  Max:   {np.max(all_mse):.6f}")
    else:
        print("  No valid data")
    
    print(f"\nAverage Waypoint Error:")
    if all_waypoint_error:
        print(f"  Count: {len(all_waypoint_error)}")
        print(f"  Mean:  {np.mean(all_waypoint_error):.6f}")
        print(f"  Std:   {np.std(all_waypoint_error):.6f}")
        print(f"  Min:   {np.min(all_waypoint_error):.6f}")
        print(f"  Max:   {np.max(all_waypoint_error):.6f}")
    else:
        print("  No valid data")
    
    print(f"\nAverage Jerk Magnitude (Generated):")
    if all_jerk_magnitude:
        print(f"  Count: {len(all_jerk_magnitude)}")
        print(f"  Mean:  {np.mean(all_jerk_magnitude):.6f}")
        print(f"  Std:   {np.std(all_jerk_magnitude):.6f}")
        print(f"  Min:   {np.min(all_jerk_magnitude):.6f}")
        print(f"  Max:   {np.max(all_jerk_magnitude):.6f}")
    else:
        print("  No valid data")
    
    print(f"\nAverage Jerk Magnitude (Actual):")
    if all_jerk_magnitude_actual:
        print(f"  Count: {len(all_jerk_magnitude_actual)}")
        print(f"  Mean:  {np.mean(all_jerk_magnitude_actual):.6f}")
        print(f"  Std:   {np.std(all_jerk_magnitude_actual):.6f}")
        print(f"  Min:   {np.min(all_jerk_magnitude_actual):.6f}")
        print(f"  Max:   {np.max(all_jerk_magnitude_actual):.6f}")
    else:
        print("  No valid data")
    
    print(f"\nMean Action Magnitude:")
    if all_action_magnitude:
        print(f"  Count: {len(all_action_magnitude)}")
        print(f"  Mean:  {np.mean(all_action_magnitude):.6f}")
        print(f"  Std:   {np.std(all_action_magnitude):.6f}")
        print(f"  Min:   {np.min(all_action_magnitude):.6f}")
        print(f"  Max:   {np.max(all_action_magnitude):.6f}")
    else:
        print("  No valid data")
    
    # Print per-file statistics
    print("\n" + "="*80)
    print("PER-FILE STATISTICS")
    print("="*80)
    
    for filename, results in all_results.items():
        print(f"\n{filename}:")
        file_mse = [r['mse'] for r in results if not np.isnan(r['mse'])]
        file_waypoint = [r['waypoint_error'] for r in results if not np.isnan(r['waypoint_error'])]
        file_jerk = [r['avg_jerk_magnitude'] for r in results if not np.isnan(r['avg_jerk_magnitude'])]
        file_jerk_actual = [r['avg_jerk_magnitude_actual'] for r in results if not np.isnan(r['avg_jerk_magnitude_actual'])]
        file_action = [r['mean_action_magnitude'] for r in results if not np.isnan(r['mean_action_magnitude'])]
        
        if file_mse:
            print(f"  MSE: Mean={np.mean(file_mse):.6f}, Count={len(file_mse)}")
        if file_waypoint:
            print(f"  Waypoint Error: Mean={np.mean(file_waypoint):.6f}, Count={len(file_waypoint)}")
        if file_jerk:
            print(f"  Jerk Magnitude (Generated): Mean={np.mean(file_jerk):.6f}, Count={len(file_jerk)}")
        if file_jerk_actual:
            print(f"  Jerk Magnitude (Actual): Mean={np.mean(file_jerk_actual):.6f}, Count={len(file_jerk_actual)}")
        if file_action:
            print(f"  Action Magnitude: Mean={np.mean(file_action):.6f}, Count={len(file_action)}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze test results from vehicle_test_results folder'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='vehicle_test_results',
        help='Directory containing JSON result files (default: vehicle_test_results)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='Time step for trajectory (default: 0.1 seconds)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file to save results (optional)'
    )
    
    args = parser.parse_args()
    
    # Find all JSON files in results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    json_files = sorted(results_dir.glob('*.json'))
    
    if len(json_files) == 0:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) in {results_dir}")
    
    # Analyze all files
    all_results = {}
    for json_file in json_files:
        try:
            results = analyze_json_file(json_file, args.dt)
            all_results[json_file.name] = results
            
            # Print per-trajectory results for this file
            print(f"\n  Trajectories analyzed: {len(results)}")
            for result in results[:5]:  # Show first 5
                print(f"    Traj {result['trajectory_index']}: "
                      f"MSE={result['mse']:.6f}, "
                      f"WaypointErr={result['waypoint_error']:.6f}, "
                      f"JerkMag={result['avg_jerk_magnitude']:.6f}, "
                      f"JerkMagActual={result['avg_jerk_magnitude_actual']:.6f}, "
                      f"ActionMag={result['mean_action_magnitude']:.6f}")
            if len(results) > 5:
                print(f"    ... and {len(results) - 5} more")
        except Exception as e:
            print(f"Error analyzing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print_summary(all_results)
    
    # Save results if requested
    if args.output:
        output_data = {
            'dt': args.dt,
            'results_by_file': all_results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
