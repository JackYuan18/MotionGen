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
    seq_len_pred = predicted.shape[0]
    seq_len_actual = actual.shape[0]
    min_len = min(seq_len_pred, seq_len_actual)
    # Compute MSE for overlapping portion
    norm = np.linalg.norm((predicted[:min_len, :2] - actual[:min_len, :2]), axis=1)

    return np.mean(norm) + np.linalg.norm((predicted[-1, :2] - actual[-1, :2]))*(float(seq_len_pred > seq_len_actual))

def compute_waypoint_error(
    predicted: np.ndarray,
    actual_downsampled: np.ndarray,
    downsampled_indices: np.ndarray
) -> float:
    """
    Compute average waypoint error between generated states and downsampled states
    at the time indices of the downsampled states.
    
    Args:
        predicted: [seq_len, 4] predicted trajectory states
        actual_downsampled: [downsampled_seq_len, 4] downsampled actual trajectory states
        downsampled_indices: Array of indices where waypoints are located in the complete trajectory
    
    Returns:
        Average waypoint error (scalar)
    """
    if actual_downsampled is None or len(actual_downsampled) == 0:
        return float('nan')
    
    if downsampled_indices is None or len(downsampled_indices) == 0:
        return float('nan')
    
    # Convert to numpy array if needed
    if not isinstance(downsampled_indices, np.ndarray):
        downsampled_indices = np.array(downsampled_indices)
    
    # Ensure indices are within bounds of predicted trajectory
    valid_mask = (downsampled_indices >= 0) & (downsampled_indices < len(predicted))
    waypoint_indices = downsampled_indices[valid_mask]
    
    # Limit to available waypoints (should match length of actual_downsampled)
    num_waypoints = min(len(waypoint_indices), len(actual_downsampled))
    
    if num_waypoints == 0:
        return float('nan')
    
    # Extract predicted states at waypoint indices
    predicted_at_waypoints = predicted[waypoint_indices[:num_waypoints]]
    
    # Compare with actual downsampled states (only up to num_waypoints)
    actual_at_waypoints = actual_downsampled[:num_waypoints]
    
    # Compute position error (x, y) at waypoints
    errors = np.linalg.norm(predicted_at_waypoints[:, :2] - actual_at_waypoints[:, :2], axis=1)
    avg_error = np.mean(errors)
    
    return float(avg_error)


def compute_jerk_acceleration_velocity_magnitude(states: np.ndarray, dt: float = 0.1) -> float:
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
    jerk_magnitudes = np.sqrt(jx**2 + jy**2)
    acceleration_magnitudes= np.linalg.norm(np.column_stack([ax, ay]), axis=1)
    velocity_magnitudes = np.linalg.norm(np.column_stack([vx, vy]), axis=1)
    # Return average jerk magnitude
    
    
    return {
        'velocity_magnitudes': velocity_magnitudes,
        'acceleration_magnitudes': acceleration_magnitudes,
        'jerk_magnitudes': jerk_magnitudes,
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


def analyze_trajectory(trajectory_data: Dict, dt: float = 0.1) -> Dict:
    """
    Analyze a single trajectory and compute all metrics.
    
    Args:
        trajectory_data: Dictionary containing trajectory data from JSON for one downsampled
        dt: Time step (default: 0.1 seconds)
    
    Returns:
        Dictionary with computed metrics
    """
    # Extract data
    traj_idx = trajectory_data.get('trajectory_index', -1)
    predicted_trajectory = np.array(trajectory_data.get('predicted_trajectory', []))
    actual_trajectory_complete = np.array(trajectory_data.get('actual_trajectory_complete', []))
    actual_trajectory_downsampled = trajectory_data.get('actual_trajectory_downsampled')
    downsampled_indices = trajectory_data.get('Downsampled_indices', None)

    
    if actual_trajectory_downsampled is not None:
        actual_trajectory_downsampled = np.array(actual_trajectory_downsampled)
    
    # Convert downsampled_indices to numpy array if available
    if downsampled_indices is not None:
        if not isinstance(downsampled_indices, np.ndarray):
            downsampled_indices = np.array(downsampled_indices)
    
    # Check if we have valid data
    if len(predicted_trajectory) == 0 or len(actual_trajectory_complete) == 0:
        return {
            'trajectory_index': traj_idx,
            'mse': float('nan'),
            'waypoint_error': float('nan'),
            'avg_jerk_magnitude': float('nan'),
            'avg_jerk_magnitude_actual': float('nan'),
            'mean_action_magnitude': float('nan'),
            'max_velocity_predicted': float('nan'),
            'max_acceleration_predicted': float('nan'),
            'max_jerk_predicted': float('nan'),
            'mean_velocity_predicted': float('nan'),
            'std_velocity_predicted': float('nan'),
            'mean_acceleration_predicted': float('nan'),
            'std_acceleration_predicted': float('nan'),
            'max_velocity_actual': float('nan'),
            'max_acceleration_actual': float('nan'),
            'max_jerk_actual': float('nan'),
            'mean_velocity_actual': float('nan'),
            'std_velocity_actual': float('nan'),
            'mean_acceleration_actual': float('nan'),
            'std_acceleration_actual': float('nan')
        }
    
    # Compute metrics
    mse = compute_mse(predicted_trajectory, actual_trajectory_complete)
    waypoint_error = compute_waypoint_error(
        predicted_trajectory,
        actual_trajectory_downsampled,
        downsampled_indices
    )
    
    
    # Compute velocity and acceleration statistics for predicted trajectory
    predicted_stats = compute_jerk_acceleration_velocity_magnitude(predicted_trajectory, dt)
    
    # Compute velocity and acceleration statistics for actual trajectory
    actual_stats = compute_jerk_acceleration_velocity_magnitude(actual_trajectory_complete, dt)

    
    return {
        'trajectory_index': traj_idx,
        'mse': mse,
        'waypoint_error': waypoint_error,
        'avg_jerk_magnitude_predicted': np.mean(predicted_stats['jerk_magnitudes']),
        'std_jerk_magnitudes_predicted': np.std(predicted_stats['jerk_magnitudes']),
        'max_jerk_predicted': np.max(np.abs(predicted_stats['jerk_magnitudes'])),
        
        'avg_jerk_magnitude_actual': np.mean(actual_stats['jerk_magnitudes']),
        'std_jerk_magnitudes_actual': np.std(actual_stats['jerk_magnitudes']),
        'max_jerk_actual': np.max(np.abs(actual_stats['jerk_magnitudes'])),
        
        'avg_velocity_magnitudes_predicted': np.mean(predicted_stats['velocity_magnitudes']),
        'std_velocity_magnitudes_predicted': np.std(predicted_stats['velocity_magnitudes']),
        'max_velocity_predicted': np.max(predicted_stats['velocity_magnitudes']),

        'avg_acceleration_magnitudes_predicted': np.mean(predicted_stats['acceleration_magnitudes']),
        'std_acceleration_magnitudes_predicted': np.std(predicted_stats['acceleration_magnitudes']),
        'max_acceleration_predicted': np.max(predicted_stats['acceleration_magnitudes'])        
    }




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
    # New metrics for predicted trajectories

    all_velocity_magnitudes_predicted = []
    all_accelerations_magnitudes_predicted = []
    all_velocity_magnitudes_actual = []
    all_accelerations_magnitudes_actual = []
    # New metrics for actual trajectories
    
    
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
            # Collect new metrics for predicted
            if not np.isnan(result['velocities_magnitudes_predicted']):
                all_velocity_magnitudes_predicted.append(result['velocities_magnitudes_predicted'])
            if not np.isnan(result['accelerations_magnitudes_predicted']):
                all_accelerations_magnitudes_predicted.append(result['accelerations_magnitudes_predicted'])
            if not np.isnan(result['velocities_magnitudes_actual']):
                all_velocity_magnitudes_actual.append(result['velocities_magnitudes_actual'])
            if not np.isnan(result['accelerations_magnitudes_actual']):
                all_accelerations_magnitudes_actual.append(result['accelerations_magnitudes_actual'])
    
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
    
    # Print new metrics for predicted trajectories
    print(f"\nVelocity Magnitudes (Predicted):")
    if all_velocity_magnitudes_predicted:
        print(f"  Count: {len(all_velocity_magnitudes_predicted)}")
        print(f"  Mean:  {np.mean(all_velocity_magnitudes_predicted):.6f}")
        print(f"  Std:   {np.std(all_velocity_magnitudes_predicted):.6f}")
        print(f"  Min:   {np.min(all_velocity_magnitudes_predicted):.6f}")
        print(f"  Max:   {np.max(all_velocity_magnitudes_predicted):.6f}")
    else:
        print("  No valid data")
    print(f"\nAccelerations Magnitudes (Predicted):")
    if all_accelerations_magnitudes_predicted:
        print(f"  Count: {len(all_accelerations_magnitudes_predicted)}")
        print(f"  Mean:  {np.mean(all_accelerations_magnitudes_predicted):.6f}")
        print(f"  Std:   {np.std(all_accelerations_magnitudes_predicted):.6f}")
        print(f"  Min:   {np.min(all_accelerations_magnitudes_predicted):.6f}")
        print(f"  Max:   {np.max(all_accelerations_magnitudes_predicted):.6f}")
    else:
        print("  No valid data")
    print(f"\nVelocity Magnitudes (Actual):")
    if all_velocity_magnitudes_actual:
        print(f"  Count: {len(all_velocity_magnitudes_actual)}")
        print(f"  Mean:  {np.mean(all_velocity_magnitudes_actual):.6f}")
        print(f"  Std:   {np.std(all_velocity_magnitudes_actual):.6f}")
        print(f"  Min:   {np.min(all_velocity_magnitudes_actual):.6f}")
        print(f"  Max:   {np.max(all_velocity_magnitudes_actual):.6f}")
    else:
        print("  No valid data")
    print(f"\nAccelerations Magnitudes (Actual):")
    if all_accelerations_magnitudes_actual:
        print(f"  Count: {len(all_accelerations_magnitudes_actual)}")
        print(f"  Mean:  {np.mean(all_accelerations_magnitudes_actual):.6f}")
        print(f"  Std:   {np.std(all_accelerations_magnitudes_actual):.6f}")
        print(f"  Min:   {np.min(all_accelerations_magnitudes_actual):.6f}")
        print(f"  Max:   {np.max(all_accelerations_magnitudes_actual):.6f}")
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
        # New metrics for predicted
        file_max_vel_pred = [r.get('max_velocity_predicted', np.nan) for r in results if not np.isnan(r.get('max_velocity_predicted', np.nan))]
        file_max_acc_pred = [r.get('max_acceleration_predicted', np.nan) for r in results if not np.isnan(r.get('max_acceleration_predicted', np.nan))]
        file_max_jerk_pred = [r.get('max_jerk_predicted', np.nan) for r in results if not np.isnan(r.get('max_jerk_predicted', np.nan))]
        file_mean_vel_pred = [r.get('mean_velocity_predicted', np.nan) for r in results if not np.isnan(r.get('mean_velocity_predicted', np.nan))]
        file_std_vel_pred = [r.get('std_velocity_predicted', np.nan) for r in results if not np.isnan(r.get('std_velocity_predicted', np.nan))]
        file_mean_acc_pred = [r.get('mean_acceleration_predicted', np.nan) for r in results if not np.isnan(r.get('mean_acceleration_predicted', np.nan))]
        file_std_acc_pred = [r.get('std_acceleration_predicted', np.nan) for r in results if not np.isnan(r.get('std_acceleration_predicted', np.nan))]
        # New metrics for actual
        file_max_vel_actual = [r.get('max_velocity_actual', np.nan) for r in results if not np.isnan(r.get('max_velocity_actual', np.nan))]
        file_max_acc_actual = [r.get('max_acceleration_actual', np.nan) for r in results if not np.isnan(r.get('max_acceleration_actual', np.nan))]
        file_max_jerk_actual = [r.get('max_jerk_actual', np.nan) for r in results if not np.isnan(r.get('max_jerk_actual', np.nan))]
        file_mean_vel_actual = [r.get('mean_velocity_actual', np.nan) for r in results if not np.isnan(r.get('mean_velocity_actual', np.nan))]
        file_std_vel_actual = [r.get('std_velocity_actual', np.nan) for r in results if not np.isnan(r.get('std_velocity_actual', np.nan))]
        file_mean_acc_actual = [r.get('mean_acceleration_actual', np.nan) for r in results if not np.isnan(r.get('mean_acceleration_actual', np.nan))]
        file_std_acc_actual = [r.get('std_acceleration_actual', np.nan) for r in results if not np.isnan(r.get('std_acceleration_actual', np.nan))]
        
        if file_mse:
            print(f"  MSE: Mean={np.mean(file_mse):.6f}, Std={np.std(file_mse):.6f}, Count={len(file_mse)}")
        if file_waypoint:
            print(f"  Waypoint Error: Mean={np.mean(file_waypoint):.6f}, Std={np.std(file_waypoint):.6f}, Count={len(file_waypoint)}")
        if file_jerk:
            print(f"  Jerk Magnitude (Generated): Mean={np.mean(file_jerk):.6f}, Std={np.std(file_jerk):.6f}, Count={len(file_jerk)}")
        if file_jerk_actual:
            print(f"  Jerk Magnitude (Actual): Mean={np.mean(file_jerk_actual):.6f}, Std={np.std(file_jerk_actual):.6f}, Count={len(file_jerk_actual)}")
        if file_action:
            print(f"  Action Magnitude: Mean={np.mean(file_action):.6f}, Std={np.std(file_action):.6f}, Count={len(file_action)}")
        # Print new metrics for predicted
        if file_max_vel_pred:
            print(f"  Max Velocity (Predicted): Mean={np.mean(file_max_vel_pred):.6f}, Std={np.std(file_max_vel_pred):.6f}, Count={len(file_max_vel_pred)}")
        if file_max_acc_pred:
            print(f"  Max Acceleration (Predicted): Mean={np.mean(file_max_acc_pred):.6f}, Std={np.std(file_max_acc_pred):.6f}, Count={len(file_max_acc_pred)}")
        if file_max_jerk_pred:
            print(f"  Max Jerk (Predicted): Mean={np.mean(file_max_jerk_pred):.6f}, Std={np.std(file_max_jerk_pred):.6f}, Count={len(file_max_jerk_pred)}")
        if file_mean_vel_pred:
            print(f"  Mean Velocity (Predicted): Mean={np.mean(file_mean_vel_pred):.6f}, Std={np.std(file_mean_vel_pred):.6f}, Count={len(file_mean_vel_pred)}")
        if file_std_vel_pred:
            print(f"  Std Velocity (Predicted): Mean={np.mean(file_std_vel_pred):.6f}, Std={np.std(file_std_vel_pred):.6f}, Count={len(file_std_vel_pred)}")
        if file_mean_acc_pred:
            print(f"  Mean Acceleration (Predicted): Mean={np.mean(file_mean_acc_pred):.6f}, Std={np.std(file_mean_acc_pred):.6f}, Count={len(file_mean_acc_pred)}")
        if file_std_acc_pred:
            print(f"  Std Acceleration (Predicted): Mean={np.mean(file_std_acc_pred):.6f}, Std={np.std(file_std_acc_pred):.6f}, Count={len(file_std_acc_pred)}")
        # Print new metrics for actual
        if file_max_vel_actual:
            print(f"  Max Velocity (Actual): Mean={np.mean(file_max_vel_actual):.6f}, Std={np.std(file_max_vel_actual):.6f}, Count={len(file_max_vel_actual)}")
        if file_max_acc_actual:
            print(f"  Max Acceleration (Actual): Mean={np.mean(file_max_acc_actual):.6f}, Std={np.std(file_max_acc_actual):.6f}, Count={len(file_max_acc_actual)}")
        if file_max_jerk_actual:
            print(f"  Max Jerk (Actual): Mean={np.mean(file_max_jerk_actual):.6f}, Std={np.std(file_max_jerk_actual):.6f}, Count={len(file_max_jerk_actual)}")
        if file_mean_vel_actual:
            print(f"  Mean Velocity (Actual): Mean={np.mean(file_mean_vel_actual):.6f}, Std={np.std(file_mean_vel_actual):.6f}, Count={len(file_mean_vel_actual)}")
        if file_std_vel_actual:
            print(f"  Std Velocity (Actual): Mean={np.mean(file_std_vel_actual):.6f}, Std={np.std(file_std_vel_actual):.6f}, Count={len(file_std_vel_actual)}")
        if file_mean_acc_actual:
            print(f"  Mean Acceleration (Actual): Mean={np.mean(file_mean_acc_actual):.6f}, Std={np.std(file_mean_acc_actual):.6f}, Count={len(file_mean_acc_actual)}")
        if file_std_acc_actual:
            print(f"  Std Acceleration (Actual): Mean={np.mean(file_std_acc_actual):.6f}, Std={np.std(file_std_acc_actual):.6f}, Count={len(file_std_acc_actual)}")


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

                      f"AvgVelPred={result['avg_velocity_magnitudes_predicted']:.6f}, "
                      f"StdVelPred={result['std_velocity_magnitudes_predicted']:.6f}, "
                      f"MaxVelPred={result['max_velocity_predicted']:.6f}, "

                      f"AvgAccPred={result['avg_acceleration_magnitudes_predicted']:.6f}, "
                      f"MaxAccPred={result['max_acceleration_predicted']:.6f}, "
                      f"StdAccPred={result['std_acceleration_magnitudes_predicted']:.6f}, "

                      f"AvgJerkPred={result['avg_jerk_magnitude_predicted']:.6f}, "
                      f"MaxJerkPred={result['max_jerk_predicted']:.6f}, "
                      f"StdJerkPred={result['std_jerk_magnitudes_predicted']:.6f}, "
                      
                      f"AvgVelActual={result['avg_velocity_magnitudes_actual']:.6f}, "
                      f"StdVelActual={result['std_velocity_magnitudes_actual']:.6f}, "
                      f"MaxVelActual={result['max_velocity_actual']:.6f}, "

                      f"AvgAccActual={result['avg_acceleration_magnitudes_actual']:.6f}, "
                      f"StdAccActual={result['std_acceleration_magnitudes_actual']:.6f}, "
                      f"MaxAccActual={result['max_acceleration_actual']:.6f}, "

                      f"AvgJerkActual={result['avg_jerk_magnitude_actual']:.6f}, "
                      f"StdJerkActual={result['std_jerk_magnitudes_actual']:.6f}, "
                      f"MaxJerkActual={result['max_jerk_actual']:.6f}")
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
