#!/usr/bin/env python3
"""
Script to load saved models from results folder and plot training rewards.
Groups results by model type and averages across different seeds.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
from collections import defaultdict


def smooth_data(data, window_size=10):
    """
    Apply moving average smoothing to data.

    Parameters
    ----------
    data : np.ndarray
        Data to smooth
    window_size : int
        Size of smoothing window

    Returns
    -------
    np.ndarray
        Smoothed data
    """
    if window_size <= 1 or len(data) < window_size:
        return data

    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def extract_model_name(file_path):
    """
    Extract model identifier from file path, removing seed suffix.

    Parameters
    ----------
    file_path : str
        Path to the pickle file

    Returns
    -------
    str
        Model identifier (without seed number)
    """
    file_name = os.path.basename(file_path)
    # Extract meaningful identifier from filename
    # e.g., "Acrobot-v1_GreedyAC_expectile_True0.8_data_1.pkl" -> "GreedyAC_expectile_True0.8"

    # Remove .pkl extension
    name = file_name.replace('.pkl', '')

    # Remove seed identifier (data_X pattern at the end)
    # Match patterns like _data_1, _data_2, etc.
    import re
    name = re.sub(r'_data_\d+$', '', name)

    # Try to extract model configuration
    # This can be customized based on your naming convention
    parts = name.split('_')

    # Find the agent name and relevant config
    if 'expectile' in name:
        # Find expectile configuration
        for i, part in enumerate(parts):
            if part == 'expectile' and i + 1 < len(parts):
                # Get agent name and expectile configuration
                agent_idx = None
                for j in range(i-1, -1, -1):
                    if 'AC' in parts[j] or 'SAC' in parts[j] or 'VAC' in parts[j]:
                        agent_idx = j
                        break

                if agent_idx is not None:
                    # Include expectile and the next 3 parameters: use_expectile, expectile_value, expectile_mode
                    expectile_config = parts[i:i+4]
                    model_id = '_'.join([parts[agent_idx]] + expectile_config)
                    return model_id

    # Default: use agent name and some config identifier
    # Remove environment prefix if present (e.g., "Acrobot-v1_")
    for i, part in enumerate(parts):
        if 'AC' in part or 'SAC' in part or 'VAC' in part:
            # Include agent and everything after it (excluding env name)
            return '_'.join(parts[i:])

    return name


def load_results_from_folder(results_folder="./results"):
    """
    Load all pickle files from the results folder and group by model type.

    Parameters
    ----------
    results_folder : str
        Path to the results folder containing experiment data

    Returns
    -------
    dict
        Dictionary mapping model names to lists of run data
    """
    # Find all pickle files recursively
    pkl_files = glob(os.path.join(results_folder, "**/*.pkl"), recursive=True)

    if not pkl_files:
        print(f"No pickle files found in {results_folder}")
        return {}

    print(f"Found {len(pkl_files)} pickle file(s):")
    for f in pkl_files:
        print(f"  - {f}")

    # Group data by model type
    models_data = defaultdict(list)

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

                # Extract model identifier
                model_name = extract_model_name(pkl_file)

                # Store all runs from this file
                for hp_idx in data['experiment_data'].keys():
                    hp_data = data['experiment_data'][hp_idx]
                    for run in hp_data['runs']:
                        models_data[model_name].append({
                            'file_path': pkl_file,
                            'hp_idx': hp_idx,
                            'run_data': run,
                            'agent_params': hp_data['agent_hyperparams']
                        })

                print(f"Successfully loaded: {pkl_file} -> Model: {model_name}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    print(f"\nGrouped into {len(models_data)} model(s):")
    for model_name, runs in models_data.items():
        print(f"  {model_name}: {len(runs)} run(s)")

    return models_data


def plot_evaluation_rewards(models_data, save_path="./evaluation_rewards.png",
                           smooth_window=1, show_std=True):
    """
    Plot evaluation rewards averaged across seeds for each model.

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to lists of run data
    save_path : str
        Path to save the plot
    smooth_window : int
        Window size for smoothing (1 = no smoothing)
    show_std : bool
        Whether to show standard error shading
    """
    if not models_data:
        print("No data to plot")
        return

    # Academic paper style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, (model_name, runs) in enumerate(models_data.items()):
        print(f"\nProcessing model: {model_name} with {len(runs)} run(s)")

        # Check if evaluation data exists
        has_eval_data = False
        for run_info in runs:
            run_data = run_info['run_data']
            if len(run_data['eval_episode_rewards']) > 0 and run_data['eval_episode_rewards'].shape[1] > 0:
                has_eval_data = True
                break

        if not has_eval_data:
            print(f"  WARNING: No evaluation data found for {model_name}")
            print(f"  Make sure eval_episodes > 0 in your config")
            continue

        # Collect evaluation data from all seeds
        all_eval_rewards = []
        all_timesteps = []

        for run_info in runs:
            run_data = run_info['run_data']
            eval_rewards = run_data['eval_episode_rewards']
            timesteps_at_eval = run_data['timesteps_at_eval']

            if len(eval_rewards) == 0 or eval_rewards.shape[1] == 0:
                continue

            # Average over evaluation episodes at each timestep
            mean_eval_at_timestep = np.mean(eval_rewards, axis=1)
            all_eval_rewards.append(mean_eval_at_timestep)
            all_timesteps.append(timesteps_at_eval)

        if len(all_eval_rewards) == 0:
            print(f"  No valid evaluation data for {model_name}")
            continue

        # Find common timestep grid
        min_len = min(len(r) for r in all_eval_rewards)
        all_eval_rewards_trimmed = [r[:min_len] for r in all_eval_rewards]
        all_timesteps_trimmed = [s[:min_len] for s in all_timesteps]

        # Convert to numpy arrays
        all_eval_rewards_trimmed = np.array(all_eval_rewards_trimmed)
        all_timesteps_trimmed = np.array(all_timesteps_trimmed)

        # Calculate mean across seeds
        mean_rewards = np.mean(all_eval_rewards_trimmed, axis=0)
        std_rewards = np.std(all_eval_rewards_trimmed, axis=0)
        stderr_rewards = std_rewards / np.sqrt(len(all_eval_rewards_trimmed))
        mean_timesteps = np.mean(all_timesteps_trimmed, axis=0)
        n_runs = len(all_eval_rewards_trimmed)

        # Apply smoothing if requested
        if smooth_window > 1:
            mean_rewards = smooth_data(mean_rewards, smooth_window)
            stderr_rewards = smooth_data(stderr_rewards, smooth_window)
            mean_timesteps = mean_timesteps[smooth_window-1:]

        # Get color for this model
        color = colors[idx % len(colors)]

        # Plot mean line
        label = f"{model_name}" if n_runs == 1 else f"{model_name} (n={n_runs})"
        ax.plot(mean_timesteps, mean_rewards, label=label, color=color,
                linewidth=2.0, marker='o', markersize=3,
                markevery=max(1, len(mean_timesteps)//10))

        # Add shaded error region if multiple runs
        if show_std and n_runs > 1:
            ax.fill_between(mean_timesteps,
                           mean_rewards - stderr_rewards,
                           mean_rewards + stderr_rewards,
                           alpha=0.2, color=color)

        print(f"  Plotted: {len(mean_timesteps)} evaluation points, "
              f"final reward: {mean_rewards[-1]:.2f} ± {stderr_rewards[-1]:.2f}")

    # Academic paper styling
    ax.set_xlabel('Timesteps', fontsize=14)
    ax.set_ylabel('Evaluation Return', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nEvaluation rewards plot saved to: {save_path}")
    plt.close()


def plot_training_rewards(models_data, save_path="./training_rewards.png",
                         smooth_window=1, show_std=True):
    """
    Plot training rewards averaged across seeds for each model.

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to lists of run data
    save_path : str
        Path to save the plot
    smooth_window : int
        Window size for smoothing (1 = no smoothing)
    show_std : bool
        Whether to show standard error shading
    """
    if not models_data:
        print("No data to plot")
        return

    # Academic paper style
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Professional color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for idx, (model_name, runs) in enumerate(models_data.items()):
        print(f"\nProcessing model: {model_name} with {len(runs)} run(s)")

        # Find the maximum timesteps across all runs to create common grid
        max_timesteps = 0
        for run_info in runs:
            run_data = run_info['run_data']
            train_steps = run_data['train_episode_steps']
            cumulative_steps = np.cumsum(train_steps)
            max_timesteps = max(max_timesteps, cumulative_steps[-1])

        print(f"  Max timesteps: {max_timesteps}")

        # Create common timestep grid (every 500 timesteps)
        common_timesteps = np.arange(0, max_timesteps + 1, 500)

        # Interpolate all runs onto common grid
        interpolated_rewards = []

        for run_info in runs:
            run_data = run_info['run_data']
            train_rewards = run_data['train_episode_rewards']
            train_steps = run_data['train_episode_steps']

            # Calculate cumulative timesteps
            cumulative_steps = np.cumsum(train_steps)

            # Interpolate rewards onto common timestep grid
            interp_rewards = np.interp(common_timesteps, cumulative_steps, train_rewards)
            interpolated_rewards.append(interp_rewards)

        if len(interpolated_rewards) == 0:
            print(f"  No training data found for {model_name}")
            continue

        # Convert to numpy array for easier manipulation
        interpolated_rewards = np.array(interpolated_rewards)

        # Calculate mean and standard error across seeds
        mean_rewards = np.mean(interpolated_rewards, axis=0)
        std_rewards = np.std(interpolated_rewards, axis=0)
        stderr_rewards = std_rewards / np.sqrt(len(interpolated_rewards))
        n_runs = len(interpolated_rewards)

        # Apply smoothing if requested
        if smooth_window > 1:
            mean_rewards = smooth_data(mean_rewards, smooth_window)
            stderr_rewards = smooth_data(stderr_rewards, smooth_window)
            timesteps_plot = common_timesteps[smooth_window-1:]
        else:
            timesteps_plot = common_timesteps

        # Get color for this model
        color = colors[idx % len(colors)]

        # Plot mean line
        label = f"{model_name}" if n_runs == 1 else f"{model_name} (n={n_runs})"
        ax.plot(timesteps_plot, mean_rewards, label=label, color=color,
                linewidth=2.0)

        # Add shaded error region if multiple runs
        if show_std and n_runs > 1:
            ax.fill_between(timesteps_plot,
                           mean_rewards - stderr_rewards,
                           mean_rewards + stderr_rewards,
                           alpha=0.2, color=color)

        print(f"  Plotted: {len(timesteps_plot)} points, "
              f"final reward: {mean_rewards[-1]:.2f} ± {stderr_rewards[-1]:.2f}")

    # Academic paper styling
    ax.set_xlabel('Timesteps', fontsize=14)
    ax.set_ylabel('Episode Return', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining rewards plot saved to: {save_path}")
    plt.close()


def print_summary_statistics(models_data):
    """
    Print summary statistics for all models.

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to lists of run data
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for model_name, runs in models_data.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        print(f"Number of seeds/runs: {len(runs)}")

        if len(runs) == 0:
            continue

        # Get environment and agent info from first run
        first_file = runs[0]['file_path']
        try:
            with open(first_file, 'rb') as f:
                data = pickle.load(f)
                env_name = data['experiment']['environment']['env_name']
                agent_name = data['experiment']['agent']['agent_name']
                print(f"Environment: {env_name}")
                print(f"Agent: {agent_name}")
        except:
            pass

        # Collect statistics across all seeds
        final_train_rewards = []
        avg_train_rewards = []
        max_train_rewards = []

        for run_info in runs:
            run_data = run_info['run_data']
            train_rewards = run_data['train_episode_rewards']

            # Last 10% of training
            last_10pct = max(1, len(train_rewards) // 10)
            final_train_rewards.append(np.mean(train_rewards[-last_10pct:]))
            avg_train_rewards.append(np.mean(train_rewards))
            max_train_rewards.append(np.max(train_rewards))

        print(f"\nTraining Performance:")
        if final_train_rewards:
            print(f"  Final Reward (last 10%): {np.mean(final_train_rewards):.2f} ± {np.std(final_train_rewards):.2f}")
            print(f"  Average Reward: {np.mean(avg_train_rewards):.2f} ± {np.std(avg_train_rewards):.2f}")
            print(f"  Max Reward: {np.mean(max_train_rewards):.2f} ± {np.std(max_train_rewards):.2f}")


def main(results_folder="./results", smooth_window=1,
         output_path="./rewards.png", show_std=True, plot_type="train"):
    """
    Main function to load results and create rewards plot.

    Parameters
    ----------
    results_folder : str
        Path to results folder
    smooth_window : int
        Smoothing window size (1 = no smoothing)
    output_path : str
        Path to save the plot
    show_std : bool
        Whether to show standard error shading
    plot_type : str
        Type of plot: 'train', 'eval', or 'both'
    """
    plot_type_name = "TRAINING" if plot_type == "train" else "EVALUATION" if plot_type == "eval" else "TRAINING & EVALUATION"
    print("="*80)
    print(f"LOADING AND PLOTTING {plot_type_name} REWARDS ACROSS SEEDS")
    print("="*80)

    # Find all subdirectories in results folder
    subdirs = [d for d in os.listdir(results_folder)
               if os.path.isdir(os.path.join(results_folder, d))]

    # If no subdirectories, process the results folder directly
    if not subdirs:
        print("No subdirectories found. Processing results folder directly.")
        subdirs = ['.']

    print(f"\nFound {len(subdirs)} folder(s) to process:")
    for subdir in subdirs:
        print(f"  - {subdir}")

    all_plots = []

    # Process each subdirectory separately
    for subdir in subdirs:
        folder_path = os.path.join(results_folder, subdir) if subdir != '.' else results_folder
        folder_name = subdir if subdir != '.' else 'results'

        print("\n" + "="*80)
        print(f"PROCESSING FOLDER: {folder_name}")
        print("="*80)

        # Load data grouped by model for this folder
        models_data = load_results_from_folder(folder_path)

        if not models_data:
            print(f"No data found in {folder_name}. Skipping.")
            continue

        # Print summary statistics
        print_summary_statistics(models_data)

        # Create plots based on type
        print("\n" + "="*80)
        print(f"GENERATING {plot_type_name} REWARDS PLOT FOR {folder_name}")
        print("="*80)

        # Generate output paths with folder name
        base_name = os.path.splitext(output_path)[0]
        ext = os.path.splitext(output_path)[1] if os.path.splitext(output_path)[1] else '.png'

        if plot_type in ["train", "both"]:
            if plot_type == "train":
                train_path = f"{base_name}_{folder_name}{ext}"
            else:
                train_path = f"{base_name}_{folder_name}_train{ext}"

            plot_training_rewards(models_data, save_path=train_path,
                                 smooth_window=smooth_window, show_std=show_std)
            all_plots.append(("Training", folder_name, train_path))

        if plot_type in ["eval", "both"]:
            if plot_type == "eval":
                eval_path = f"{base_name}_{folder_name}{ext}"
            else:
                eval_path = f"{base_name}_{folder_name}_eval{ext}"

            plot_evaluation_rewards(models_data, save_path=eval_path,
                                   smooth_window=smooth_window, show_std=show_std)
            all_plots.append(("Evaluation", folder_name, eval_path))

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nPlot(s) saved:")
    for plot_type_str, folder_name, path in all_plots:
        print(f"  {plot_type_str} ({folder_name}): {path}")
    if smooth_window > 1:
        print(f"Smoothing window: {smooth_window}")
    print(f"Standard error shading: {'enabled' if show_std else 'disabled'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot training and/or evaluation rewards averaged across seeds for different models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot training rewards with error shading
  python plot_results.py

  # Plot evaluation rewards
  python plot_results.py --plot-type eval --output evaluation_rewards.png

  # Plot both training and evaluation rewards
  python plot_results.py --plot-type both --output rewards.png

  # With smoothing
  python plot_results.py --smooth 10

  # Without error shading (just mean lines)
  python plot_results.py --no-std

  # Custom output path with smoothing
  python plot_results.py --output my_plot.png --smooth 20
        """
    )

    parser.add_argument('--results-folder', type=str, default='./results',
                       help='Path to results folder (default: ./results)')
    parser.add_argument('--smooth', type=int, default=1,
                       help='Smoothing window size (default: 1, no smoothing)')
    parser.add_argument('--output', type=str, default='./training_rewards.png',
                       help='Output path for plot (default: ./training_rewards.png)')
    parser.add_argument('--no-std', action='store_true',
                       help='Disable standard error shading')
    parser.add_argument('--plot-type', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Type of plot to generate: train, eval, or both (default: train)')

    args = parser.parse_args()

    main(results_folder=args.results_folder,
         smooth_window=args.smooth,
         output_path=args.output,
         show_std=not args.no_std,
         plot_type=args.plot_type)
