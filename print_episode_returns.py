#!/usr/bin/env python3
"""
Script to read a result file and print each episode return.
"""

import pickle
import sys
import numpy as np


def print_episode_returns(file_path, return_type='train'):
    """
    Read a result file and print each episode return.

    Parameters
    ----------
    file_path : str
        Path to the pickle result file
    return_type : str
        Type of returns to print: 'train', 'eval', or 'both'
    """
    try:
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loading results from: {file_path}")
        print("=" * 80)

        # Extract experiment data
        experiment_data = data['experiment_data']

        # Iterate through all hyperparameter configurations
        for hp_idx in experiment_data.keys():
            hp_data = experiment_data[hp_idx]

            print(f"\nHyperparameter Configuration #{hp_idx}")
            print("-" * 80)

            # Print agent hyperparameters
            if 'agent_hyperparams' in hp_data:
                print("Agent Hyperparameters:")
                for key, value in hp_data['agent_hyperparams'].items():
                    print(f"  {key}: {value}")
                print()

            # Iterate through all runs
            for run_idx, run_data in enumerate(hp_data['runs']):
                print(f"\n{'='*80}")
                print(f"Run #{run_data.get('run_number', run_idx + 1)}")
                print(f"Random Seed: {run_data.get('random_seed', 'N/A')}")
                print(f"Total Timesteps: {run_data.get('total_timesteps', 'N/A')}")
                print(f"Total Train Episodes: {run_data.get('total_train_episodes', 'N/A')}")
                print(f"{'='*80}")

                # Print training episode returns
                if return_type in ['train', 'both']:
                    train_rewards = run_data.get('train_episode_rewards', [])
                    if len(train_rewards) > 0:
                        print(f"\nTraining Episode Returns ({len(train_rewards)} episodes):")
                        print("-" * 80)
                        for ep_idx, reward in enumerate(train_rewards, 1):
                            print(f"Episode {ep_idx:4d}: {reward:10.2f}")

                        print(f"\nTraining Statistics:")
                        print(f"  Mean:   {np.mean(train_rewards):10.2f}")
                        print(f"  Std:    {np.std(train_rewards):10.2f}")
                        print(f"  Min:    {np.min(train_rewards):10.2f}")
                        print(f"  Max:    {np.max(train_rewards):10.2f}")
                        print(f"  Median: {np.median(train_rewards):10.2f}")
                    else:
                        print("\nNo training episode returns found.")

                # Print evaluation episode returns
                if return_type in ['eval', 'both']:
                    eval_rewards = run_data.get('eval_episode_rewards', [])
                    if len(eval_rewards) > 0 and eval_rewards.shape[1] > 0:
                        print(f"\nEvaluation Episode Returns:")
                        print(f"Shape: {eval_rewards.shape} (eval_points x episodes_per_eval)")
                        print("-" * 80)
                        timesteps_at_eval = run_data.get('timesteps_at_eval', list(range(len(eval_rewards))))

                        for eval_idx, (timestep, rewards) in enumerate(zip(timesteps_at_eval, eval_rewards)):
                            mean_reward = np.mean(rewards)
                            std_reward = np.std(rewards)
                            print(f"Eval #{eval_idx + 1:3d} @ Timestep {timestep:6d}: "
                                  f"Mean={mean_reward:8.2f}, Std={std_reward:6.2f}, "
                                  f"Returns={rewards}")

                        # Overall evaluation statistics
                        all_eval_returns = eval_rewards.flatten()
                        print(f"\nOverall Evaluation Statistics:")
                        print(f"  Mean:   {np.mean(all_eval_returns):10.2f}")
                        print(f"  Std:    {np.std(all_eval_returns):10.2f}")
                        print(f"  Min:    {np.min(all_eval_returns):10.2f}")
                        print(f"  Max:    {np.max(all_eval_returns):10.2f}")
                        print(f"  Median: {np.median(all_eval_returns):10.2f}")
                    else:
                        print("\nNo evaluation episode returns found.")

        print("\n" + "=" * 80)
        print("Done!")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Print episode returns from a result file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print training episode returns
  python print_episode_returns.py results/Acrobot-v1_GreedyACresults/Acrobot-v1_GreedyAC_expectile_True0.8_data_1.pkl

  # Print evaluation episode returns
  python print_episode_returns.py results/file.pkl --type eval

  # Print both training and evaluation returns
  python print_episode_returns.py results/file.pkl --type both
        """
    )

    parser.add_argument('file_path', type=str,
                       help='Path to the pickle result file')
    parser.add_argument('--type', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Type of returns to print: train, eval, or both (default: train)')

    args = parser.parse_args()

    print_episode_returns(args.file_path, return_type=args.type)
