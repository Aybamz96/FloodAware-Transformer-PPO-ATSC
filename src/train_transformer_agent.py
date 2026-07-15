"""
Transformer-based PPO training script for traffic control.

Trains a PPO agent with a Transformer feature extractor on CartPole-v1
as a proof-of-concept before deployment on SUMO traffic simulation.
"""

import os
import json
import gymnasium as gym  # ← CHANGE THIS LINE
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Tuple, Dict, Callable
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from transformer_traffic_control import (
    TransformerFeatureExtractor,
    TransformerPolicy,
    SequentialObsWrapper,
    create_observation_wrapper,
)



def train_transformer_agent(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 50_000,
    seq_length: int = 30,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    n_epochs: int = 10,
    gae_lambda: float = 0.95,
    model_name: str = "transformer_traffic_agent",
) -> Tuple[PPO, Dict]:
    """
    Train a PPO agent with Transformer feature extractor.

    Args:
        env_id: Gymnasium environment ID
        total_timesteps: Total training timesteps
        seq_length: Sequence length for state history
        learning_rate: Initial learning rate
        n_steps: Steps per update
        batch_size: Mini-batch size
        gamma: Discount factor
        n_epochs: Number of PPO epochs
        gae_lambda: GAE lambda
        model_name: Name for saved models

    Returns:
        model: Trained PPO agent
        metadata: Training metadata dictionary
    """
    print("\n" + "="*70)
    print("TRANSFORMER-BASED PPO TRAINING")
    print("="*70)
    print(f"Environment: {env_id}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Sequence length: {seq_length}")
    print(f"PPO config:")
    print(f"  - learning_rate: {learning_rate}")
    print(f"  - n_steps: {n_steps}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - gamma: {gamma}")
    print("="*70)

    # Create environment
    try:
        env = gym.make(env_id)
        print(f"✓ Environment created: {env_id}")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")

        # Wrap with sequential obs wrapper
        env = create_observation_wrapper(env, seq_length=seq_length)
        print(f"✓ Observation wrapper applied")
        print(f"  New observation space: {env.observation_space}")

        # Create dummy env to check feature dimension
        temp_obs, _ = env.reset()
        n_features = temp_obs.shape[1]
        print(f"  Detected features: {n_features}")

    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        raise

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")

    # Create policy kwargs with dynamic n_features
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            d_model=256,
            nhead=8,
            num_layers=2,
            dropout=0.1,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    # Learning rate schedule
    def lr_schedule(progress: float) -> float:
        return learning_rate * (1 - progress)

    try:
        print("="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")

        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )

        # Callbacks
        os.makedirs("./models/checkpoints", exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path="./models/checkpoints",
            name_prefix="transformer_agent",
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path="./models/best",
            log_path="./models/logs",
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=False,
            render=False,
        )

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        print("\n✓ Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Save model
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/{model_name}_final"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}.zip")

    # Metadata
    metadata = {
        "model_name": model_name,
        "environment": env_id,
        "total_timesteps": total_timesteps,
        "seq_length": seq_length,
        "n_features": n_features,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "device": device,
        "timestamp": str(datetime.now()),
    }

    metadata_path = f"./models/{model_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")

    return model, metadata


if __name__ == "__main__":
    # Train the agent
    model, metadata = train_transformer_agent(
        env_id="CartPole-v1",
        total_timesteps=50_000,
        seq_length=30,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        n_epochs=10,
        gae_lambda=0.95,
        model_name="transformer_traffic_agent",
    )

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for key, value in metadata.items():
        print(f"{key:20s}: {value}")
    print("="*70)
