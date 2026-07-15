"""
Transformer-based traffic control using PyTorch and Gymnasium.

Implements:
1. TransformerFeatureExtractor: Attention-based feature extraction
2. TransformerPolicy: Custom policy with transformer backbone
3. SequentialObsWrapper: Buffers observations into sequences
4. Helper functions for training and evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Dict, Any, Optional, Type

# ✅ GYMNASIUM IMPORTS
import gymnasium
from gymnasium import spaces
from gymnasium.core import Wrapper

# ✅ STABLE BASELINES3 IMPORTS
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule  # ✅ ADD THIS!

from stable_baselines3.common.distributions import Distribution, CategoricalDistribution, DiagGaussianDistribution
import torch.optim as optim


# ============================================================================
# TRANSFORMER FEATURE EXTRACTOR
# ============================================================================

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for sequential observations.

    Expects input shape: (seq_length, n_features)
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        """
        Initialize Transformer feature extractor.

        Args:
            observation_space: Gymnasium Box space with shape (seq_length, n_features)
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        # ✅ FIX: Use len(shape) instead of ndim
        assert len(observation_space.shape) == 2, \
            f"Expected 2D observation space, got shape {observation_space.shape}"

        seq_length, n_features = observation_space.shape
        features_dim = d_model  # Output dimension of transformer

        super().__init__(observation_space, features_dim)

        print(f"  [TransformerFeatureExtractor]")
        print(f"    Input shape: ({seq_length}, {n_features})")
        print(f"    d_model: {d_model}, nhead: {nhead}, num_layers: {num_layers}")

        # Input projection: project from n_features → d_model
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(seq_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Mean pooling → output features_dim
        self.output_projection = nn.Linear(d_model, self.features_dim)

    def _create_positional_encoding(
            self,
            seq_length: int,
            d_model: int,
    ) -> torch.nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            observations: Shape (batch_size, seq_length, n_features)

        Returns:
            features: Shape (batch_size, features_dim)
        """
        # Project input: (batch, seq_len, n_features) → (batch, seq_len, d_model)
        x = self.input_projection(observations)

        # Add positional encoding
        x = x + self.positional_encoding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Mean pooling: (batch, seq_len, d_model) → (batch, d_model)
        x = x.mean(dim=1)

        # Project to output dimension
        x = self.output_projection(x)

        return x


# ============================================================================
# TRANSFORMER POLICY
# ============================================================================

class TransformerPolicy(ActorCriticPolicy):
    """
    Custom policy using Transformer feature extractor.
    """

    def __init__(
            self,
            observation_space: gymnasium.Space,  # ✅ CHANGE: gym → gymnasium
            action_space: gymnasium.Space,  # ✅ CHANGE: gym → gymnasium
            lr_schedule: Schedule,
            net_arch: Optional[Dict[str, list]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 2,
            dropout: float = 0.1,
            **kwargs,
    ):
        """
        Initialize TransformerPolicy.

        Args:
            observation_space: Observation space (gymnasium.Space)
            action_space: Action space (gymnasium.Space)
            lr_schedule: Learning rate schedule
            net_arch: Network architecture dict
            activation_fn: Activation function
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        Args:
            lr_schedule: Learning rate schedule
        """
        self.features_extractor = TransformerFeatureExtractor(
            self.observation_space,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        features_dim = self.features_extractor.features_dim

        # Policy and value networks
        self.mlp_extractor = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Actor head
        self.action_net = nn.Linear(256, self.action_space.n)

        # Critic head
        self.value_net = nn.Linear(256, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr_schedule(1),
        )

    def forward(
            self,
            obs: torch.Tensor,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy.

        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions

        Returns:
            actions, values
        """
        features = self.features_extractor(obs)
        mlp_out = self.mlp_extractor(features)

        logits = self.action_net(mlp_out)
        values = self.value_net(mlp_out)

        return logits, values

    def _predict(
            self,
            observation: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get predicted actions.

        Args:
            observation: Observations
            deterministic: Whether to use deterministic actions

        Returns:
            actions
        """
        logits, _ = self.forward(observation, deterministic)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return actions


# ============================================================================
# SEQUENTIAL OBSERVATION WRAPPER
# ============================================================================

class SequentialObsWrapper(gymnasium.Wrapper):
    """
    Wraps observations into sequential format (seq_length, n_features).
    """

    def __init__(self, env: gymnasium.Env, seq_length: int = 30):
        super().__init__(env)

        self.seq_length = seq_length

        # ✅ FIX: Use .shape instead of .ndim
        original_shape = self.env.observation_space.shape
        n_features = original_shape[0] if len(original_shape) > 0 else 1

        self.observation_history = deque(maxlen=seq_length)

        # New observation space: (seq_length, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_length, n_features),
            dtype=np.float32,
        )

        print(f"  [SequentialObsWrapper]")
        print(f"    Original obs shape: {original_shape}")
        print(f"    New obs shape: (seq_length={seq_length}, n_features={n_features})")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # ✅ FIX: Ensure obs is 1D
        if np.isscalar(obs):
            obs = np.array([obs], dtype=np.float32)
        elif obs.ndim == 0:
            obs = np.array([obs.item()], dtype=np.float32)

        # Initialize history with first observation
        self.observation_history.clear()
        for _ in range(self.seq_length):
            self.observation_history.append(obs)

        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # ✅ FIX: Ensure obs is 1D
        if np.isscalar(obs):
            obs = np.array([obs], dtype=np.float32)
        elif obs.ndim == 0:
            obs = np.array([obs.item()], dtype=np.float32)

        self.observation_history.append(obs)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Convert history deque to (seq_length, n_features) array."""
        seq = np.array(list(self.observation_history), dtype=np.float32)
        return seq


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_observation_wrapper(env, seq_length: int = 30):
    """
    Wrap environment to provide sequential observations.

    Automatically detects number of features from the environment.

    Args:
        env: Base gymnasium environment
        seq_length: Sequence length (default: 30)

    Returns:
        wrapped_env: Environment that outputs (seq_length, n_features) observations
    """
    return SequentialObsWrapper(env, seq_length=seq_length)


# ============================================================================
# TRANSFORMER TRAFFIC CONTROLLER
# ============================================================================

class TransformerTrafficController:
    """
    Traffic control agent using Transformer-based policy.

    Manages real-time state conversion, history buffer maintenance,
    and policy inference for SUMO traffic simulation.

    Args:
        model: Trained PPO model with TransformerPolicy
        seq_length: Sequence length for state history (default: 30)
        n_features: Number of state features (default: 12)
        device: Torch device (default: 'cuda' if available else 'cpu')
    """

    def __init__(
        self,
        model,
        seq_length: int = 30,
        n_features: int = 12,
        device: str = None,
    ):
        self.model = model
        self.seq_length = seq_length
        self.n_features = n_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Circular history buffer for state features
        self.state_history = deque(maxlen=seq_length)

        # Initialize with zeros
        for _ in range(seq_length):
            self.state_history.append(np.zeros(n_features, dtype=np.float32))

    def update_state(self, features: np.ndarray) -> None:
        """
        Update state history with new observation.

        Args:
            features: Array of shape (n_features,)
        """
        assert features.shape == (self.n_features,), \
            f"Expected features shape ({self.n_features},), got {features.shape}"
        self.state_history.append(features.astype(np.float32))

    def get_action(self, deterministic: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Get action from policy based on current state history.

        Args:
            deterministic: Whether to use deterministic policy

        Returns:
            action: Discrete action (0 or 1 for CartPole)
            value: State value estimate
        """
        # Convert state history to tensor
        state_array = np.array(list(self.state_history), dtype=np.float32)
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)

        # Get action from policy
        with torch.no_grad():
            action, value = self.model.policy.forward(state_tensor, deterministic=deterministic)

        return action.cpu().numpy()[0], value.cpu().numpy()[0]

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """
        Extract attention weights from Transformer feature extractor.

        Returns:
            attention_weights: Numpy array of attention weights or None
        """
        try:
            # Access the feature extractor's transformer encoder
            feature_extractor = self.model.policy.features_extractor

            if not hasattr(feature_extractor, 'transformer_encoder'):
                return None

            # This is a simplified extraction; full attention visualization
            # would require hooks or custom forward passes
            return None

        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")
            return None

    def reset_history(self) -> None:
        """Reset state history to zeros."""
        self.state_history.clear()
        for _ in range(self.seq_length):
            self.state_history.append(np.zeros(self.n_features, dtype=np.float32))
