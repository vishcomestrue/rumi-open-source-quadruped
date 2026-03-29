"""Standalone policy loader — no mjlab dependency.

Loads the actor MLP and obs normalizer directly from the checkpoint weights.
Network architecture: Linear(41->512)->ELU->Linear(512->256)->ELU->
                      Linear(256->128)->ELU->Linear(128->12)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class _ObsNormalizer(nn.Module):
    """Running-mean/std normalizer, loaded from checkpoint."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.register_buffer("_mean",  torch.zeros(1, obs_dim))
        self.register_buffer("_std",   torch.ones(1,  obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(41, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GetupPolicy:
    """Wraps obs normalizer + MLP actor for inference.

    Usage:
        policy = GetupPolicy("checkpoint/latest_getup.pt")
        raw_action = policy(obs_np)   # obs_np: np.ndarray [41]
        # raw_action: np.ndarray [12], unscaled policy output
    """

    def __init__(self, checkpoint_path: str | Path):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt["actor_state_dict"]

        self._normalizer = _ObsNormalizer(obs_dim=41)
        self._mlp = _MLP()

        # Load normalizer buffers
        self._normalizer._mean.copy_(state["obs_normalizer._mean"])
        self._normalizer._std.copy_(state["obs_normalizer._std"])

        # Load MLP weights  (checkpoint keys: mlp.0.weight, mlp.0.bias, ...)
        mlp_state = {
            k.replace("mlp.", ""): v
            for k, v in state.items()
            if k.startswith("mlp.")
        }
        self._mlp.net.load_state_dict(mlp_state)

        self._normalizer.eval()
        self._mlp.eval()

        iter_n = ckpt.get("iter", "?")
        print(f"[Policy] Loaded checkpoint: {Path(checkpoint_path).name}  (iter {iter_n})")

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Args:
            obs: float32 array of shape [41].

        Returns:
            raw_action: float32 array of shape [12], unscaled policy output.
                        Multiply by 0.075 to get motor position offsets in rad.
        """
        x = torch.from_numpy(obs).float().unsqueeze(0)   # [1, 41]
        x = self._normalizer(x)
        x = self._mlp(x)
        return x.squeeze(0).numpy()                       # [12]
