"""Standalone velocity policy loader — no mjlab dependency.

Loads the actor MLP directly from the checkpoint weights.
Network architecture: Linear(48->512)->ELU->Linear(512->256)->ELU->
                      Linear(256->128)->ELU->Linear(128->12)

No obs normalizer (obs_normalization=False in rumi_velocity rl_cfg).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VelocityPolicy:
    """Wraps actor MLP for inference. No obs normalization.

    Usage:
        policy = VelocityPolicy("checkpoint/latest_velocity.pt")
        raw_action = policy(obs_np)   # obs_np: np.ndarray [48]
        # raw_action: np.ndarray [12], unscaled policy output
    """

    def __init__(self, checkpoint_path: str | Path):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state = ckpt["actor_state_dict"]

        self._mlp = _MLP()

        mlp_state = {
            k.replace("mlp.", ""): v
            for k, v in state.items()
            if k.startswith("mlp.")
        }
        self._mlp.net.load_state_dict(mlp_state)
        self._mlp.eval()

        iter_n = ckpt.get("iter", "?")
        print(f"[Policy] Loaded checkpoint: {Path(checkpoint_path).name}  (iter {iter_n})")

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass.

        Args:
            obs: float32 array of shape [48].

        Returns:
            raw_action: float32 array of shape [12], unscaled policy output.
                        Multiply by 0.075 to get motor position offsets in rad.
        """
        x = torch.from_numpy(obs).float().unsqueeze(0)   # [1, 48]
        x = self._mlp(x)
        return x.squeeze(0).numpy()                       # [12]
