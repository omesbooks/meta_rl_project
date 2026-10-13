"""Hybrid PPO policy: a discrete trading action + a continuous stop-loss distance.

Action encoding (Box, shape (2,), float32):
    a[0] = discrete action index (0..n_actions-1, stored as a float)
    a[1] = stop-loss distance in ATR units (Gaussian sample; the env clips it to sl_range)

The two heads share the actor trunk and are treated as independent, so
    log pi(a|s) = log Cat(a[0] | logits(s)) + log N(a[1] | mu(s), sigma)
    entropy      = H(Cat) + sl_ent_scale * H(N)
sigma is one state-independent parameter (SB3's DiagGaussian convention,
initialised from log_std_init). Deterministic mode = (argmax, mu).

Why subclass DiagGaussianDistribution: ActorCriticPolicy._build picks the head
builder by isinstance, so a DiagGaussian subclass slots into SB3 unchanged -
only `proba_distribution_net` / `proba_distribution` and the sampling / log-prob
methods are overridden. PPO's rollout clips Box actions to the bounds before
env.step but stores and evaluates the unclipped sample, which keeps log-probs
consistent between collection and the update (same as any SB3 Box policy).
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.distributions import Categorical, Normal


def hybrid_action_space(n_actions: int, sl_range=(0.5, 3.0)) -> spaces.Box:
    lo, hi = float(sl_range[0]), float(sl_range[1])
    return spaces.Box(low=np.array([0.0, lo], dtype=np.float32),
                      high=np.array([float(n_actions - 1), hi], dtype=np.float32), dtype=np.float32)


def n_discrete_actions(space) -> int:
    """Number of discrete trading actions for a Discrete space or a hybrid Box."""
    if hasattr(space, "n"):
        return int(space.n)
    if isinstance(space, spaces.Box) and space.shape == (2,):
        return int(round(float(space.high[0]))) + 1
    raise ValueError(f"unsupported action space {space}")


def is_hybrid_space(space) -> bool:
    return isinstance(space, spaces.Box) and space.shape == (2,)


def split_action(action, n_actions: int, sl_range):
    """(index, sl_atr) from a hybrid action array; robust to float indices and clipping."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    idx = int(np.clip(np.rint(a[0]), 0, n_actions - 1))
    sl = float(np.clip(a[1], sl_range[0], sl_range[1]))
    return idx, sl


class HybridDistribution(DiagGaussianDistribution):
    def __init__(self, n_actions: int, sl_ent_scale: float = 1.0):
        super().__init__(action_dim=2)
        self.n_actions = int(n_actions)
        self.sl_ent_scale = float(sl_ent_scale)
        self.cat: Categorical | None = None
        self.gauss: Normal | None = None

    # ---- construction (called by ActorCriticPolicy._build) ----
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        action_net = nn.Linear(latent_dim, self.n_actions + 1)      # k logits + 1 mean
        log_std = nn.Parameter(th.ones(1) * log_std_init, requires_grad=True)
        return action_net, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor):
        logits = mean_actions[..., : self.n_actions]
        mu = mean_actions[..., self.n_actions:]
        self.cat = Categorical(logits=logits)
        self.gauss = Normal(mu, th.ones_like(mu) * log_std.exp())
        self.distribution = self.gauss                              # keeps SB3 helpers that touch .distribution sane
        return self

    # ---- probabilities ----
    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        idx = actions[..., 0].round().long().clamp(0, self.n_actions - 1)
        sl = actions[..., 1:2]
        return self.cat.log_prob(idx) + self.gauss.log_prob(sl).sum(dim=-1)

    def entropy(self) -> th.Tensor:
        return self.cat.entropy() + self.sl_ent_scale * self.gauss.entropy().sum(dim=-1)

    def sample(self) -> th.Tensor:
        idx = self.cat.sample().to(th.float32).unsqueeze(-1)
        sl = self.gauss.rsample()
        return th.cat([idx, sl], dim=-1)

    def mode(self) -> th.Tensor:
        idx = self.cat.probs.argmax(dim=-1).to(th.float32).unsqueeze(-1)
        return th.cat([idx, self.gauss.mean], dim=-1)

    def actions_from_params(self, mean_actions, log_std, deterministic: bool = False):
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions, log_std):
        actions = self.actions_from_params(mean_actions, log_std)
        return actions, self.log_prob(actions)

    # convenience for backtests / exports
    def action_probs(self) -> th.Tensor:
        return self.cat.probs

    def sl_mean(self) -> th.Tensor:
        return self.gauss.mean[..., 0]


class HybridPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose action head is a HybridDistribution.

    policy_kwargs: n_actions, sl_range=(lo, hi), sl_ent_scale, log_std_init (+ the usual net_arch, ...).
    """

    def __init__(self, *args, n_actions: int = 4, sl_range=(0.5, 3.0), sl_ent_scale: float = 1.0, **kwargs):
        self.n_actions = int(n_actions)
        self.sl_range = (float(sl_range[0]), float(sl_range[1]))
        self.sl_ent_scale = float(sl_ent_scale)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        # swap the default DiagGaussian(2) for the hybrid head before the networks are built
        self.action_dist = HybridDistribution(self.n_actions, self.sl_ent_scale)
        super()._build(lr_schedule)
        # start the SL mean in the middle of the range: SB3's ortho init zeroes the bias, and a
        # mean outside [lo, hi] gets clipped by the env, which hides the head's effect from PPO
        with th.no_grad():
            self.action_net.bias[-1] = 0.5 * (self.sl_range[0] + self.sl_range[1])

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(n_actions=self.n_actions, sl_range=self.sl_range, sl_ent_scale=self.sl_ent_scale)
        return data

    @th.no_grad()
    def heads(self, obs: th.Tensor):
        """(action probs, sl mean in ATR, value) for a batch of observations - deterministic read-out."""
        obs = obs if isinstance(obs, th.Tensor) else th.as_tensor(obs, dtype=th.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features[0]); latent_vf = self.mlp_extractor.forward_critic(features[1])
        dist = self._get_action_dist_from_latent(latent_pi)
        sl = dist.sl_mean().clamp(self.sl_range[0], self.sl_range[1])
        return dist.action_probs(), sl, self.value_net(latent_vf).squeeze(-1)
