"""Success Guided Sampling (SGS) over a fixed set of task configurations.

Implements the reset sampler of Zhang et al., "A Balanced Data Diet: Addressing
the Exploration Bottleneck in Mega-Scale RL for Robot Control" (arXiv
2610.12465), Sec. 3.2 and App. B.2:

* Before training, a fixed set of ``N`` task configurations is drawn from the
  task distribution; every episode reset picks one of them.
* For each configuration the last ``H`` Boolean episode outcomes are kept in a
  circular buffer and its success rate ``p_i`` is their mean.
* Each configuration is scored with a Beta-shaped kernel in mode-concentration
  form, ``Beta(1 + kappa*t, 1 + kappa*(1 - t))`` (Eq. 1)::

      w_i = (p_i + eps)^(kappa*t) * (1 - p_i + eps)^(kappa*(1 - t))
      w~_i = max(w_i, eps)            l_i = log(w~_i + eps)

  and the next configuration is drawn i.i.d. from ``softmax(l / T_eff)`` with
  ``T_eff = max(T, 1)`` (Eq. 2). ``eps`` keeps every configuration's
  probability non-zero so unsolved ones keep getting re-estimated.

Defaults are the paper's manipulation setting (``t=0.5, kappa=1, eps=1e-4,
T=2, H=100``). The paper does not say how a configuration with fewer than
``H`` (or zero) recorded outcomes is scored; here ``p_i`` is the mean over the
outcomes recorded so far, and 0 for a never-visited configuration, so sampling
starts uniform (the paper's Fig. 7 notes early-training sampling is uniform).

``mode="uniform"`` is the baseline: configurations are drawn uniformly from the
same fixed set (the paper's uniform sampling draws from the reset buffer), while
the same outcome history is still recorded for logging.
"""

from __future__ import annotations

import numpy as np

SGS = "sgs"
UNIFORM = "uniform"
MODES = (SGS, UNIFORM)


class SuccessGuidedSampler:
    """Adaptive (or uniform) sampler over ``num_configs`` fixed configurations."""

    def __init__(
        self,
        num_configs: int,
        *,
        mode: str = SGS,
        history: int = 100,
        target: float = 0.5,
        kappa: float = 1.0,
        temperature: float = 2.0,
        eps: float = 1e-4,
        seed: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.num_configs = int(num_configs)
        self.mode = mode
        self.history = int(history)
        self.target = float(target)
        self.kappa = float(kappa)
        self.temperature = float(temperature)
        self.eps = float(eps)
        self._rng = np.random.default_rng(seed)
        self._outcomes = np.zeros((self.num_configs, self.history), dtype=bool)
        self._count = np.zeros(self.num_configs, dtype=np.int64)  # total outcomes recorded

    # ------------------------------------------------------------------

    def success_rates(self) -> np.ndarray:
        """Rolling success estimate ``p_i`` (mean of the last <= H outcomes)."""
        filled = np.minimum(self._count, self.history)
        wins = self._outcomes.sum(axis=1)
        return np.where(filled > 0, wins / np.maximum(filled, 1), 0.0)

    def logits(self) -> np.ndarray:
        p = self.success_rates()
        a = self.kappa * self.target
        b = self.kappa * (1.0 - self.target)
        w = (p + self.eps) ** a * (1.0 - p + self.eps) ** b
        w = np.maximum(w, self.eps)
        return np.log(w + self.eps)

    def probabilities(self) -> np.ndarray:
        if self.mode == UNIFORM:
            return np.full(self.num_configs, 1.0 / self.num_configs)
        z = self.logits() / max(self.temperature, 1.0)
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, n: int) -> np.ndarray:
        """Draw ``n`` configuration indices i.i.d. from the current distribution."""
        if self.mode == UNIFORM:
            return self._rng.integers(0, self.num_configs, size=n)
        return self._rng.choice(self.num_configs, size=n, p=self.probabilities())

    def update(self, idx: np.ndarray, success: np.ndarray) -> None:
        """Record one Boolean outcome per finished episode (``idx`` may repeat)."""
        for i, s in zip(np.asarray(idx, dtype=np.int64), np.asarray(success, dtype=bool)):
            self._outcomes[i, self._count[i] % self.history] = s
            self._count[i] += 1

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Scalar summaries for logging."""
        p = self.success_rates()
        visited = self._count > 0
        probs = self.probabilities()
        frontier = visited & (p > 0.0) & (p < 1.0)
        return {
            "visited_frac": float(visited.mean()),
            # Mean rolling success over visited configurations ~ success under
            # uniform sampling of the fixed set (biased to recent history).
            "p_mean_visited": float(p[visited].mean()) if visited.any() else 0.0,
            "frontier_frac": float(frontier.mean()),
            "solved_frac": float((visited & (p >= 0.999)).mean()),
            # Normalized effective sample size of the sampling distribution
            # (1 = uniform, -> 0 = concentrated on a few configurations).
            "ess_frac": float(1.0 / np.sum(probs**2) / self.num_configs),
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "outcomes": self._outcomes.copy(),
            "count": self._count.copy(),
            "p": self.success_rates(),
            "probs": self.probabilities(),
        }
