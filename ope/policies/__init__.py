"""Frozen, position-aware contextual bandit policies for Open Bandit logs."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit

__all__ = ["LinUCB", "LogisticTS"]


def _with_intercept(context: np.ndarray, dtype: type) -> np.ndarray:
    design = np.empty((len(context), context.shape[1] + 1), dtype=dtype)
    design[:, :-1] = context
    design[:, -1] = 1
    return design


def _training_groups(feedback: dict):
    """Visit each (position, action) once, including pairs with no exposures."""
    action, position, reward = (feedback[k] for k in ("action", "position", "reward"))
    n_actions, len_list = feedback["n_actions"], feedback["len_list"]
    context = feedback["context"]
    if (
        context.ndim != 2 or not len(context)
        or action.shape != position.shape or action.shape != reward.shape
        or action.shape != (len(context),)
        or not np.issubdtype(action.dtype, np.integer)
        or not np.issubdtype(position.dtype, np.integer)
        or np.any((action < 0) | (action >= n_actions))
        or np.any((position < 0) | (position >= len_list))
        or not np.isin(reward, [0, 1]).all()
    ):
        raise ValueError("Expected aligned contexts, valid actions/positions and binary rewards")

    ids = position * n_actions + action
    ordered = np.argsort(ids, kind="stable")
    ends = np.r_[0, np.bincount(ids, minlength=len_list * n_actions).cumsum()]
    for group in range(len(ends) - 1):
        rows = ordered[ends[group]:ends[group + 1]]
        x = _with_intercept(context[rows], np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Training contexts must be finite")
        yield (*divmod(group, n_actions), x, reward[rows])


class LinUCB:
    """Disjoint ridge regression with a separate confidence bonus for each slot/item."""

    def __init__(self, *, alpha: float = 0.02, ridge: float = 1.0):
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("LinUCB requires a nonnegative alpha and a positive ridge")
        self.alpha = alpha
        self.ridge = ridge

    def fit(self, feedback: dict) -> LinUCB:
        slots, actions = feedback["len_list"], feedback["n_actions"]
        d = feedback["context"].shape[1] + 1
        identity = np.eye(d)
        weights = np.empty((slots, actions, d), dtype=np.float32)
        inverse_roots = np.empty((slots, actions, d, d), dtype=np.float32)
        for p, a, x, reward in _training_groups(feedback):
            chol = np.linalg.cholesky(x.T @ x + self.ridge * identity)
            weights[p, a] = cho_solve((chol, True), x.T @ reward)
            # If A = LL^T, the variance is ||L^-1 x||^2.
            inverse_roots[p, a] = solve_triangular(chol, identity, lower=True).T
        self.weights_ = weights
        self.inverse_roots_ = inverse_roots
        return self

    def predict_action_dist(self, context: np.ndarray) -> np.ndarray:
        """Return (rows, actions, slots) probabilities for a batch of contexts."""
        x = _with_intercept(context, np.float32)
        slots, actions, _ = self.weights_.shape
        probabilities = np.zeros((len(x), actions, slots), dtype=np.float32)
        rows = np.arange(len(x))
        for p in range(slots):
            scores = x @ self.weights_[p].T
            if self.alpha:
                for a in range(actions):
                    projected = x @ self.inverse_roots_[p, a]
                    scores[:, a] += self.alpha * np.sqrt(
                        np.einsum("ij,ij->i", projected, projected)
                    )
            probabilities[rows, scores.argmax(axis=1), p] = 1
        return probabilities


def _logistic_posterior(
    x: np.ndarray, reward: np.ndarray, prior: np.ndarray, precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the MAP and the Cholesky factor of its posterior precision."""
    regularizer = precision * np.eye(len(prior))
    if not len(x):
        return prior, np.linalg.cholesky(regularizer)

    def objective(theta):
        logits = x @ theta
        delta = theta - prior
        loss = np.logaddexp(0, logits) - reward * logits
        gradient = x.T @ (expit(logits) - reward) + precision * delta
        return loss.sum() + 0.5 * precision * (delta @ delta), gradient

    fit = minimize(
        objective, prior, method="L-BFGS-B", jac=True,
        options={"maxiter": 1000, "gtol": 1e-6, "ftol": 1e-12},
    )
    if not fit.success:
        raise RuntimeError(f"Logistic MAP fit failed: {fit.message}")
    logits = x @ fit.x
    curvature = expit(logits) * expit(-logits)
    hessian = x.T @ (curvature[:, None] * x) + regularizer
    return fit.x, np.linalg.cholesky(hessian)


class LogisticTS:
    """Logistic Thompson sampling with a frozen bank of Laplace posterior draws."""

    def __init__(self, *, prior_precision: float = 1.0, n_draws: int = 64, seed: int = 12345):
        if not np.isfinite(prior_precision) or prior_precision <= 0 or n_draws <= 0 or seed < 0:
            raise ValueError("LogisticTS requires positive prior precision/draw count and a nonnegative seed")
        self.prior_precision = prior_precision
        self.n_draws = n_draws
        self.seed = seed

    def fit(self, feedback: dict) -> LogisticTS:
        slots, actions = feedback["len_list"], feedback["n_actions"]
        d = feedback["context"].shape[1] + 1
        reward, position = feedback["reward"], feedback["position"]
        rng = np.random.default_rng(self.seed)
        draws = np.empty((slots, d, self.n_draws, actions), dtype=np.float32)
        priors = np.zeros((slots, d))
        for p in range(slots):
            observed = reward[position == p]
            clicks = observed.sum()
            # Center the intercept on the smoothed training CTR for this slot.
            priors[p, -1] = np.log((clicks + 0.5) / (len(observed) - clicks + 0.5))
        for p, a, x, r in _training_groups(feedback):
            mean, chol = _logistic_posterior(x, r, priors[p], self.prior_precision)
            noise = rng.standard_normal((d, self.n_draws))
            draws[p, :, :, a] = mean[:, None] + solve_triangular(chol.T, noise)
        self.draws_ = draws
        return self

    def predict_action_dist(self, context: np.ndarray) -> np.ndarray:
        """Estimate each action's probability by its share of posterior-draw wins."""
        x = _with_intercept(context, np.float32)
        slots, d, draws, actions = self.draws_.shape
        probabilities = np.empty((len(x), actions, slots), dtype=np.float32)
        row_offsets = (actions * np.arange(len(x)))[:, None]
        for p in range(slots):
            # The sigmoid is increasing, so a logit argmax is also a CTR argmax.
            scores = (x @ self.draws_[p].reshape(d, draws * actions)).reshape(-1, draws, actions)
            winners = scores.argmax(axis=2)
            counts = np.bincount((row_offsets + winners).ravel(), minlength=len(x) * actions)
            probabilities[:, :, p] = counts.reshape(-1, actions) / draws
        return probabilities
