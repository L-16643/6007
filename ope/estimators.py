"""Position-aware DM / IPS / SNIPS / DR estimators for OBP-style feedback.

The logged data are assumed to contain one action and one recommendation slot per
round.  For round i, let p_i be the behavior propensity and let

    w_i = pi_e(a_i | x_i, l_i) / p_i

be the importance weight.  With q_hat(x, a, l) the predicted click probability,
the implemented estimators are

    DM    = mean_i sum_a pi_e(a | x_i, l_i) * q_hat(x_i, a, l_i)
    IPS   = mean_i w_i * r_i
    SNIPS = sum_i w_i * r_i / sum_i w_i
    DR    = mean_i [sum_a pi_e(a | x_i, l_i) * q_hat(x_i, a, l_i)
                    + w_i * (r_i - q_hat(x_i, a_i, l_i))].

Clipping, when requested, is applied to the importance weight before any of the
IPS / SNIPS / DR calculations.
"""
from __future__ import annotations

import numpy as np


def evaluate(feedback, action_dist, predicted_reward, *, clip=None):
    """Estimate all four policy values for the same logged feedback.

    Parameters
    ----------
    feedback : dict
        OBP-style dictionary.  Required keys are reward, action, position,
        pscore, n_actions and len_list.
    action_dist : ndarray
        Target-policy probabilities with shape (n_rounds, n_actions, len_list),
        or (n_rounds, n_actions) for a position-independent policy.
    predicted_reward : ndarray
        Out-of-fold q_hat with shape (n_rounds, n_actions, len_list).
    clip : float, optional
        Upper bound for importance weights.  None means no clipping.

    Returns
    -------
    dict
        Dictionary with keys DM, IPS, SNIPS and DR.
    """
    reward, action, position, pscore = (
        np.asarray(feedback[key]) for key in ("reward", "action", "position", "pscore")
    )
    n_rounds = reward.size
    if not n_rounds or any(
        arr.shape != (n_rounds,) for arr in (reward, action, position, pscore)
    ):
        raise ValueError("expected aligned nonempty feedback vectors")

    shape = (n_rounds, feedback["n_actions"], feedback["len_list"])
    pi = np.asarray(action_dist, dtype=float)
    if pi.shape == shape[:2]:
        # A position-independent policy is broadcast over recommendation slots.
        pi = np.broadcast_to(pi[:, :, None], shape)
    q_hat = np.asarray(predicted_reward, dtype=float)
    if pi.shape != shape or q_hat.shape != shape:
        raise ValueError("expected round x action x position tensors")

    for index, size in ((action, shape[1]), (position, shape[2])):
        if index.dtype.kind not in "iu" or np.any((index < 0) | (index >= size)):
            raise ValueError("invalid logged action or position")
    if np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1):
        raise ValueError("invalid evaluation distribution")
    if not np.isfinite(q_hat).all() or not np.isfinite(reward).all():
        raise ValueError("nonfinite reward prediction")
    if not np.all((pscore > 0) & (pscore <= 1)):
        raise ValueError("invalid propensity score")

    rows = np.arange(n_rounds)
    weights = pi[rows, action, position] / pscore
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        weights = np.minimum(weights, clip)
    if not np.isfinite(weights).all() or not weights.any():
        raise ValueError("nonfinite weights or zero observed target-policy mass")

    dm_round = (pi[rows, :, position] * q_hat[rows, :, position]).sum(axis=1)
    return dict(
        DM=dm_round.mean(),
        IPS=np.mean(weights * reward),
        SNIPS=np.average(reward, weights=weights),
        DR=np.mean(dm_round + weights * (reward - q_hat[rows, action, position])),
    )
