"""Out-of-fold click probabilities conditional on action and slot."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def cross_fit(feedback, folds=3, seed=42):
    """Return out-of-fold predictions q_hat with shape (n, n_actions, len_list).

    A separate logistic-regression model is fitted for every action/slot pair
    on the training portion of each fold.  The prediction for a row is always
    produced by a model that did not see that row, which reduces the optimistic
    bias of the Direct Method and the Doubly Robust estimator.

    Parameters
    ----------
    feedback : dict
        OBP-style bandit feedback.  Must contain context, action, position,
        reward, n_actions and len_list.
    folds : int, default=3
        Number of cross-fitting folds.
    seed : int, default=42
        Random seed used by KFold.

    Returns
    -------
    q_hat : ndarray, shape (n_rounds, n_actions, len_list)
        Predicted probability of a click for every action and slot.
    """
    x = np.asarray(feedback["context"], dtype=float)
    action = np.asarray(feedback["action"])
    position = np.asarray(feedback["position"])
    reward = np.asarray(feedback["reward"])
    n_rounds = len(reward)
    n_actions = int(feedback["n_actions"])
    len_list = int(feedback["len_list"])

    if x.ndim != 2 or x.shape[0] != n_rounds:
        raise ValueError("context must have shape (n_rounds, n_features)")
    if any(arr.shape != (n_rounds,) for arr in (action, position, reward)):
        raise ValueError("feedback vectors must be aligned")
    if not np.isin(reward, [0, 1]).all():
        raise ValueError("expected binary clicks")
    if n_actions <= 0 or len_list <= 0:
        raise ValueError("n_actions and len_list must be positive")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if n_rounds < folds:
        raise ValueError("not enough rounds for the requested folds")
    if np.any((action < 0) | (action >= n_actions)):
        raise ValueError("action is out of range")
    if np.any((position < 0) | (position >= len_list)):
        raise ValueError("position is out of range")

    q_hat = np.empty((n_rounds, n_actions, len_list), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    for train, test in splitter.split(x):
        # Use only training-fold labels for the fallback prior.
        fold_prior = float(np.clip(reward[train].mean(), 1e-6, 1.0 - 1e-6))
        for action_id in range(n_actions):
            for slot in range(len_list):
                subset = train[(action[train] == action_id) & (position[train] == slot)]
                labels = reward[subset]
                if labels.size == 0:
                    q_hat[test, action_id, slot] = fold_prior
                elif np.unique(labels).size < 2:
                    # Smoothed empirical rate when the training fold contains
                    # only one class, so predict_proba cannot be used.
                    numerator = labels.sum() + 2.0 * fold_prior
                    q_hat[test, action_id, slot] = numerator / (labels.size + 2.0)
                else:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(x[subset], labels)
                    q_hat[test, action_id, slot] = model.predict_proba(x[test])[:, 1]

    if not np.isfinite(q_hat).all():
        raise RuntimeError("reward model produced non-finite predictions")
    return q_hat
