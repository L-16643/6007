"""Checks for member C's position-aware frozen policies."""

import numpy as np

from ope.policies import LinUCB, LogisticTS
from scripts.train_policies import _context_batches


def _feedback():
    # The best item reverses between positions and between the two contexts.
    p, a, x = np.array(np.meshgrid([0, 1], [0, 1], [-1, 1], indexing="ij")).reshape(3, -1)
    return {
        "context": np.repeat(x, 64)[:, None].astype(np.float32),
        "action": np.repeat(a, 64),
        "position": np.repeat(p, 64),
        "reward": np.repeat((a == (p ^ (x < 0))).astype(int), 64),
        "n_actions": 2,
        "len_list": 2,
    }


def test_linucb_matches_disjoint_ridge_and_confidence_bonus():
    feedback = _feedback()
    repetitions = np.where(feedback["action"] == 0, 2, 1)
    for key in ("context", "action", "position", "reward"):
        feedback[key] = np.repeat(feedback[key], repetitions, axis=0)
    model = LinUCB(alpha=0.2, ridge=2).fit(feedback)
    context = np.array([[-2], [-1], [0], [1], [2]], dtype=np.float32)
    design = np.c_[context, np.ones(len(context))]
    scores = np.empty((len(context), 2, 2))
    for p in range(2):
        for a in range(2):
            mask = (feedback["position"] == p) & (feedback["action"] == a)
            x = np.c_[feedback["context"][mask], np.ones(mask.sum())]
            inv = np.linalg.inv(x.T @ x + 2 * np.eye(2))
            mean = design @ inv @ x.T @ feedback["reward"][mask]
            bonus = 0.2 * np.sqrt(np.einsum("ij,jk,ik->i", design, inv, design))
            scores[:, a, p] = mean + bonus

    chosen = model.predict_action_dist(context).argmax(axis=1)
    np.testing.assert_array_equal(chosen, scores.argmax(axis=1))
    # At x=0 the exploration bonus makes the less-observed item win.
    greedy = LinUCB(alpha=0, ridge=2).fit(feedback).predict_action_dist(context).argmax(axis=1)
    np.testing.assert_array_equal(chosen[2], [1, 1])
    np.testing.assert_array_equal(greedy[2], [0, 0])


def test_logistic_ts_is_frozen_and_models_context_and_slot():
    feedback = _feedback()
    context = np.array([[-1], [1], [-1]], dtype=np.float32)
    model = LogisticTS(seed=7).fit(feedback)
    probabilities = model.predict_action_dist(context)
    np.testing.assert_array_equal(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(probabilities[:1], model.predict_action_dist(context[:1]))
    np.testing.assert_array_equal(probabilities, LogisticTS(seed=7).fit(feedback).predict_action_dist(context))
    # At x=0 the scores come from the sampled intercepts alone.
    neutral = model.predict_action_dist(np.zeros((1, 1), dtype=np.float32))[0]
    winners = model.draws_[:, -1].argmax(axis=-1)
    expected = np.stack([(winners == a).mean(axis=1) for a in range(2)])
    np.testing.assert_array_equal(neutral, expected)
    assert ((neutral > 0) & (neutral < 1)).any()


def test_logistic_ts_handles_unseen_items_and_single_class_groups():
    feedback = _feedback()
    feedback["n_actions"] = 3
    probabilities = LogisticTS().fit(feedback).predict_action_dist(feedback["context"][:2])
    assert probabilities.shape == (2, 3, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(probabilities.sum(axis=1), 1)


def test_batched_context_deduplication_restores_source_order():
    context = np.array([[3, 1], [1, 2], [3, 1], [2, 2], [1, 2]], dtype=np.float32)
    for rows, unique, inverse in _context_batches(context, 3):
        np.testing.assert_array_equal(unique[inverse], context[rows])
