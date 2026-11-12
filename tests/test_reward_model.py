import importlib.util
import unittest

import numpy as np

from ope.reward_model import cross_fit
from ope_demo import build_feedback


class RewardModelTest(unittest.TestCase):
    def setUp(self):
        self.feedback = build_feedback("local", n_rounds=300, seed=42)

    def test_cross_fit_shape_and_range(self):
        q_hat = cross_fit(self.feedback, folds=3, seed=42)
        self.assertEqual(q_hat.shape, (300, 3, 1))
        self.assertTrue(np.isfinite(q_hat).all())
        self.assertGreaterEqual(q_hat.min(), 0.0)
        self.assertLessEqual(q_hat.max(), 1.0)

    def test_cross_fit_is_reproducible(self):
        first = cross_fit(self.feedback, folds=3, seed=7)
        second = cross_fit(self.feedback, folds=3, seed=7)
        np.testing.assert_allclose(first, second)

    def test_invalid_folds(self):
        with self.assertRaises(ValueError):
            cross_fit(self.feedback, folds=1)
        with self.assertRaises(ValueError):
            cross_fit(self.feedback, folds=301)

    def test_non_binary_reward(self):
        feedback = dict(self.feedback)
        feedback["reward"] = np.full(300, 0.5)
        with self.assertRaises(ValueError):
            cross_fit(feedback)

    @unittest.skipUnless(
        importlib.util.find_spec("obp") is not None,
        "Open Bandit Pipeline is not installed",
    )
    def test_obp_synthetic_feedback_contract(self):
        feedback = build_feedback("obp", n_rounds=200, seed=42)
        self.assertEqual(feedback["expected_reward"].shape, (200, 3))
        self.assertEqual(feedback["context"].shape, (200, 2))
        self.assertTrue(np.all(feedback["pscore"] > 0))
        q_hat = cross_fit(feedback, folds=2, seed=42)
        self.assertEqual(q_hat.shape, (200, 3, 1))


if __name__ == "__main__":
    unittest.main()
