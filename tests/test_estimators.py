import importlib.util
import unittest

import numpy as np

from ope.estimators import evaluate


class EstimatorTest(unittest.TestCase):
    def setUp(self):
        self.bf = dict(
            action=np.array([0, 1]),
            position=np.array([1, 0]),
            reward=np.array([1, 0]),
            pscore=np.array([0.5, 0.5]),
            n_actions=2,
            len_list=2,
        )
        self.pi = np.array([[[0.5, 1], [0.5, 0]], [[0.75, 0], [0.25, 1]]])
        self.q = np.full((2, 2, 2), 0.4)

    def test_hand_calculated_values(self):
        values = evaluate(self.bf, self.pi, self.q)
        for name, expected in dict(DM=0.4, IPS=1.0, SNIPS=0.8, DR=0.9).items():
            self.assertAlmostEqual(values[name], expected)

    def test_invalid_propensity(self):
        self.bf["pscore"] = np.array([0.0, 0.5])
        with self.assertRaises(ValueError):
            evaluate(self.bf, self.pi, self.q)

    def test_action_distribution_must_sum_to_one(self):
        bad_pi = self.pi.copy()
        bad_pi[0, 0, 0] += 0.1
        with self.assertRaises(ValueError):
            evaluate(self.bf, bad_pi, self.q)

    def test_position_agnostic_action_distribution_is_broadcast(self):
        pi_2d = np.array([[0.5, 0.5], [0.75, 0.25]])
        broadcast = evaluate(self.bf, pi_2d, self.q)
        explicit_dist = np.broadcast_to(pi_2d[:, :, None], self.pi.shape)
        explicit = evaluate(self.bf, explicit_dist, self.q)
        for name in ("DM", "IPS", "SNIPS", "DR"):
            self.assertAlmostEqual(broadcast[name], explicit[name])

    def test_clipping_caps_importance_weights(self):
        unclipped = evaluate(self.bf, self.pi, self.q)
        clipped = evaluate(self.bf, self.pi, self.q, clip=0.75)
        self.assertLess(clipped["IPS"], unclipped["IPS"])

    @unittest.skipUnless(
        importlib.util.find_spec("obp") is not None,
        "Open Bandit Pipeline is not installed",
    )
    def test_matches_obp_reference_estimators(self):
        from obp.ope import (
            DirectMethod,
            DoublyRobust,
            InverseProbabilityWeighting,
            SelfNormalizedInverseProbabilityWeighting,
        )

        ours = evaluate(self.bf, self.pi, self.q)
        common = dict(
            reward=self.bf["reward"],
            action=self.bf["action"],
            pscore=self.bf["pscore"],
            action_dist=self.pi,
            position=self.bf["position"],
        )
        reference = {
            "DM": DirectMethod().estimate_policy_value(
                action_dist=self.pi,
                estimated_rewards_by_reg_model=self.q,
                position=self.bf["position"],
            ),
            "IPS": InverseProbabilityWeighting().estimate_policy_value(**common),
            "SNIPS": SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(
                **common
            ),
            "DR": DoublyRobust().estimate_policy_value(
                **common, estimated_rewards_by_reg_model=self.q
            ),
        }
        for name, value in reference.items():
            self.assertAlmostEqual(ours[name], value, places=12)
