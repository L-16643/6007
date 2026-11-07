"""数据流水线与 Ground Truth（成员 B）。

    from ope.data import load_bandit_feedback
    bf = load_bandit_feedback("all", policy="random", split="train")
    X, a, r, ps = bf["context"], bf["action"], bf["reward"], bf["pscore"]
"""

from .dataset import DataConfig, build_all, build_campaign, load_bandit_feedback, load_encoder, load_meta
from .features import ContextEncoder, build_action_context
from .ground_truth import on_policy_value, policy_value_from_random_logs, uniform_action_dist, validate_action_dist
from .io import CAMPAIGNS, LEN_LIST, N_ITEMS, POLICIES, download_full, download_sample

__all__ = [
    "CAMPAIGNS",
    "LEN_LIST",
    "N_ITEMS",
    "POLICIES",
    "ContextEncoder",
    "DataConfig",
    "build_action_context",
    "build_all",
    "build_campaign",
    "download_full",
    "download_sample",
    "load_bandit_feedback",
    "load_encoder",
    "load_meta",
    "on_policy_value",
    "policy_value_from_random_logs",
    "uniform_action_dist",
    "validate_action_dist",
]
