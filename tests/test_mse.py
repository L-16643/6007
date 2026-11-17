import numpy as np

from ope.estimators import evaluate
from scripts.evaluate_mse import summarize


def test_mse_averages_squared_errors_with_valid_count():
    common = dict(campaign="all", source="sample", policy="linucb", estimator="SNIPS",
                  seed=42, n=4, reference=0.25)
    rows = [dict(common, squared_error=e) for e in (0.01, 0.09, np.nan)]
    result, = summarize(rows)
    assert result["repeats"] == 3 and result["valid_repeats"] == 2
    assert np.isclose(result["mse"], 0.05)


def test_zero_overlap_only_invalidates_snips():
    log = dict(action=[0], position=[0], reward=[1], pscore=[0.5], n_actions=2, len_list=1)
    values = evaluate(log, [[0, 1]], np.array([[[0.2], [0.1]]]), allow_zero_mass=True)
    assert np.isnan(values["SNIPS"])
    assert values["DR"] == values["DM"] == 0.1
    assert values["IPS"] == 0
