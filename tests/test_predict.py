import numpy as np
import pandas as pd
import pytest

from time_to_default.predict import CompetingRisksCox, fit_default_scale
from time_to_default.survival import life_table


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(3)
    n = 30_000
    x = rng.normal(size=n)
    term_60 = rng.integers(0, 2, n).astype(float)
    term = np.where(term_60 == 1, 60, 36)
    t_default = rng.exponential(80 / 1.4 ** x)
    t_prepay = rng.exponential(60, n)
    t_censor = rng.uniform(1, 70, n)
    t = np.minimum.reduce([t_default, t_prepay, t_censor, term])
    duration = np.clip(np.ceil(t), 1, term)
    event = np.select([t == t_default, t == t_prepay], [1, 2], 0)
    return pd.DataFrame({
        "int_rate": x, "term_60": term_60, "duration": duration,
        "default": (event == 1).astype(int), "prepay": (event == 2).astype(int), "event": event,
    })


def test_average_prediction_matches_aalen_johansen(simulated):
    model = CompetingRisksCox(covariates=["int_rate"]).fit(simulated)
    pred = model.cumulative_default(simulated, horizons=(12, 24))
    for term_60, months in [(0.0, 36), (1.0, 60)]:
        rows = simulated["term_60"] == term_60
        tab = life_table(simulated.loc[rows, "duration"], simulated.loc[rows, "event"])
        for h in (12, 24):
            assert abs(pred.loc[rows, f"pd_{h}m"].mean() - tab.loc[h, "cif_default"]) < 0.01


def test_riskier_covariate_gets_higher_pd(simulated):
    model = CompetingRisksCox(covariates=["int_rate"]).fit(simulated)
    pred = model.cumulative_default(simulated, horizons=(12,))
    assert np.corrcoef(simulated["int_rate"], pred["pd_12m"])[0, 1] > 0.9


def test_no_default_probability_after_term(simulated):
    model = CompetingRisksCox(covariates=["int_rate"]).fit(simulated)
    p = model.default_probabilities(simulated.head(500))
    short = simulated.head(500)["term_60"].to_numpy() == 0
    assert np.all(p[short, 36:] == 0)


def test_period_model_without_time_effect_matches_single_model(simulated):
    single = CompetingRisksCox(covariates=["int_rate"]).fit(simulated)
    by_period = CompetingRisksCox(covariates=["int_rate"], periods=[(0, 12), (12, 60)]).fit(simulated)
    a = single.cumulative_default(simulated.head(2000))["pd_24m"]
    b = by_period.cumulative_default(simulated.head(2000))["pd_24m"]
    assert np.abs(a - b).max() < 0.01


def test_default_scale_hits_target(simulated):
    model = CompetingRisksCox(covariates=["int_rate"]).fit(simulated)
    sample = simulated.head(5000)
    base = model.cumulative_default(sample, horizons=(12,))["pd_12m"].mean()
    scale = fit_default_scale(model, sample, observed=1.2 * base, horizon=12)
    assert scale > 1.2  # o pré-pagamento compete, então o hazard precisa subir um pouco mais que 20%
    model.default_scale = scale
    assert abs(model.cumulative_default(sample, horizons=(12,))["pd_12m"].mean() - 1.2 * base) < 1e-3
