import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from time_to_default.survival import critical_window, default_share_by_window, life_table, logrank_by

rng = np.random.default_rng(0)
N = 3_000
DURATIONS = rng.integers(1, 37, N)
EVENTS = rng.choice([0, 1, 2], N, p=[0.5, 0.15, 0.35])


def test_km_matches_lifelines():
    tab = life_table(DURATIONS, EVENTS)
    kmf = KaplanMeierFitter().fit(DURATIONS, EVENTS == 1)
    expected = kmf.survival_function_at_times(tab.index).to_numpy()
    np.testing.assert_allclose(tab["km_default"].to_numpy(), expected)


def test_cumulative_incidence_matches_lifelines():
    tab = life_table(DURATIONS, EVENTS)
    # sem jitter: o lifelines só sorteia desempates quando há eventos e censuras no mesmo tempo
    ajf = AalenJohansenFitter(calculate_variance=False, jitter_level=0.0).fit(
        DURATIONS, EVENTS, event_of_interest=1
    )
    expected = ajf.cumulative_density_.iloc[:, 0].reindex(tab.index, method="ffill")
    np.testing.assert_allclose(tab["cif_default"].to_numpy(), expected.to_numpy(), atol=1e-10)


def test_incidences_add_up_with_event_free():
    tab = life_table(DURATIONS, EVENTS)
    np.testing.assert_allclose(tab["cif_default"] + tab["cif_prepay"] + tab["event_free"], 1.0)


def test_naive_km_overstates_default_with_competing_risk():
    tab = life_table(DURATIONS, EVENTS)
    assert (1 - tab["km_default"].iloc[-1]) > tab["cif_default"].iloc[-1]


def test_hand_example():
    # 4 contratos: default no mês 1, pré-pagamento no 2, censura no 2, default no 3
    tab = life_table([1, 2, 2, 3], [1, 2, 0, 1])
    assert tab["at_risk"].tolist() == [4, 3, 1]
    assert np.isclose(tab.loc[1, "cif_default"], 1 / 4)
    assert np.isclose(tab.loc[2, "cif_prepay"], 3 / 4 * 1 / 3)
    assert np.isclose(tab.loc[3, "cif_default"], 1 / 4 + 3 / 4 * 2 / 3 * 1)


def test_weighted_logrank_equals_row_level():
    df = pd.DataFrame({
        "duration": DURATIONS, "event": EVENTS, "g": rng.choice(["a", "b", "c"], N),
    })
    row_level = multivariate_logrank_test(df["duration"], df["g"], df["event"] == 1)
    assert np.isclose(logrank_by(df, "g")["estatística"], row_level.test_statistic)


def test_critical_window_around_peak():
    h = pd.Series([0.1, 0.5, 0.95, 1.0, 0.92, 0.6, 0.95], index=range(1, 8))
    w = critical_window(h)
    # o 0,95 do mês 7 passa do corte, mas não é contínuo ao pico
    assert (w["peak_month"], w["start"], w["end"]) == (4, 3, 5)


def test_default_shares_sum_to_one():
    shares = default_share_by_window(life_table(DURATIONS, EVENTS), [0, 12, 24, 36])
    assert np.isclose(shares.sum(), 1.0)
    assert list(shares.index) == ["1–12", "13–24", "25–36"]
