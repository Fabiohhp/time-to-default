import numpy as np
import pandas as pd

from time_to_default.cox import design_matrix, fit_cox, interpret, period_dataset


def loans(n=4):
    return pd.DataFrame({
        "int_rate": [10.0] * n, "term_months": [36, 60, 36, 36][:n], "dti": [20.0] * n,
        "pti": [8.0, 50.0, 8.0, 8.0][:n], "annual_inc": [65_536.0] * n, "fico": [700.0] * n,
        "home": ["financiada", "aluguel", "própria", "aluguel"][:n],
        "emp_years": [3.0, np.nan, 10.0, 0.0][:n], "verified": [1, 0, 1, 0][:n],
        "revol_util": [50.0, 892.0, 30.0, np.nan][:n], "inq_last_6mths": [0.0, 9.0, 1.0, 2.0][:n],
        "delinq_2yrs": [0.0, 2.0, 0.0, 0.0][:n], "duration": [10, 20, 30, 5][:n], "event": [1, 2, 0, 1][:n],
    })


def test_design_matrix_units_and_treatments():
    X = design_matrix(loans())
    assert len(X) == 3  # a linha sem uso do rotativo sai
    row = X.iloc[1]
    assert row["term_60"] == 1 and row["home_rent"] == 1 and row["home_own"] == 0
    assert row["emp_missing"] == 1 and row["emp_years"] == 0
    assert row["pti"] == 21.5 and np.isclose(row["revol_util"], 10.3) and row["inquiries"] == 6
    assert row["past_delinq"] == 1 and row["default"] == 0  # pré-pagamento não é default
    assert X.iloc[0]["dti"] == 2.0 and X.iloc[0]["log2_income"] == 16.0 and X.iloc[0]["fico"] == 35.0


def test_interpretation_sentences():
    assert interpret("int_rate", 1.14) == "Cada +1 p.p. ao ano (taxa de juros) aumenta o risco de default em 14,0%"
    assert interpret("home_rent", 0.9).startswith("Mora de aluguel (vs. imóvel financiado) reduz")


def test_period_dataset_truncates_and_sets_entry():
    X = pd.DataFrame({"duration": [3, 8, 15], "default": [1, 1, 1]})
    p = period_dataset(X, 6, 12)
    assert p["duration"].tolist() == [8, 12]
    assert p["default"].tolist() == [1, 0]
    assert (p["entry"] == 6).all()


def test_cox_recovers_known_hazard_ratio():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(size=n)
    # tempos exponenciais com HR = 1,5 por unidade de x, censura aleatória e tempo em meses
    t = rng.exponential(30 / 1.5 ** x)
    c = rng.uniform(0, 60, n)
    X = pd.DataFrame({"int_rate": x, "duration": np.ceil(np.minimum(t, c)), "default": (t <= c).astype(int)})
    hr = fit_cox(X, ["int_rate"]).hazard_ratios_["int_rate"]
    assert abs(hr - 1.5) < 0.05
