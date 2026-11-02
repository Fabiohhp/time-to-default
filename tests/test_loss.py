import numpy as np
import pandas as pd

from time_to_default.loss import ead_matrix, estimate_ead_factor, estimate_lgd, expected_loss, scheduled_balance


def price_installment(principal, annual_rate_pct, n):
    r = annual_rate_pct / 1200
    return principal * r / (1 - (1 + r) ** -n)


def test_balance_starts_at_principal_and_ends_at_zero():
    inst = price_installment(10_000, 12.0, 36)
    assert np.isclose(scheduled_balance(10_000, 12.0, inst, 0), 10_000)
    assert np.isclose(scheduled_balance(10_000, 12.0, inst, 36), 0, atol=1e-6)
    assert scheduled_balance(10_000, 12.0, inst, 18) < 10_000 / 2 + 1_000


def test_lgd_and_ead_factor():
    defaults = pd.DataFrame({"loan_amnt": [10_000.0, 20_000.0], "total_rec_prncp": [2_000.0, 0.0],
                             "recoveries": [1_000.0, 2_000.0], "collection_recovery_fee": [200.0, 0.0],
                             "int_rate": [12.0, 12.0], "duration": [1, 1],
                             "installment": [price_installment(10_000, 12.0, 36), price_installment(20_000, 12.0, 36)]})
    # exposição 8.000 + 20.000; recuperação líquida 800 + 2.000
    assert np.isclose(estimate_lgd(defaults), 1 - 2_800 / 28_000)
    # default no mês 1: o saldo agendado é o principal inteiro (30.000), o real é 28.000
    assert np.isclose(estimate_ead_factor(defaults), 28_000 / 30_000)


def test_ead_matrix_is_zero_after_term():
    loans = pd.DataFrame({"loan_amnt": [10_000.0], "int_rate": [12.0], "term_months": [36],
                          "installment": [price_installment(10_000, 12.0, 36)]})
    ead = ead_matrix(loans)
    assert np.isclose(ead[0, 0], 10_000) and ead[0, 35] > 0 and (ead[0, 36:] == 0).all()


class FlatModel:
    """Modelo fictício: probabilidade fixa de default em cada mês."""

    def __init__(self, p):
        self.p = p

    def default_probabilities(self, X):
        return np.full((len(X), 60), self.p)


def test_expected_loss_matches_hand_computation():
    loans = pd.DataFrame({"loan_amnt": [10_000.0], "int_rate": [12.0], "term_months": [36],
                          "installment": [price_installment(10_000, 12.0, 36)]})
    X = pd.DataFrame(index=loans.index)
    out = expected_loss(FlatModel(0.01), X, loans, lgd=0.9, horizons=(12,), discount=False)
    expected_12 = 0.01 * 0.9 * ead_matrix(loans)[0, :12].sum()
    assert np.isclose(out.loc[0, "el_12m"], expected_12)
    assert np.isclose(out.loc[0, "pd_lifetime"], 0.36)  # 36 meses × 1%, nada depois do prazo
    discounted = expected_loss(FlatModel(0.01), X, loans, lgd=0.9, horizons=(12,))
    assert discounted.loc[0, "el_12m"] < out.loc[0, "el_12m"]
