"""Perda esperada (PCLD) por contrato: PD mensal × EAD × LGD, descontada.

    PE(h) = Σ_{t ≤ h} P(default no mês t) · EAD(t) · LGD / (1 + i)^t

- P(default no mês t) vem do modelo de riscos competitivos (`predict`), então quem
  pré-paga deixa de contribuir para a perda.
- EAD(t) é o saldo devedor da tabela de amortização depois de t − 1 parcelas pagas
  (o default acontece na t-ésima), multiplicado por um fator de ajuste estimado nos
  defaults: na prática o tomador costuma atrasar um pouco antes de parar de pagar.
- LGD é a perda líquida média dos defaults com recuperação já madura.
- O desconto usa a taxa do próprio contrato (taxa efetiva), como pede a IFRS 9.

O estágio 1 provisiona a perda de 12 meses; o estágio 2, a perda até o fim do contrato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from time_to_default.predict import MAX_MONTH, CompetingRisksCox


def scheduled_balance(principal, annual_rate_pct, installment, payments_made) -> np.ndarray:
    """Saldo devedor depois de `payments_made` parcelas (tabela Price), nunca negativo."""
    r = np.asarray(annual_rate_pct, dtype=float) / 1200
    k = np.asarray(payments_made, dtype=float)
    growth = (1 + r) ** k
    balance = np.asarray(principal) * growth - np.asarray(installment) * (growth - 1) / r
    return np.clip(balance, 0, None)


def estimate_lgd(defaults: pd.DataFrame) -> float:
    """LGD ponderada pela exposição: 1 − recuperação líquida ÷ saldo no default."""
    ead = defaults["loan_amnt"] - defaults["total_rec_prncp"]
    net = defaults["recoveries"] - defaults["collection_recovery_fee"]
    keep = ead > 0
    return float(1 - net[keep].sum() / ead[keep].sum())


def estimate_ead_factor(defaults: pd.DataFrame) -> float:
    """Saldo real no default ÷ saldo da tabela de amortização, ponderado pela exposição."""
    real = defaults["loan_amnt"] - defaults["total_rec_prncp"]
    sched = scheduled_balance(defaults["loan_amnt"], defaults["int_rate"], defaults["installment"],
                              defaults["duration"] - 1)
    return float(real.sum() / sched.sum())


def ead_matrix(loans: pd.DataFrame, factor: float = 1.0) -> np.ndarray:
    """EAD se o default acontecer no mês t (contratos × 60 meses), zero depois do prazo."""
    months = np.arange(1, MAX_MONTH + 1)
    term = loans["term_months"].to_numpy()[:, None]
    bal = scheduled_balance(loans["loan_amnt"].to_numpy()[:, None], loans["int_rate"].to_numpy()[:, None],
                            loans["installment"].to_numpy()[:, None], months[None, :] - 1)
    return np.where(months[None, :] <= term, factor * bal, 0.0)


def expected_loss(model: CompetingRisksCox, X: pd.DataFrame, loans: pd.DataFrame, lgd: float,
                  ead_factor: float = 1.0, horizons=(12,), discount: bool = True,
                  chunk: int = 100_000) -> pd.DataFrame:
    """PD e perda esperada por contrato nos horizontes pedidos e até o fim do prazo.

    `X` é a matriz do modelo e `loans` traz `loan_amnt`, `int_rate`, `installment` e
    `term_months`, com o mesmo índice.
    """
    parts = []
    months = np.arange(1, MAX_MONTH + 1)
    for i in range(0, len(X), chunk):
        xp, lp = X.iloc[i:i + chunk], loans.iloc[i:i + chunk]
        p = model.default_probabilities(xp)
        ead = ead_matrix(lp, ead_factor)
        df = (1 + lp["int_rate"].to_numpy()[:, None] / 1200) ** -months[None, :] if discount else 1.0
        loss = p * ead * lgd * df
        term_idx = lp["term_months"].to_numpy() - 1
        out = {}
        for h in horizons:
            out[f"pd_{h}m"] = p[:, :h].sum(axis=1)
            out[f"el_{h}m"] = loss[:, :h].sum(axis=1)
        cum_p, cum_loss = np.cumsum(p, axis=1), np.cumsum(loss, axis=1)
        rows = np.arange(len(xp))
        out["pd_lifetime"] = cum_p[rows, term_idx]
        out["el_lifetime"] = cum_loss[rows, term_idx]
        parts.append(pd.DataFrame(out, index=xp.index))
    return pd.concat(parts)
