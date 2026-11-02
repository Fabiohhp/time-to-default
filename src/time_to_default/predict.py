"""Probabilidade de default por horizonte com riscos competitivos.

Dois Cox específicos por causa (default e pré-pagamento), estratificados por prazo,
dão o hazard mensal de cada evento para cada contrato:

    ΔH_k(t | x) = ΔH0_k(t, prazo) · exp(β_k' x)

e a incidência acumulada de default sai da mesma recursão do Aalen-Johansen:

    S(t)        = S(t-1) · exp(-ΔH_1(t) - ΔH_2(t))
    P(default em t) = S(t-1) · (1 - exp(-ΔH(t))) · ΔH_1(t) / ΔH(t)

O prazo vira estrato (baseline próprio para 36 e 60 meses), porque o hazard de
pré-pagamento muda muito com o prazo e a curva de 36 meses termina no mês 36.

Com `periods`, cada causa é ajustada por fase do contrato (coeficientes que mudam em
degraus no tempo). Os incrementos do hazard são montados a partir do `baseline_hazard_`
de cada fase, preenchido com zero fora da janela. O `predict_cumulative_hazard` do
lifelines devolve, para modelos com entrada atrasada, o valor do primeiro tempo de
evento antes da janela, o que criaria um hazard fantasma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from time_to_default.cox import COVARIATES, period_dataset

PREDICT_COVARIATES = [c for c in COVARIATES if c != "term_60"]
MAX_MONTH = 60


def _fit(X: pd.DataFrame, event_col: str, covariates) -> CoxPHFitter:
    cols = [c for c in covariates if X[c].nunique() > 1]
    strata = ["term_60"] if X["term_60"].nunique() > 1 else None
    extra = ["entry"] if "entry" in X.columns else []
    keep = cols + (strata or []) + ["duration", event_col] + extra
    return CoxPHFitter().fit(X[keep], "duration", event_col, strata=strata,
                             entry_col="entry" if extra else None)


def _baseline_increments(model: CoxPHFitter) -> pd.DataFrame:
    """Incrementos mensais do hazard basal, meses 1..60 em linhas e prazo (0/1) em colunas."""
    base = model.baseline_hazard_.reindex(range(1, MAX_MONTH + 1), fill_value=0.0)
    if model.strata is None:
        # modelo sem estrato: só existe um prazo no recorte (fase 37–60, só 60 meses)
        base = pd.concat({1.0: base.iloc[:, 0]}, axis=1)
    base.columns = [float(c) if not isinstance(c, tuple) else float(c[0]) for c in base.columns]
    return base


class CompetingRisksCox:
    """Cox específico por causa para default e pré-pagamento, com previsão de incidência."""

    def __init__(self, covariates=PREDICT_COVARIATES, periods=None):
        self.covariates = list(covariates)
        self.periods = periods
        # multiplicador do hazard de default (recalibração de nível, ver `fit_default_scale`)
        self.default_scale = 1.0
        self.models: dict[str, list[tuple[tuple[int, int], CoxPHFitter]]] = {}

    def fit(self, X: pd.DataFrame) -> "CompetingRisksCox":
        windows = self.periods or [(0, MAX_MONTH)]
        for cause in ("default", "prepay"):
            self.models[cause] = [
                ((start, end), _fit(period_dataset(X, start, end, cause) if self.periods else X,
                                    cause, self.covariates))
                for start, end in windows
            ]
        return self

    def hazard_increments(self, X: pd.DataFrame, cause: str) -> np.ndarray:
        """Matriz contratos × meses (1..60) com o incremento do hazard da causa."""
        out = np.zeros((len(X), MAX_MONTH))
        term = X["term_60"].to_numpy(dtype=float)
        for _, model in self.models[cause]:
            base = _baseline_increments(model)
            partial = model.predict_partial_hazard(X).to_numpy()
            for stratum in base.columns:
                rows = term == stratum
                out[rows] += np.outer(partial[rows], base[stratum].to_numpy())
        months = np.arange(1, MAX_MONTH + 1)
        term_months = np.where(term == 1, 60, 36)
        out[months[None, :] > term_months[:, None]] = 0.0  # nada acontece depois do prazo
        return out

    def default_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """P(default no mês t) para t = 1..60, já descontando o pré-pagamento.

        Devolve a matriz inteira (contratos × 60); para bases grandes use
        `cumulative_default`, que processa em blocos.
        """
        h1 = self.default_scale * self.hazard_increments(X, "default")
        return _default_probabilities(h1, self.hazard_increments(X, "prepay"))

    def cumulative_default(self, X: pd.DataFrame, horizons=(12, 24), chunk: int = 100_000) -> pd.DataFrame:
        """Incidência acumulada de default nos horizontes pedidos (e no prazo do contrato)."""
        parts = []
        for i in range(0, len(X), chunk):
            part = X.iloc[i:i + chunk]
            cum = np.cumsum(self.default_probabilities(part), axis=1)
            out = {f"pd_{h}m": cum[:, h - 1] for h in horizons}
            term_months = np.where(part["term_60"].to_numpy() == 1, 60, 36)
            out["pd_lifetime"] = cum[np.arange(len(part)), term_months - 1]
            parts.append(pd.DataFrame(out, index=part.index))
        return pd.concat(parts)


def _default_probabilities(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Recursão do Aalen-Johansen a partir dos incrementos mensais de cada hazard."""
    total = h1 + h2
    before = np.hstack([np.ones((len(h1), 1)), np.exp(-np.cumsum(total, axis=1))[:, :-1]])
    share = np.divide(h1, total, out=np.zeros_like(h1), where=total > 0)
    return before * (1 - np.exp(-total)) * share


def fit_default_scale(model: CompetingRisksCox, X: pd.DataFrame, observed: float, horizon: int = 12,
                      low: float = 0.25, high: float = 4.0, tol: float = 1e-4) -> float:
    """Multiplicador do hazard de default que iguala a PD média prevista à observada em `horizon`.

    É a recalibração de nível usual em risco de crédito: os efeitos relativos (β) ficam
    como estão e só o nível do risco é ajustado a uma safra mais recente.
    """
    # só os meses até o horizonte importam, então as matrizes ficam pequenas
    h1 = np.vstack([model.hazard_increments(X.iloc[i:i + 100_000], "default")[:, :horizon]
                    for i in range(0, len(X), 100_000)])
    h2 = np.vstack([model.hazard_increments(X.iloc[i:i + 100_000], "prepay")[:, :horizon]
                    for i in range(0, len(X), 100_000)])

    def mean_pd(scale: float) -> float:
        return float(_default_probabilities(scale * h1, h2).sum(axis=1).mean())

    while high - low > tol:
        mid = (low + high) / 2
        low, high = (mid, high) if mean_pd(mid) < observed else (low, mid)
    return (low + high) / 2
