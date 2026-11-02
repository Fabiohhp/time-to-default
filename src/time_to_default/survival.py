"""Tábua de vida mensal: Kaplan-Meier, riscos competitivos (Aalen-Johansen) e hazard.

Como o tempo é medido em meses inteiros, os estimadores saem exatos de uma tábua com,
para cada mês t, quantos contratos estavam em risco e quantos tiveram cada desfecho.
Convenção (a mesma do lifelines): quem é censurado no mês t ainda conta como em risco em t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test

from time_to_default.events import CENSORED, DEFAULT, PREPAY


def life_table(durations, events, weights=None) -> pd.DataFrame:
    """Tábua de vida mensal com hazards, Kaplan-Meier e incidências acumuladas.

    Colunas:
    - `at_risk`, `default`, `prepay`, `censored`: contagens no mês t;
    - `hazard_default`, `hazard_prepay`: probabilidade do evento no mês t para quem chegou a t;
    - `km_default`: Kaplan-Meier do default tratando o pré-pagamento como censura;
    - `event_free`: probabilidade de não ter tido evento nenhum até t;
    - `cif_default`, `cif_prepay`: incidência acumulada de cada evento (Aalen-Johansen).
    """
    df = pd.DataFrame({"t": np.asarray(durations), "e": np.asarray(events)})
    df["w"] = 1.0 if weights is None else np.asarray(weights, dtype=float)
    counts = (
        df.pivot_table(index="t", columns="e", values="w", aggfunc="sum", fill_value=0)
        .reindex(columns=[CENSORED, DEFAULT, PREPAY], fill_value=0)
    )
    counts = counts.reindex(range(1, int(df["t"].max()) + 1), fill_value=0)
    counts.columns = ["censored", "default", "prepay"]
    counts.index.name = "month"

    total = counts.sum(axis=1)
    out = counts.copy()
    out.insert(0, "at_risk", total[::-1].cumsum()[::-1])
    n = out["at_risk"].replace(0, np.nan)
    out["hazard_default"] = out["default"] / n
    out["hazard_prepay"] = out["prepay"] / n
    out["km_default"] = (1 - out["hazard_default"].fillna(0)).cumprod()
    out["event_free"] = (1 - (out["hazard_default"] + out["hazard_prepay"]).fillna(0)).cumprod()
    event_free_before = out["event_free"].shift(1, fill_value=1.0)
    out["cif_default"] = (event_free_before * out["hazard_default"].fillna(0)).cumsum()
    out["cif_prepay"] = (event_free_before * out["hazard_prepay"].fillna(0)).cumsum()
    return out


def life_tables_by(data: pd.DataFrame, group: str) -> dict:
    """Uma tábua de vida por categoria de `group`, na ordem das categorias."""
    levels = (
        data[group].cat.categories if isinstance(data[group].dtype, pd.CategoricalDtype)
        else sorted(data[group].dropna().unique())
    )
    return {
        level: life_table(sub["duration"], sub["event"])
        for level in levels
        if len(sub := data[data[group] == level])
    }


def summary_at(tables: dict, months=(12, 24, 36)) -> pd.DataFrame:
    """Default acumulado por grupo em horizontes fixos: KM ingênuo x Aalen-Johansen."""
    rows = []
    for level, tab in tables.items():
        row = {"grupo": level, "contratos": int(tab["at_risk"].iloc[0])}
        for m in months:
            if m <= tab.index.max():
                row[f"KM {m}m"] = 1 - tab.loc[m, "km_default"]
                row[f"AJ {m}m"] = tab.loc[m, "cif_default"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("grupo")


def logrank_by(data: pd.DataFrame, group: str) -> pd.Series:
    """Teste log-rank do hazard de default entre as categorias de `group`.

    Os contratos são agregados por (grupo, mês, default) e entram com peso, o que
    dá o mesmo resultado do teste linha a linha em uma fração do tempo.
    """
    agg = (
        data.assign(is_default=(data["event"] == DEFAULT).astype(int))
        .groupby([group, "duration", "is_default"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    res = multivariate_logrank_test(
        agg["duration"], agg[group].astype(str), agg["is_default"], weights=agg["n"]
    )
    return pd.Series({
        "estatística": res.test_statistic,
        "graus de liberdade": res.degrees_of_freedom,
        "p-valor": res.p_value,
    }, name=group)


def smoothed_hazard(tab: pd.DataFrame, window: int = 3, drop_last: bool = True) -> pd.Series:
    """Hazard de default (em % ao mês) suavizado por média móvel centrada.

    O último mês do contrato fica de fora por padrão. Ele acumula os contratos que
    atrasaram, pagaram a última parcela depois do vencimento e só então foram
    baixados, e o tempo deles foi truncado no prazo.
    """
    h = 100 * tab["hazard_default"]
    if drop_last:
        h = h.iloc[:-1]
    return h.rolling(window, center=True, min_periods=1).mean()


def critical_window(hazard: pd.Series, share_of_peak: float = 0.9) -> dict:
    """Maior trecho contínuo em torno do pico em que o hazard fica acima de `share_of_peak` do pico."""
    peak_month = int(hazard.idxmax())
    above = hazard >= share_of_peak * hazard.max()
    start = end = peak_month
    while start - 1 in above.index and above[start - 1]:
        start -= 1
    while end + 1 in above.index and above[end + 1]:
        end += 1
    return {"peak_month": peak_month, "peak_hazard": float(hazard.max()), "start": start, "end": end}


def default_share_by_window(tab: pd.DataFrame, bins) -> pd.Series:
    """Fração dos defaults (Aalen-Johansen) que acontece em cada janela de meses."""
    increments = tab["cif_default"].diff().fillna(tab["cif_default"])
    windows = pd.cut(increments.index, bins, labels=[f"{a + 1}–{b}" for a, b in zip(bins[:-1], bins[1:])])
    return increments.groupby(windows, observed=False).sum() / tab["cif_default"].iloc[-1]
