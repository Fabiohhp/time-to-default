"""Filtros de escopo e variáveis derivadas da base de sobrevivência."""

from __future__ import annotations

import numpy as np
import pandas as pd

from time_to_default.data import ROOT
from time_to_default.events import KNOWN_STATUSES, build_survival_target

PROCESSED_PATH = ROOT / "data" / "processed" / "loans_survival.parquet"

SNAPSHOT = pd.Timestamp("2019-03-01")  # último last_pymnt_d da base: data da foto
# a partir de dez/2018 os defaults ainda não viraram charge-off (ver notebook 01)
OBSERVATION_END = pd.Timestamp("2018-11-01")

INCOME_BINS = [0, 40_000, 60_000, 80_000, 100_000, 150_000, np.inf]
INCOME_LABELS = ["até 40 mil", "40–60 mil", "60–80 mil", "80–100 mil", "100–150 mil", "acima de 150 mil"]

HOME_LABELS = {"MORTGAGE": "financiada", "RENT": "aluguel", "OWN": "própria"}

EMP_YEARS = {"< 1 year": 0, "1 year": 1, "10+ years": 10} | {f"{i} years": i for i in range(2, 10)}


def scope_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica os filtros de escopo em sequência e devolve a base filtrada e o funil.

    - fora da política de crédito: contratos de 2007 a 2010 que a própria LendingClub
      marcou como fora da política vigente;
    - pedidos conjuntos: `annual_inc` e `dti` são só do titular, enquanto a aprovação
      considerou a renda do casal (`annual_inc_joint`, `dti_joint`). O DTI do titular
      chega a 999 e não mede o comprometimento real. Além disso, o produto só existe
      desde out/2015 e 76% desses contratos ainda estavam em aberto na foto;
    - moradia rara (ANY, OTHER, NONE): pouco mais de mil contratos, sem tamanho para análise;
    - renda zero: o comprometimento de renda não é calculável.
    """
    steps = [
        ("base bruta", pd.Series(True, index=df.index)),
        ("dentro da política de crédito", df["loan_status"].isin(KNOWN_STATUSES)),
        ("pedido individual", df["application_type"] == "Individual"),
        ("moradia própria, financiada ou aluguel", df["home_ownership"].isin(HOME_LABELS.keys())),
        ("renda anual positiva", df["annual_inc"] > 0),
        ("emitido antes do fim da observação", df["issue_d"] < OBSERVATION_END),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = []
    for name, mask in steps:
        keep &= mask.fillna(False).astype(bool)
        funnel.append({"etapa": name, "contratos": int(keep.sum())})
    funnel = pd.DataFrame(funnel)
    funnel["removidos"] = (-funnel["contratos"].diff()).fillna(0).astype(int)
    return df[keep].copy(), funnel


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variáveis usadas nas curvas e no modelo de Cox."""
    out = pd.DataFrame(index=df.index)
    out["int_rate"] = df["int_rate"]
    out["dti"] = df["dti"]
    # comprometimento da renda com a parcela deste empréstimo, em % da renda mensal
    out["pti"] = 100 * df["installment"] / (df["annual_inc"] / 12)
    out["annual_inc"] = df["annual_inc"]
    out["log_income"] = np.log(df["annual_inc"])
    out["income_band"] = pd.cut(
        df["annual_inc"], INCOME_BINS, labels=INCOME_LABELS, right=False
    )
    out["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    out["loan_amnt"] = df["loan_amnt"]
    out["grade"] = df["grade"].astype(str).astype("category")
    out["home"] = pd.Categorical(
        df["home_ownership"].astype(str).map(HOME_LABELS), categories=list(HOME_LABELS.values())
    )
    out["emp_years"] = df["emp_length"].astype(str).map(EMP_YEARS)  # NaN = não informado
    out["verified"] = (df["verification_status"] != "Not Verified").astype(int)
    out["purpose"] = df["purpose"].astype(str)
    out["revol_util"] = df["revol_util"]
    out["inq_last_6mths"] = df["inq_last_6mths"]
    out["delinq_2yrs"] = df["delinq_2yrs"]
    out["installment"] = df["installment"]
    out["total_rec_prncp"] = df["total_rec_prncp"]
    out["recoveries"] = df["recoveries"]
    out["collection_recovery_fee"] = df["collection_recovery_fee"]
    out["issue_d"] = df["issue_d"]
    out["vintage"] = df["issue_d"].dt.year
    return out


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base de sobrevivência final (alvo + variáveis) e o funil de filtros."""
    df, funnel = scope_filters(raw)
    df["loan_status"] = df["loan_status"].astype(str)
    target = build_survival_target(df, OBSERVATION_END)
    data = pd.concat([df[["id", "loan_status"]], target, add_features(df)], axis=1)
    return data.reset_index(drop=True), funnel
