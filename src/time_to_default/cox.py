"""Modelo de riscos proporcionais de Cox para o hazard específico do default.

O pré-pagamento entra como censura: o Cox estima o efeito das covariáveis sobre o
hazard de default *entre os contratos ainda ativos* (hazard específico da causa),
que é a leitura correta para efeitos. Probabilidades acumuladas usam riscos competitivos.

As covariáveis são transformadas para que cada coeficiente tenha uma unidade de
leitura direta (por exemplo, +1 p.p. de juros ou +10 p.p. de DTI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from time_to_default.events import DEFAULT, PREPAY

TRAIN_VINTAGES = range(2007, 2016)  # safras de ajuste; 2016–2018 ficam para o teste fora do tempo

# nome no modelo -> (rótulo, unidade de leitura do hazard ratio)
COVARIATES = {
    "int_rate": ("taxa de juros", "+1 p.p. ao ano"),
    "term_60": ("prazo de 60 meses", "60 vs. 36 meses"),
    "dti": ("comprometimento com dívidas (DTI)", "+10 p.p. da renda"),
    "pti": ("comprometimento com a parcela (PTI)", "+1 p.p. da renda"),
    "log2_income": ("renda anual", "renda 2× maior"),
    "fico": ("score FICO", "+20 pontos"),
    "home_rent": ("mora de aluguel", "vs. imóvel financiado"),
    "home_own": ("imóvel próprio quitado", "vs. imóvel financiado"),
    "emp_years": ("tempo de emprego", "+1 ano"),
    "emp_missing": ("tempo de emprego não informado", "vs. informado"),
    "verified": ("renda verificada", "vs. não verificada"),
    "revol_util": ("uso do rotativo", "+10 p.p. do limite"),
    "inquiries": ("consultas de crédito em 6 meses", "+1 consulta"),
    "past_delinq": ("atraso de 30+ dias nos últimos 2 anos", "sim vs. não"),
}


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Covariáveis do Cox, na unidade de `COVARIATES`, mais `duration`, `default` e `prepay`.

    Tratamentos:
    - PTI e uso do rotativo limitados ao percentil 99,9 (há uso do rotativo de 892%);
    - consultas limitadas a 6 (acima disso há poucos casos);
    - tempo de emprego faltante vira 0 com indicador próprio, porque a ausência é informativa;
    - linhas sem uso do rotativo (460 no treino) são descartadas.
    """
    X = pd.DataFrame(index=df.index)
    X["int_rate"] = df["int_rate"]
    X["term_60"] = (df["term_months"] == 60).astype(float)
    X["dti"] = df["dti"] / 10
    X["pti"] = df["pti"].clip(upper=21.5)
    X["log2_income"] = np.log2(df["annual_inc"])
    X["fico"] = df["fico"] / 20
    X["home_rent"] = (df["home"] == "aluguel").astype(float)
    X["home_own"] = (df["home"] == "própria").astype(float)
    X["emp_missing"] = df["emp_years"].isna().astype(float)
    X["emp_years"] = df["emp_years"].fillna(0)
    X["verified"] = df["verified"].astype(float)
    X["revol_util"] = df["revol_util"].clip(upper=103) / 10
    X["inquiries"] = df["inq_last_6mths"].clip(upper=6)
    X["past_delinq"] = (df["delinq_2yrs"] > 0).astype(float)
    X = X[list(COVARIATES)]
    X["duration"] = df["duration"]
    X["default"] = (df["event"] == DEFAULT).astype(int)
    X["prepay"] = (df["event"] == PREPAY).astype(int)
    return X.dropna()


def fit_cox(X: pd.DataFrame, covariates=None, **fit_kwargs) -> CoxPHFitter:
    """Ajusta o Cox (Efron para empates); covariáveis constantes no recorte são descartadas."""
    covariates = [c for c in (covariates or COVARIATES) if X[c].nunique() > 1]
    cols = covariates + ["duration", "default"]
    extra = [c for c in ("entry",) if c in X.columns]
    return CoxPHFitter().fit(
        X[cols + extra], "duration", "default", entry_col=extra[0] if extra else None, **fit_kwargs
    )


def hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Hazard ratios com IC de 95% e a leitura em português de cada um."""
    s = cph.summary
    out = pd.DataFrame({
        "variável": [COVARIATES[c][0] for c in s.index],
        "unidade": [COVARIATES[c][1] for c in s.index],
        "HR": s["exp(coef)"],
        "IC 95% inf": s["exp(coef) lower 95%"],
        "IC 95% sup": s["exp(coef) upper 95%"],
        "p-valor": s["p"],
    }, index=s.index)
    out["leitura"] = [interpret(c, hr) for c, hr in out["HR"].items()]
    return out


def interpret(covariate: str, hr: float) -> str:
    """Frase no formato 'Cada +1 p.p. de juros aumenta o risco relativo de default em 9%'."""
    label, unit = COVARIATES[covariate]
    change = 100 * (hr - 1)
    verb = "aumenta" if change >= 0 else "reduz"
    pct = f"{abs(change):.1f}%".replace(".", ",")
    if unit.startswith("vs.") or unit.startswith("sim") or unit.startswith("60"):
        return f"{label.capitalize()} ({unit}) {verb} o risco de default em {pct}"
    return f"Cada {unit} ({label}) {verb} o risco de default em {pct}"


# fases do contrato para os hazard ratios por período (meses desde a emissão)
PERIODS = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 60)]


def period_label(start: int, end: int) -> str:
    return f"{start + 1}–{end}"


def period_dataset(X: pd.DataFrame, start: int, end: int, event_col: str = "default") -> pd.DataFrame:
    """Contratos em risco na janela (start, end], com entrada atrasada em `start`.

    Ajustar o Cox em cada janela mostra se o hazard ratio muda ao longo do contrato,
    que é a forma mais legível de enxergar uma violação de riscos proporcionais.
    """
    sub = X[X["duration"] > start].copy()
    sub["entry"] = start
    beyond = sub["duration"] > end
    sub.loc[beyond, event_col] = 0
    sub.loc[beyond, "duration"] = end
    return sub


def fit_by_period(X: pd.DataFrame, periods=PERIODS) -> dict:
    """Um Cox por fase do contrato, cada um com os contratos ainda em risco no início da fase.

    Equivale a um Cox com coeficientes que mudam por degrau ao longo do tempo: a
    verossimilhança parcial se fatora por tempo de evento, e cada evento cai em uma fase.
    """
    return {period_label(a, b): fit_cox(period_dataset(X, a, b)) for a, b in periods}


def period_hr_table(models: dict) -> pd.DataFrame:
    """Hazard ratio de cada covariável (linhas) em cada fase do contrato (colunas)."""
    return pd.DataFrame({p: m.hazard_ratios_ for p, m in models.items()}).reindex(list(COVARIATES))
