"""Construção do tempo até o evento e do tipo de evento a partir do LendingClub.

A base não traz o histórico mensal de pagamentos, só a foto final de cada contrato.
O tempo é reconstruído com as datas de emissão (`issue_d`) e do último pagamento
(`last_pymnt_d`), e o tipo de evento sai do `loan_status`.

Convenções (em meses desde a emissão):
- Default: o tempo é o mês da primeira parcela não paga, ou seja, um mês depois do
  último pagamento. Um contrato que nunca pagou nada tem tempo 1.
- Pré-pagamento: o tempo é o mês da quitação. Quitar na última parcela ou depois
  não é pré-pagamento, e o contrato é censurado no prazo (sobreviveu ao contrato).
- Em dia na data da foto: censurado no fim da observação.
- Em atraso, sem baixa ainda: censurado no último pagamento, o último momento
  em que se sabe que estava saudável.

Fim da observação (censura administrativa): um contrato só vira "Charged Off" uns
5 meses depois da primeira parcela não paga. Quem parou de pagar perto da data da
foto ainda aparece como em atraso (ou até em dia), então os defaults dos últimos
meses estão incompletos. Para não subestimar o risco, a observação de *todos* os
contratos termina em `observation_end`, alguns meses antes da foto: qualquer evento
depois disso vira censura nessa data. Contratos emitidos a partir de
`observation_end` ficam com tempo 0 e devem ser descartados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CENSORED, DEFAULT, PREPAY = 0, 1, 2
EVENT_LABELS = {CENSORED: "censurado", DEFAULT: "default", PREPAY: "pré-pagamento"}

DEFAULT_STATUSES = {"Charged Off", "Default"}
PAID_STATUSES = {"Fully Paid"}
CURRENT_STATUSES = {"Current"}
LATE_STATUSES = {"In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}
KNOWN_STATUSES = DEFAULT_STATUSES | PAID_STATUSES | CURRENT_STATUSES | LATE_STATUSES


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Diferença em meses de calendário entre duas séries de datas (end - start)."""
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def parse_month(s: pd.Series) -> pd.Series:
    """Converte datas no formato do LendingClub ('Dec-2015') para datetime."""
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")


def build_survival_target(df: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Devolve `duration` (meses), `event` (0, 1 ou 2) e `term_months` para cada contrato.

    `df` precisa de `issue_d` e `last_pymnt_d` já em datetime, `loan_status` e `term`.
    Contratos com status fora de `KNOWN_STATUSES` (por exemplo, os que não atendem à
    política de crédito) devem ser removidos antes.
    """
    unknown = set(df["loan_status"].unique()) - KNOWN_STATUSES
    if unknown:
        raise ValueError(f"loan_status não reconhecido: {sorted(unknown)}")

    term = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    status = df["loan_status"]
    paid_months = months_between(df["issue_d"], df["last_pymnt_d"])
    # quem nunca pagou não tem last_pymnt_d e fica com zero meses pagos
    paid_months = paid_months.fillna(0).clip(lower=0)

    is_default = status.isin(DEFAULT_STATUSES)
    is_paid = status.isin(PAID_STATUSES)
    is_current = status.isin(CURRENT_STATUSES)
    is_late = status.isin(LATE_STATUSES)

    duration = pd.Series(np.nan, index=df.index)
    event = pd.Series(CENSORED, index=df.index)

    duration[is_default] = paid_months[is_default] + 1
    event[is_default] = DEFAULT

    prepaid = is_paid & (paid_months < term)
    duration[is_paid] = np.minimum(paid_months[is_paid], term[is_paid])
    event[prepaid] = PREPAY

    observed = months_between(df["issue_d"], pd.Series(observation_end, index=df.index))
    duration[is_current] = observed[is_current]
    duration[is_late] = paid_months[is_late]

    # o evento nunca acontece depois do fim do contrato nem no mês zero
    duration = duration.clip(lower=1, upper=term)

    # censura administrativa: nada é observado depois do fim da observação
    truncated = duration > observed
    duration[truncated] = observed[truncated].clip(lower=0)
    event[truncated] = CENSORED

    return pd.DataFrame(
        {"duration": duration.astype(int), "event": event.astype(int), "term_months": term},
        index=df.index,
    )
