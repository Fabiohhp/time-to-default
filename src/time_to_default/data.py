"""Leitura do LendingClub em pedaços, com só as colunas usadas e tipos compactos.

O CSV bruto tem 2,2 milhões de linhas e 151 colunas (1,6 GB descompactado). Lido de
uma vez com o pandas, estoura a memória de uma máquina de 8 GB, então a leitura é
feita em blocos e o resultado é salvo em parquet uma única vez.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from time_to_default.events import parse_month

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
INTERIM_PATH = ROOT / "data" / "processed" / "loans_raw.parquet"

COLUMNS = {
    "id": "string",
    "issue_d": "string",
    "last_pymnt_d": "string",
    "loan_status": "category",
    "term": "category",
    "loan_amnt": "float32",
    "int_rate": "float32",
    "installment": "float32",
    "annual_inc": "float64",
    "dti": "float32",
    "grade": "category",
    "sub_grade": "category",
    "emp_length": "category",
    "home_ownership": "category",
    "verification_status": "category",
    "purpose": "category",
    "application_type": "category",
    "fico_range_low": "float32",
    "fico_range_high": "float32",
    "revol_util": "float32",
    "delinq_2yrs": "float32",
    "inq_last_6mths": "float32",
    "open_acc": "float32",
    "pub_rec": "float32",
    "mort_acc": "float32",
    "addr_state": "category",
    # perda dado o default (notebook 06)
    "total_rec_prncp": "float32",
    "recoveries": "float32",
    "collection_recovery_fee": "float32",
}


def read_raw(path: Path = RAW_PATH, chunksize: int = 200_000) -> pd.DataFrame:
    """Lê o CSV bruto em blocos e devolve as colunas de `COLUMNS` com datas convertidas."""
    chunks = []
    reader = pd.read_csv(
        path,
        usecols=list(COLUMNS),
        dtype={c: ("string" if t == "category" else t) for c, t in COLUMNS.items()},
        chunksize=chunksize,
    )
    for chunk in reader:
        # o arquivo tem linhas de rodapé sem id numérico (totais do relatório)
        chunk = chunk[chunk["id"].str.fullmatch(r"\d+", na=False)]
        for col in ("issue_d", "last_pymnt_d"):
            chunk[col] = parse_month(chunk[col])
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    cats = [c for c, t in COLUMNS.items() if t == "category"]
    df[cats] = df[cats].astype("category")
    df["id"] = df["id"].astype("int64")
    return df


def load_raw(rebuild: bool = False) -> pd.DataFrame:
    """Devolve a base bruta reduzida, criando o parquet na primeira chamada."""
    if rebuild or not INTERIM_PATH.exists():
        INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
        read_raw().to_parquet(INTERIM_PATH, index=False)
    return pd.read_parquet(INTERIM_PATH)
