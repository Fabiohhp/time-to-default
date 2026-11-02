"""Baixa a base do LendingClub do Kaggle para data/raw/ (não precisa de credencial).

Só o arquivo de contratos aceitos é usado (`accepted_2007_to_2018Q4.csv.gz`, 375 MB).
"""
import sys
from pathlib import Path

import kagglehub

DATASET = "wordsforthewise/lending-club"
FILE = "accepted_2007_to_2018Q4.csv.gz"


def main(out_dir: Path) -> None:
    target = out_dir / FILE
    if target.exists():
        print(f"{target} já existe, nada a baixar")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(DATASET, path=FILE, output_dir=str(out_dir))
    print(f"baixado em {path}")


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "data" / "raw"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
