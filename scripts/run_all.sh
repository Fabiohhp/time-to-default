#!/usr/bin/env bash
# Reproduz o projeto do zero: dados, dependências, testes, notebooks em ordem e README.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync
uv run python scripts/download_data.py
uv run pytest -q

# a base parquet é recriada a partir do CSV, então resultados antigos não vazam para a nova rodada
rm -f data/processed/*.parquet reports/metrics.json

for notebook in notebooks/0*.ipynb; do
    echo "executando $notebook"
    uv run jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "$notebook"
done

uv run python scripts/build_readme.py
