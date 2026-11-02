"""Métricas salvas em reports/metrics.json: fonte única dos números citados no README."""
import json

from time_to_default.data import ROOT

METRICS_PATH = ROOT / "reports" / "metrics.json"


def update_metrics(section: str, values: dict) -> dict:
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics[section] = values
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def pct(value: float, decimals: int = 1) -> str:
    """Formata proporção como percentual no padrão brasileiro: 0,123 -> 12,3%."""
    return f"{100 * value:.{decimals}f}%".replace(".", ",")
