"""Estilo visual comum aos notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns

from time_to_default.data import ROOT

FIG_DIR = ROOT / "reports" / "figures"

# laranja = default, azul = pré-pagamento / geral, cinza = censura
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
LIGHT_GRAY = "#c3c2b7"
EVENT_COLORS = {"default": ORANGE, "pré-pagamento": BLUE, "censurado": LIGHT_GRAY}
# ordem fixa da paleta categórica validada (nunca reciclar cores)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def set_style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#d0cfca", "grid.color": "#ecebe7", "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.bbox": "tight",
    })


def savefig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)

# rampa sequencial azul para categorias ordenadas (mais escuro = mais da dimensão)
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]


def ordinal_colors(n: int) -> list[str]:
    """`n` passos espaçados da rampa azul, do mais claro ao mais escuro."""
    idx = [round(i * (len(BLUE_RAMP) - 1) / max(n - 1, 1)) for i in range(n)]
    return [BLUE_RAMP[i] for i in idx]


def percent_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    import matplotlib.ticker as mtick

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(mtick.PercentFormatter(1, decimals=decimals))


def plot_curves_by_term(tables_by_term: dict, column: str, colors: list[str], title: str,
                        ylabel: str, direct_labels: bool = False):
    """Painéis lado a lado (36 e 60 meses) com uma curva por grupo.

    `tables_by_term` = {prazo: {grupo: tábua de vida}}. Com até 4 grupos, cada curva
    recebe rótulo na ponta; acima disso, só a legenda.
    """
    terms = list(tables_by_term)
    fig, axes = plt.subplots(1, len(terms), figsize=(11, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": terms})
    for ax, term in zip(axes, terms):
        for (level, tab), color in zip(tables_by_term[term].items(), colors):
            ax.plot(tab.index, tab[column], color=color, label=str(level))
            if direct_labels:
                ax.annotate(str(level), (tab.index[-1], tab[column].iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=9, color=GRAY)
        ax.set_title(f"contratos de {term} meses", fontsize=11, fontweight="normal", color=GRAY)
        ax.set_xlabel("meses desde a emissão")
        ax.set_xticks(range(0, term + 1, 12))
        ax.set_xlim(0, term * (1.18 if direct_labels else 1.0))
        percent_axis(ax)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontweight="bold", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(len(labels), 7), loc="lower center",
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    return fig
