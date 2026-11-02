import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import CLASS_COLORS, CLASSES, FIGURES, TABLES


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e6e5e1",
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.4g}")


PALETTE = [CLASS_COLORS[c] for c in CLASSES]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")


def save_table(df, name):
    df.to_csv(TABLES / f"{name}.csv")


def stacked_class_bar(df, col, ax, order=None, target="delay_status", title=None, sort_by_risk=True):
    """100% stacked bar of class shares per level, with n annotated."""
    ct = pd.crosstab(df[col], df[target])
    share = ct.div(ct.sum(1), axis=0) * 100
    if order is not None:
        share, ct = share.loc[order], ct.loc[order]
    elif sort_by_risk:
        key = share["On-Time"].sort_values(ascending=False).index
        share, ct = share.loc[key], ct.loc[key]
    share[CLASSES].plot.barh(stacked=True, ax=ax, color=PALETTE, width=0.8, edgecolor="white", linewidth=1)
    for i, (lvl, n) in enumerate(ct.sum(1).items()):
        ax.text(101, i, f"n={n:,}", va="center", fontsize=8, color="#52514e")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of shipments (%)")
    ax.set_ylabel(col)
    ax.invert_yaxis()
    ax.legend(title="delay_status", bbox_to_anchor=(1.12, 1), loc="upper left")
    ax.set_title(title or f"Delay class mix by {col}")
    return share
