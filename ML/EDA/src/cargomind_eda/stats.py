"""Statistical helpers: effect sizes first, p-values second.

With ~10k shipments almost any difference is 'significant', so every test here is
reported alongside an effect size and, where useful, a confidence interval.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .config import CLASSES


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def cramers_v(table):
    """Bias-corrected Cramér's V (Bergsma 2013) for a contingency table."""
    table = np.asarray(table)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))


def chi2_vs_target(df, col, target="delay_status"):
    """Chi-square test of independence + Cramér's V + expected-count check."""
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
    return {
        "feature": col,
        "levels": ct.shape[0],
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": cramers_v(ct),
        "min_expected": exp.min(),
        "pct_cells_expected_lt5": (exp < 5).mean() * 100,
    }


def class_rates(df, col, target="delay_status", positive=("At-Risk", "Critical")):
    """Per-level class shares, n, and a Wilson CI for the share of `positive` classes."""
    ct = pd.crosstab(df[col], df[target])
    out = ct.div(ct.sum(1), axis=0).mul(100).round(1)
    out.columns = [f"pct_{c}" for c in out.columns]
    out.insert(0, "n", ct.sum(1))
    k = ct[list(positive)].sum(1)
    ci = [wilson_ci(a, b) for a, b in zip(k, ct.sum(1))]
    out["pct_delayed(>12h)"] = (k / ct.sum(1) * 100).round(1)
    out["ci95_low"] = [round(c[0] * 100, 1) for c in ci]
    out["ci95_high"] = [round(c[1] * 100, 1) for c in ci]
    return out


def epsilon_squared(h, n):
    """Effect size for Kruskal-Wallis H (Tomczak & Tomczak 2014)."""
    return h * (n + 1) / (n**2 - 1)


def cliffs_delta(x, y):
    """Cliff's delta = P(X>Y) - P(X<Y), via Mann-Whitney U (O(n log n))."""
    x, y = np.asarray(x), np.asarray(y)
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def numeric_vs_target(df, col, target="delay_status", classes=CLASSES):
    """Kruskal-Wallis across the three classes + pairwise Cliff's delta + Spearman vs delay hours."""
    d = df[[col, target, "actual_delay_hours"]].dropna()
    groups = [d.loc[d[target] == c, col].values for c in classes]
    h, p = stats.kruskal(*groups)
    row = {
        "feature": col,
        "n": len(d),
        # Share of rows at the most common value. Above ~0.5 the tie correction inflates H, so epsilon_sq
        # overstates the effect; rank such features by Cliff's delta / Spearman instead.
        "tie_share": d[col].value_counts(normalize=True).iloc[0],
        "kruskal_H": h,
        "p_value": p,
        "epsilon_sq": epsilon_squared(h, len(d)),
    }
    for c, g in zip(classes, groups):
        row[f"median_{c}"] = np.median(g)
    for (a, ga), (b, gb) in itertools.combinations(zip(classes, groups), 2):
        row[f"cliff_{b}_vs_{a}"] = -cliffs_delta(ga, gb)  # positive => b tends to be larger
    rho, rp = stats.spearmanr(d[col], d["actual_delay_hours"])
    row["spearman_rho_delay_h"] = rho
    row["spearman_p"] = rp
    return row


def screen_numeric(df, cols, **kw):
    return pd.DataFrame([numeric_vs_target(df, c, **kw) for c in cols]).set_index("feature")


def screen_categorical(df, cols, **kw):
    return pd.DataFrame([chi2_vs_target(df, c, **kw) for c in cols]).set_index("feature")


def bootstrap_median_ci(x, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return np.quantile(meds, [alpha / 2, 1 - alpha / 2])


def magnitude_label(v, kind="cliff"):
    """Conventional magnitude bands (Romano et al. 2006 for Cliff's delta; Cohen-style for V)."""
    a = abs(v)
    if kind == "cliff":
        cuts = [(0.147, "negligible"), (0.33, "small"), (0.474, "medium")]
    else:  # cramers_v / epsilon_sq-ish
        cuts = [(0.1, "negligible"), (0.3, "small"), (0.5, "medium")]
    for c, lab in cuts:
        if a < c:
            return lab
    return "large"
