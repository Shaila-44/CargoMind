"""Feature importance and interpretability.

Methods (all on the saved models; permutation and SHAP on the test period):
* Logistic Regression coefficients (standardised numeric inputs, log-odds per SD).
* Native tree importances: Random Forest impurity decrease, XGBoost and LightGBM total gain.
* Permutation importance per feature (drop in macro-F1).
* Group permutation importance: all columns of a feature group are permuted together (same row
  shuffle), which handles correlated features that would otherwise share/hide importance. Reported
  as the drop in macro-F1 and in Critical recall.
* SHAP (TreeExplainer) for XGBoost and LightGBM.

One-hot levels and missing-value indicators are summed back to their source column so every
method reports on the same 51 input features. None of this is causal.
"""
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

from . import config  # noqa: E402
from .features import CATEGORICAL, FEATURE_GROUPS, FEATURE_TO_GROUP, FEATURES, NUMERIC  # noqa: E402

OUT = config.RESULT_DIRS["feature_importance"]
N_REPEATS = 5


def source_feature(encoded):
    if encoded.startswith("missingindicator_"):
        return encoded[len("missingindicator_"):]
    if encoded in NUMERIC:
        return encoded
    matches = [c for c in CATEGORICAL if encoded.startswith(c + "_")]
    return max(matches, key=len) if matches else encoded


def _to_source(values, names):
    s = pd.Series(values, index=[source_feature(n) for n in names])
    return s.groupby(level=0).sum().reindex(FEATURES).fillna(0)


def native_importances(models):
    out = {}
    for key, label in [("random_forest", "rf_impurity"), ("xgboost", "xgb_gain"), ("lightgbm", "lgbm_gain")]:
        pipe = models[key]
        names = pipe.named_steps["prep"].get_feature_names_out()
        est = pipe.named_steps["model"]
        if key == "xgboost":
            gain = est.get_booster().get_score(importance_type="total_gain")
            vals = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
        elif key == "lightgbm":
            vals = est.booster_.feature_importance(importance_type="gain")
        else:
            vals = est.feature_importances_
        s = _to_source(vals, names)
        out[label] = s / s.sum()
    return pd.DataFrame(out)


def lr_coefficients(model):
    from .logistic_regression_model import coefficients

    coef = coefficients(model)
    coef.to_csv(OUT / "logistic_regression_coefficients.csv")
    agg = coef.drop(columns="max_abs").abs().max(axis=1)
    return coef, _to_source(agg.values, agg.index).rename("lr_max_abs_coef")


def permutation_by_feature(models, test):
    res = {}
    for key, pipe in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = permutation_importance(pipe, test[FEATURES], test["y"], scoring="f1_macro", n_repeats=N_REPEATS,
                                       random_state=config.SEED, n_jobs=config.N_JOBS)
        res[f"{key}_perm_mean"] = pd.Series(r.importances_mean, index=FEATURES)
        res[f"{key}_perm_std"] = pd.Series(r.importances_std, index=FEATURES)
    return pd.DataFrame(res)


def permutation_by_group(models, test):
    rng = np.random.default_rng(config.SEED)
    X, y = test[FEATURES].reset_index(drop=True), test["y"].values
    rows = []
    for key, pipe in models.items():
        base = pipe.predict(X)
        base_f1 = f1_score(y, base, average="macro")
        base_rc = recall_score(y, base, labels=[2], average="macro", zero_division=0)
        for group, cols in FEATURE_GROUPS.items():
            d_f1, d_rc = [], []
            for _ in range(N_REPEATS):
                Xp = X.copy()
                perm = rng.permutation(len(X))
                Xp[cols] = X[cols].values[perm]
                p = pipe.predict(Xp)
                d_f1.append(base_f1 - f1_score(y, p, average="macro"))
                d_rc.append(base_rc - recall_score(y, p, labels=[2], average="macro", zero_division=0))
            rows.append({"model": key, "group": group, "n_features": len(cols),
                         "macro_f1_drop": np.mean(d_f1), "macro_f1_drop_std": np.std(d_f1),
                         "critical_recall_drop": np.mean(d_rc), "critical_recall_drop_std": np.std(d_rc)})
    return pd.DataFrame(rows)


def shap_values(models, test, labels):
    import shap

    out = {}
    for key in ("xgboost", "lightgbm"):
        pipe = models[key]
        names = pipe.named_steps["prep"].get_feature_names_out()
        Xt = pipe.named_steps["prep"].transform(test[FEATURES])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # -> (n, p, classes)
        for i, c in enumerate(config.CLASSES):
            out[f"{key}_shap_{c}"] = _to_source(np.abs(sv[:, :, i]).mean(0), names)
        # beeswarm for the Critical class on encoded features
        fig = plt.figure()
        shap.summary_plot(sv[:, :, 2], pd.DataFrame(Xt, columns=names), max_display=20, show=False, plot_size=(8, 7))
        plt.title(f"{labels[key]} - SHAP values for the Critical class (test period)", fontsize=11)
        plt.tight_layout()
        plt.savefig(config.RESULT_DIRS["figures"] / f"shap_critical_{key}.png", bbox_inches="tight", dpi=150)
        plt.close("all")
    return pd.DataFrame(out)


def _plot_groups(groups, labels):
    order = groups.groupby("group")["macro_f1_drop"].mean().sort_values().index
    keys = list(dict.fromkeys(groups["model"]))
    colors = ["#8a8985", "#2a78d6", "#1b9e77", "#7b3fb5"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    h = 0.8 / len(keys)
    for ax, metric, title in [(axes[0], "macro_f1_drop", "Drop in macro-F1"), (axes[1], "critical_recall_drop", "Drop in Critical recall")]:
        for j, key in enumerate(keys):
            g = groups[groups["model"] == key].set_index("group").reindex(order)
            ax.barh(np.arange(len(order)) + j * h - 0.4 + h / 2, g[metric], height=h, color=colors[j % len(colors)], label=labels[key])
        ax.axvline(0, color="#0b0b0b", lw=0.8)
        ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order])
        ax.set_title(f"{title} when the group is permuted (test)")
        ax.set_xlabel("drop (higher = model relies more on the group)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / "group_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)


def _plot_lr(coef):
    top = coef.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(top))
    for j, (c, col) in enumerate([("On-Time", "#2a78d6"), ("At-Risk", "#eda100"), ("Critical", "#e34948")]):
        ax.barh(y + (j - 1) * 0.27, top[c], height=0.27, color=col, label=c)
    ax.set_yticks(y, top.index, fontsize=8)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.set_xlabel("coefficient (log-odds per SD for numeric inputs)")
    ax.set_title("Logistic Regression: 20 largest coefficients by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIRS["figures"] / "logistic_regression_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def main(ctx, supervised_modules):
    test = ctx["test"]
    labels = {m.NAME: m.LABEL for m in supervised_modules}
    models = {m.NAME: joblib.load(config.MODELS / f"{m.NAME}.joblib") for m in supervised_modules}

    coef, lr_src = lr_coefficients(models["logistic_regression"])
    _plot_lr(coef)
    native = native_importances(models)
    perm = permutation_by_feature(models, test)
    shp = shap_values(models, test, labels)
    by_feature = pd.concat([lr_src, native, perm, shp], axis=1)
    by_feature.insert(0, "group", [FEATURE_TO_GROUP[c] for c in by_feature.index])
    by_feature.to_csv(OUT / "importance_by_feature.csv")

    groups = permutation_by_group(models, test)
    groups.to_csv(OUT / "group_permutation_importance.csv", index=False)
    _plot_groups(groups, labels)

    # share of each method's importance that falls in each group (permutation: positive drops only)
    share = {}
    for col in ["lr_max_abs_coef", "rf_impurity", "xgb_gain", "lgbm_gain", "xgboost_shap_Critical", "lightgbm_shap_Critical"] + \
            [f"{k}_perm_mean" for k in models]:
        v = by_feature[col].clip(lower=0)
        share[col] = v.groupby(by_feature["group"]).sum() / v.sum()
    pd.DataFrame(share).round(4).to_csv(OUT / "importance_share_by_group.csv")
    return by_feature, groups
