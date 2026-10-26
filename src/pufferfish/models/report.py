"""Figures for the review: what the data shows, and how the models compare against honest baselines."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, RED, GREY = "#0B3C5D", "#C00000", "#777777"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def monthly_table(df):
    t = df.groupby(df["date"].dt.month)["puffer"].agg(hauls="size", positives="sum").reindex(range(1, 13), fill_value=0)
    t["rate"] = t["positives"] / t["hauls"].replace(0, np.nan)
    ci = [wilson(int(k), int(n)) for k, n in zip(t["positives"], t["hauls"])]
    t["ci_low"], t["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return t


def plot_seasonal_signal(df, out):
    t = monthly_table(df)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=170)
    x = np.arange(1, 13)
    ax.bar(x, t["rate"].fillna(0) * 100, color=NAVY, width=0.65)
    ok = t["hauls"] > 0
    ax.errorbar(x[ok], t.loc[ok, "rate"] * 100, yerr=[np.clip((t.loc[ok, "rate"] - t.loc[ok, "ci_low"]) * 100, 0, None),
                np.clip((t.loc[ok, "ci_high"] - t.loc[ok, "rate"]) * 100, 0, None)], fmt="none", ecolor=GREY, capsize=3, lw=1)
    for xi, (k, n) in enumerate(zip(t["positives"], t["hauls"]), start=1):
        ax.text(xi, 2, f"{k}/{n}", ha="center", va="bottom", color="white" if k else GREY, fontsize=8)
    ax.axhline(df["puffer"].mean() * 100, color=RED, ls="--", lw=1.2, label=f"overall rate {df['puffer'].mean()*100:.0f}%")
    ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Hauls with pufferfish (%)"); ax.set_ylim(0, 105)
    ax.set_title("Pufferfish occurrence by month (bars: share of hauls; whiskers: 95% interval; labels: positive/hauls)", fontsize=9)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out); plt.close()


def plot_model_comparison(res, out):
    r = res[res["status"] == "ok"].copy()
    r = r.sort_values("chrono_average_precision")
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=170)
    y = np.arange(len(r))
    ap = r["chrono_average_precision"].to_numpy()
    err = [np.clip(ap - r["chrono_ap_ci_low"].to_numpy(), 0, None), np.clip(r["chrono_ap_ci_high"].to_numpy() - ap, 0, None)]
    colors = [GREY if "baseline" in m else NAVY for m in r["model"]]
    ax.barh(y, ap, color=colors, height=0.6, xerr=err, ecolor="black", capsize=3)
    ax.set_yticks(y, r["model"])
    ax.axvline(r["chrono_prevalence"].iloc[0], color=RED, ls="--", lw=1.2, label="no-skill level (share of positive hauls)")
    ax.set_xlabel("Average precision on LATER hauls (higher is better; whiskers: 95% interval)")
    ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out); plt.close()


def plot_learning_curve(lc, out):
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=170)
    for name, g in lc.groupby("model"):
        base = "baseline" in name
        ax.errorbar(g["train_hauls"], g["ap_mean"], yerr=g["ap_sd"], marker="o", capsize=3,
                    color=GREY if base else NAVY, label=name)
    ax.axhline(lc["no_skill_ap"].iloc[0], color=RED, ls="--", lw=1.2, label="no-skill level")
    ax.set_xlabel("Number of real training hauls (random subsets of the earlier hauls)")
    ax.set_ylabel("Average precision on the held-out LATER hauls")
    ax.set_title("Does more data help? (mean and spread over repeated subsets)", fontsize=9)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out); plt.close()
