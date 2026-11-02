"""Score every submission folder with the OFFICIAL closure_challenge package,
write results/scores.md, paste the table into README.md, and draw figures.

usage: python scripts/score.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from closure_challenge import evaluate_from_csv_by_case, score_from_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# folder name -> (label, bar colour, hatched?). Colour = model; hatching = SST fallback.
SUBS = {
    "baseline_rans": ("k-ω SST baseline (no ML)", "#a3a29b", False),
    "lightgbm": ("LightGBM", "#2a78d6", False),
    "lightgbm_ood": ("LightGBM + SST fallback", "#2a78d6", True),
    "mlp": ("MLP", "#eb6834", False),
    "mlp_ood": ("MLP + SST fallback", "#eb6834", True),
}
SUBS = {k: v for k, v in SUBS.items() if (ROOT / "submissions" / k).is_dir()}
PUBLISHED = {"Yang": 0.0580, "Reissmann, Fang, Sandberg": 0.0595, "Wu and Zhang": 0.0624,
             "Tian, Buchanan, Hickel, Dwight": 0.0641, "Liu, Wang, Zhao, Xiao": 0.0737,
             "Montoya, Oulghelou, Cinnella": 0.0779}

res = {k: (score_from_csv(ROOT / "submissions" / k),
           {c: float(s) for c, s in evaluate_from_csv_by_case(ROOT / "submissions" / k).items()})
       for k in SUBS}
cases = list(next(iter(res.values()))[1])

# ---------------- tables ----------------
md = ["| Model | **Overall** | " + " | ".join(cases) + " |", "|---" * (len(cases) + 2) + "|"]
for k, (tot, per) in res.items():
    md.append(f"| {SUBS[k][0]} | **{tot:.4f}** | " + " | ".join(f"{per[c]:.4f}" for c in cases) + " |")
board = sorted([(v, n) for n, v in PUBLISHED.items()]
               + [(res[k][0], SUBS[k][0] if k == "baseline_rans" else f"**{SUBS[k][0]} (this repo)**")
                  for k in res])
md += ["", "| Rank | Entry | Overall score |", "|---|---|---|"]
md += [f"| {i} | {n} | {s:.4f} |" for i, (s, n) in enumerate(board, 1)]
table = "\n".join(md)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/scores.md").write_text(table + "\n", encoding="utf-8")

readme = ROOT / "README.md"
txt = readme.read_text(encoding="utf-8")
a, b = "<!-- SCORES -->", "<!-- /SCORES -->"
if a in txt and b in txt:
    txt = txt[: txt.index(a) + len(a)] + "\n" + table + "\n" + txt[txt.index(b):]
    readme.write_text(txt, encoding="utf-8")
    print("README.md score table updated")
for k, (tot, _) in res.items():
    print(f"{k:16s} overall {tot:.4f}")

# ---------------- figure: per-case scores ----------------
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=BG)
ax.set_facecolor(BG)
xs, n = np.arange(len(cases)), len(res)
bw = 0.8 / n
for i, k in enumerate(res):
    ax.bar(xs + (i - (n - 1) / 2) * bw, [res[k][1][c] for c in cases], bw - 0.03,
           color=SUBS[k][1], hatch="////" if SUBS[k][2] else None, edgecolor=BG, linewidth=0,
           label=f"{SUBS[k][0]}  ({res[k][0]:.4f})", zorder=3)
ax.set_xticks(xs, [c.replace("alpha_", "α ").replace("_Ret_", " Reτ ").replace("_", " ")
                   for c in cases], rotation=20, ha="right")
ax.set_ylabel("scaled velocity MAE (lower is better)")
ax.grid(axis="y", color=GRID, zorder=0)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
ax.set_title("Closure Challenge test cases: official score per case", loc="left", color=INK)
for x in (3.5, 6.5):
    ax.axvline(x, color=GRID, lw=1)
for x, t in ((1.5, "periodic hills"), (5, "square/rect. duct"), (7, "NASA hump")):
    ax.text(x, ax.get_ylim()[1] * 0.97, t, ha="center", va="top", color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "results/test_scores.png", dpi=150)

# ---------------- figure: feature importance ----------------
jf = ROOT / "results/lightgbm_val.json"
if jf.exists():
    imp = json.loads(jf.read_text())["feature_importance_gain"]
    order = sorted(imp, key=imp.get)
    v = np.array([imp[k] for k in order])
    fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(order, v / v.sum() * 100, color="#2a78d6", height=0.7, zorder=3)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_xlabel("share of total split gain (%)")
    ax.set_title("LightGBM: which RANS features matter", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "results/feature_importance.png", dpi=150)
print("figures written to results/")
