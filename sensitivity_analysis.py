import re
import numpy as np
import pandas as pd
from src.utils import multi_criteria
import matplotlib.pyplot as plt
from pathlib import Path



DATA_FILE = "data/decision_matrix.csv"

df = pd.read_csv("data/results_marcot.csv", sep='\t')

alt_names = df["Alternative"].values

criteria = [
            ("Reduction cost factor", "benefit"),
            ("Weight supported by the mount (kg)", "cost"),
            ("Selected commercial output core (microns)", "cost"),
            ("Expected efficiency", "benefit"),
            ("Resolution with commercial fibers", "benefit"),
            ("S/N Ratio fraction", "benefit"),
            ("Number of OTA for high efficiency", "cost")
        ]
        
criteria = {
            name: {"type": kind}
            for name, kind in criteria
        }
        
criteria_dict = criteria
        
crit_names = list(criteria.keys())
crit_types = [criteria[n]["type"] for n in crit_names]

# (change to "_norm_CRITIC" o "_norm_MEREC" for testing)
NORM_SUFFIX = "_norm_st"
st_weight, w_var, crt_weight, w_critic, mrc_weight, w_merec, criteria = multi_criteria('data/results_marcot.csv', criteria)

# Monte Carlo
KAPPAS = [10, 50, 150]  # low, medium, high concentration
N = 30000               # nº test
RNG_SEED = 123

EXPORT_EXCEL = True
OUT_XLSX = "smaa_outputs.xlsx"
SAVE_FIG = True
OUT_PNG = "Figures/plot_sensitivity_analysis.png"
OUT_PDF = "Figures/plot_sensitivity_analysis.pdf"

KAPPA_FOR_POINTS = 50

_triplet_re = re.compile(r"\[\s*([\d\.eE\-\+]+)\s+([\d\.eE\-\+]+)\s+([\d\.eE\-\+]+)\s*\]")

def parse_triplet_series(series: pd.Series):
    lows, meds, highs = [], [], []
    for cell in series:
        if isinstance(cell, str):
            m = _triplet_re.search(cell)
            if m:
                a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
            else:
                a = b = c = np.nan
        elif isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 3:
            a, b, c = map(float, cell)
        else:
            a = b = c = np.nan
        lows.append(a); meds.append(b); highs.append(c)
    return np.array(lows, float), np.array(meds, float), np.array(highs, float)

def norm1(v):
    v = np.asarray(v, float)
    s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)


def sample_triangular(A, M, C, N, rng: np.random.Generator):
    eps = 1e-12
    A3 = A[None, :, :]
    M3 = M[None, :, :]
    C3 = C[None, :, :]
    U  = rng.random((N, A.shape[0], A.shape[1]))
    Fm = (M3 - A3) / (C3 - A3 + eps)
    left = U <= Fm
    X_left  = A3 + np.sqrt(U * (M3 - A3) * (C3 - A3 + eps))
    X_right = C3 - np.sqrt((1 - U) * (C3 - M3) * (C3 - A3 + eps))
    X = np.where(left, X_left, X_right)
    eq = (np.abs(A - M) < eps) & (np.abs(M - C) < eps)
    if np.any(eq):
        X = np.where(eq[None, :, :], M3, X)
    return X

def ranks_from_scores(S):
    order = np.argsort(-S, axis=1)
    ranks = np.empty_like(order)
    for i in range(S.shape[0]):
        ranks[i, order[i]] = np.arange(1, S.shape[1] + 1)
    return ranks

def topsis_scores(Xs, Ws):
    norm = np.sqrt(np.sum(Xs**2, axis=1))
    norm[norm == 0] = 1.0
    R = Xs / norm[:, None, :]
    V = R * Ws[:, None, :]
    v_plus  = V.max(axis=1)
    v_minus = V.min(axis=1)
    d_plus  = np.sqrt(np.sum((V - v_plus[:, None, :])**2, axis=2))
    d_minus = np.sqrt(np.sum((V - v_minus[:, None, :])**2, axis=2))
    return d_minus / (d_plus + d_minus + 1e-12)
    
def mabac_scores(Xs, Ws):
    V = Xs * Ws[:, None, :]
    eps = 1e-12
    G = np.exp(np.mean(np.log(np.clip(V, eps, None)), axis=1))
    Q = V - G[:, None, :]
    return np.sum(Q, axis=2)

def summarize_scores(S, ranks):
    n_alt = S.shape[1]
    prob_win = (ranks == 1).mean(axis=0)
    rank_exp = ranks.mean(axis=0)
    rank_acc = np.zeros((n_alt, n_alt))
    for r in range(1, n_alt + 1):
        rank_acc[:, r-1] = (ranks == r).mean(axis=0)
    pairwise = np.zeros((n_alt, n_alt))
    for a in range(n_alt):
        for b in range(n_alt):
            if a != b:
                pairwise[a, b] = np.mean(S[:, a] > S[:, b])
    return prob_win, rank_exp, rank_acc, pairwise

def build_weights_table(criteria_dict, w_var_n, w_critic_n, w_merec_n, w0_consensus):
    crit_names = list(criteria_dict.keys())
    crit_types = [criteria_dict[n]["type"] for n in crit_names]

    L = len(crit_names)
    for name, vec in [("w_var_n", w_var_n), ("w_critic_n", w_critic_n),
                      ("w_merec_n", w_merec_n),
                      ("w0_consensus", w0_consensus)]:
        assert len(vec) == L, f"Length of {name} != nº criteria ({len(vec)} != {L})"

    df = pd.DataFrame(
        {
            "Type": crit_types,
            "w_var": w_var_n,
            "w_critic": w_critic_n,
            "w_merec": w_merec_n,
            "w0_consenso": w0_consensus,
        },
        index=pd.Index(crit_names, name="Criterio")
    ).reset_index()

    return df

    
def make_paper_like_figure(A, M, C, crit_names, w0_consensus,
                           Ws_by_kappa, closeness_by_kappa, ranks_top_by_kappa, ranks_mab_by_kappa,
                           kappa_list, alt_names,
                           kappa_for_points=50,
                           save_png=True, save_pdf=True,
                           png_path="Figures/plot_sensitivity_analysis.png", pdf_path="Figures/plot_sensitivity_analysis.pdf",
                           png_path_2="Data/summary_sensitivity_analysis.png", pdf_path_2="Data/summary_sensitivity_analysis.pdf"):

    stats_rows = []
    for j, cname in enumerate(crit_names):
        row_ref = [f'C{j+1}', "Ref. Weight"]
        row_full = ["", "Full range"]
        row_q75  = ["", "75% range"]
        row_cv   = ["", "Mean % dev."]
        for kappa in kappa_list:
            Ws = Ws_by_kappa[kappa]
            col = Ws[:, j]
            q_lo, q_hi = np.quantile(col, [0.125, 0.875])
            wmin, wmax = float(col.min()), float(col.max())
            mu, sd = float(col.mean()), float(col.std(ddof=0))
            cv_pct = 100.0 * (sd / mu if mu > 0 else np.nan)
            row_ref.append(f"{w0_consensus[j]:.4f}")
            row_full.append(f"[{wmin:.4f}, {wmax:.4f}]")
            row_q75.append(f"[{q_lo:.4f}, {q_hi:.4f}]")
            row_cv.append(f"{cv_pct:.2f}%")
        stats_rows.extend([row_ref, row_full, row_q75, row_cv, ["", "", *["" for _ in kappa_list]]])

    table_cols = ["Criterion", "Sim. Parameter"] + [f"\u03BA = {k}" for k in kappa_list]

    if kappa_for_points not in kappa_list:
        raise ValueError("kappa_for_points must be in kappa_list")

    def avg_rank_stats_for_kappa(k):
        r_top = ranks_top_by_kappa[k]
        r_mab = ranks_mab_by_kappa[k]
        avg_ranks = 0.5 * (r_top + r_mab)
        mean_r = avg_ranks.mean(axis=0)
        std_r  = avg_ranks.std(axis=0)
        return mean_r, std_r

    mean_r_k, std_r_k = avg_rank_stats_for_kappa(kappa_for_points)

    order = np.argsort(mean_r_k)
    x = np.arange(len(order))
    alt_sorted = [alt_names[i] for i in order]
    mean_r_sorted = mean_r_k[order]
    std_r_sorted  = std_r_k[order]

    perf_rel_by_kappa = {}
    for kappa in kappa_list:
        S_top = closeness_by_kappa[kappa]
        mean_perf = S_top.mean(axis=0)
        mn, mx = float(mean_perf.min()), float(mean_perf.max())
        if mx - mn > 0:
            perf_rel = (mean_perf - mn) / (mx - mn)
        else:
            perf_rel = np.zeros_like(mean_perf)
        perf_rel_by_kappa[kappa] = perf_rel[order]
        
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    ax.errorbar(x, mean_r_sorted, yerr=std_r_sorted, fmt='o', color='#1f77b4',
                ecolor='#1f77b4', elinewidth=1, capsize=3, label=f"Average ranking (TOPSIS+MABAC), \u03BA={kappa_for_points}")
    ax.set_ylabel("Average ranking (2 MCDM methods) ± Standard deviation", fontsize=12)
    
    ax.set_yticks([2, 4, 6, 8, 10, 12])
    ax.set_yticklabels(['2nd', '4th', '6th', '8th', '10th', '12th'])
    ax.set_xticks(x)
    ax.invert_yaxis()
    ax.set_xticklabels(alt_sorted, rotation=65, ha="right")

    ax2 = ax.twinx()
    colors = {kappa_list[0]:"#d62728", kappa_list[1]:"#ff7f0e", kappa_list[2]:"#8c564b"}
    lstyle = {kappa_list[0]:"-", kappa_list[1]:"--", kappa_list[2]:":"}
    for kappa in kappa_list:
        ax2.plot(x, perf_rel_by_kappa[kappa], color=colors[kappa], lw=2, ls=lstyle[kappa],
                 label=f"Relative average performance, \u03BA={kappa}")
    ax2.set_ylabel("Relative average performance", fontsize=16)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=14, frameon=True)

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    if save_png:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ax.axis("off")
    table = ax.table(cellText=stats_rows, colLabels=table_cols,
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    ax.set_title("Simulation's random weight distribution parameters for different \u03BA values",
                 fontsize=14, pad=8)

    if save_png:
        fig.savefig(png_path_2, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(pdf_path_2, dpi=300, bbox_inches="tight")
    plt.close(fig)

        
def main():
    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv(DATA_FILE, sep="\t")
    crit_names = list(criteria.keys())
    A_list, M_list, C_list = [], [], []
    for name in crit_names:
        col = f"{name}{NORM_SUFFIX}"
        if col not in df.columns:
            raise KeyError(f"Column '{col}' doesn't found. Review NORM_SUFFIX/criterion name.")
        a, m, c = parse_triplet_series(df[col])
        A_list.append(a); M_list.append(m); C_list.append(c)
    A = np.column_stack(A_list)
    M = np.column_stack(M_list)
    C = np.column_stack(C_list)
    n_alt, n_crit = A.shape

    w_var_n    = norm1(w_var)
    w_critic_n = norm1(w_critic)
    w_merec_n  = norm1(w_merec)
    w0_consensus = norm1((w_var_n + w_critic_n + w_merec_n) / 3.0)

    crit_types = [criteria[name]["type"] for name in crit_names]

    # ── DIAGNÓSTICO ───────────────────────────────────────────────────────────
    print("\n=== Weights by method ===")
    print(f"{'Criterio':<45} {'w_var':>8} {'w_critic':>8} {'w_merec':>8} {'w0':>8}")
    for j, name in enumerate(crit_names):
        print(f"  {name:<43} {w_var_n[j]:8.4f} {w_critic_n[j]:8.4f} "
              f"{w_merec_n[j]:8.4f} {w0_consensus[j]:8.4f}")
    print(f"  {'SUMA':<43} {w_var_n.sum():8.4f} {w_critic_n.sum():8.4f} "
          f"{w_merec_n.sum():8.4f} {w0_consensus.sum():8.4f}")

    print("\n=== Deterministec score with w0_consensus ===")
    det_score_topsis = topsis_scores(
        M[np.newaxis, :, :],
        w0_consensus[np.newaxis, :]
    )
    order_det = np.argsort(-det_score_topsis[0])
    print(f"{'Rank':<6} {'Alternative':<15} {'TOPSIS_score':>12}")
    for rank, idx in enumerate(order_det, 1):
        print(f"  {rank:<4} {alt_names[idx]:<15} {det_score_topsis[0][idx]:12.4f}")
    # ── fin diagnóstico ───────────────────────────────────────────────────────

    all_tables = {}
    Ws_by_kappa         = {}   # ← ya estaba
    closeness_by_kappa  = {}   # ← AÑADIDO: faltaba
    ranks_top_by_kappa  = {}   # ← AÑADIDO: faltaba
    ranks_mab_by_kappa  = {}   # ← AÑADIDO: faltaba

    for kappa in KAPPAS:
        alpha = np.clip(kappa * w0_consensus, 1e-12, None)
        Ws = rng.dirichlet(alpha, size=N)
        Xs = sample_triangular(A, M, C, N, rng)

        S_top = topsis_scores(Xs, Ws)
        r_top = ranks_from_scores(S_top)

        S_mab = mabac_scores(Xs, Ws)
        r_mab = ranks_from_scores(S_mab)

        Ws_by_kappa[kappa]        = Ws
        closeness_by_kappa[kappa] = S_top
        ranks_top_by_kappa[kappa] = r_top
        ranks_mab_by_kappa[kappa] = r_mab

        def build_summary_df(S, ranks):
            prob = (ranks == 1).mean(axis=0)
            rank_exp = ranks.mean(axis=0)
            dfres = pd.DataFrame({
                "Alternative": alt_names,
                "Win_prob": prob,
                "Expected_rank": rank_exp
            })
            for r in range(1, n_alt+1):
                dfres[f"b_rank{r}"] = (ranks == r).mean(axis=0)
            return dfres.sort_values("Win_prob", ascending=False).reset_index(drop=True)

        all_tables[f"TOPSIS_summary_kappa_{kappa}"]  = build_summary_df(S_top, r_top)
        all_tables[f"MABAC_summary_kappa_{kappa}"]   = build_summary_df(S_mab, r_mab)

        print(f"\n=== TOPSIS (kappa={kappa}) - Top 5 for Win_prob ===")
        print(all_tables[f"TOPSIS_summary_kappa_{kappa}"].loc[:4, ["Alternative", "Win_prob", "Expected_rank"]].to_string(index=False))
        print(f"\n=== MABAC (kappa={kappa}) - Top 5 for Win_prob ===")
        print(all_tables[f"MABAC_summary_kappa_{kappa}"].loc[:4, ["Alternative", "Win_prob", "Expected_rank"]].to_string(index=False))

    weights_table = build_weights_table(criteria, w_var_n, w_critic_n, w_merec_n, w0_consensus)
    all_tables["Weights_by_method_and_consensus"] = weights_table

    if EXPORT_EXCEL:
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
            for name, table in all_tables.items():
                sheet = name[:31]
                table.to_excel(writer, sheet_name=sheet, index=True)
        print(f"\nExported file: {OUT_XLSX}")

    make_paper_like_figure(A, M, C, crit_names, w0_consensus,
                           Ws_by_kappa, closeness_by_kappa, ranks_top_by_kappa, ranks_mab_by_kappa,
                           KAPPAS, alt_names,
                           kappa_for_points=KAPPA_FOR_POINTS,
                           save_png=SAVE_FIG, save_pdf=SAVE_FIG,
                           png_path=OUT_PNG, pdf_path=OUT_PDF)
    if SAVE_FIG:
        print(f"Figures saved as: {OUT_PNG} and {OUT_PDF}")

if __name__ == "__main__":
    main()
