"""
utils.py — Utility functions for MARCOTool.

Functions
---------
print_results       Pretty-print the estimator results dict.
multi_criteria      Fuzzy multi-criteria weight determination and ranking
                    (Statistical Variance, CRITIC, MEREC) with TOPSIS and
                    MABAC ranking methods.
snr_cal             Compute the SNR fraction (MARCOT vs. pseudoslit) and
                    propagate uncertainties.
tables              Write LaTeX tables for the decision matrix, weights,
                    defuzzified weights, and alternative ranking.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import quad
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_CSV     = Path("data/results_marcot.csv")
DECISION_CSV    = Path("data/decision_matrix.csv")
SCORE_CSV       = Path("data/score_total.csv")
FIGURES_DIR     = Path("Figures")

# Physical / observational constants used in SNR calculations
_HC             = 1.98644586e-25   # h·c  [J·m]
_L_INI          = 1.1e-6           # J-band start [m]
_L_FIN          = 1.4e-6           # J-band end   [m]
_F_OBS_J        = 1.277494916846618e-12  # Flux density of J02530+168 [W/m²/µm]
_T_EXP          = 100              # Exposure time [s]

# Colour codes for terminal messages
_GREEN  = "\033[1;4;32m"
_YELLOW = "\033[4;93;1m"
_RESET  = "\033[0m"

# Weight-method labels
_WEIGHT_METHODS = ["St. Variance", "CRITIC", "MEREC"]


def _log_saved(filename: str) -> None:
    print(f"{_GREEN}File '{filename}' was saved successfully{_RESET}")


# ---------------------------------------------------------------------------
# print_results
# ---------------------------------------------------------------------------

# Keywords that identify each output section
_SECTION_KEYS = {
    "TELESCOPE": [
        "OTA", "Seeing", "Input", "Module", "F-number",
        "Focal length", "Total cost tel", "Total cost for each",
        "Weight", "Reduction cost factor",
    ],
    "PHOTONIC LANTERN": [
        "Expected", "Modes", "Fibers", "Required output",
        "Selected", "Super-PL", "Total cost PLs",
    ],
    "SPECTROGRAPH": [
        "Beam diameter", "Spectrograph", "Estimated cost",
        "Magnification", "Resolution",
    ],
}


def print_results(results: dict) -> None:
    """Pretty-print the estimator results grouped by instrument section."""

    def _header(title: str) -> None:
        print(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")

    shown: list[str] = []
    for section, keywords in _SECTION_KEYS.items():
        _header(section)
        keys = [k for k in results if any(s in k for s in keywords)]
        for k in keys:
            print(f"{k:<45}: {results[k]}")
        shown.extend(keys)

    _header("AUXILIARY PARAMETERS")
    for k in results:
        if k not in shown and k != "":
            print(f"{k:<45}: {results[k]}")


# ---------------------------------------------------------------------------
# Helpers shared by multi_criteria
# ---------------------------------------------------------------------------

def _fuzzy_column(arr: np.ndarray) -> list:
    """Return a list of (l, m, u) numpy arrays from a stacked column."""
    return list(map(np.array, arr))


def _defuzzify(w: np.ndarray) -> np.ndarray:
    """Apply the (l + 4m + u) / 6 centroid defuzzification formula."""
    return (w[:, 0] + 4 * w[:, 1] + w[:, 2]) / 6


def _fuzzy_weight_from_columns(col_l, col_m, col_u) -> np.ndarray:
    """
    Build a sorted (n_criteria × 3) fuzzy weight matrix from three
    columns, normalised so that the defuzzified weights sum to 1.
    """
    w = np.column_stack([
        col_l / np.nansum(col_u),
        col_m / np.nansum(col_m),
        col_u / np.nansum(col_l),
    ])
    return np.sort(w, axis=1)


def _fuzzy_distance(ideal: float, weighted: np.ndarray) -> np.ndarray:
    """Euclidean distance between a scalar ideal value and fuzzy rows."""
    return np.sqrt(
        (1 / 3) * (
            (ideal - weighted[:, 0]) ** 2
            + (ideal - weighted[:, 1]) ** 2
            + (ideal - weighted[:, 2]) ** 2
        )
    )


def _criteria_column_labels(n: int) -> list[str]:
    """Return C1, C2, … Cn labels for LaTeX tables."""
    return [f"C{i + 1}" for i in range(n)]


# ---------------------------------------------------------------------------
# multi_criteria
# ---------------------------------------------------------------------------

def multi_criteria(results_csv_path, criteria: dict):
    """
    Compute fuzzy weights (Statistical Variance, CRITIC, MEREC) and rank
    alternatives with TOPSIS and MABAC.

    Parameters
    ----------
    results_csv_path : str or Path
        Path to the MARCOTool results TSV.
    criteria : dict
        Mapping of criterion name → {"type": "benefit" | "cost"}.

    Returns
    -------
    tuple
        (st_weight, st_weight_defu, crt_weight, crt_weight_defu,
         mrc_weight, mrc_weight_defu, criteria)
    """
    if results_csv_path is None:
        results_csv_path = RESULTS_CSV

    df = pd.read_csv(results_csv_path, sep="\t")

    # Split criteria into benefit (more-is-better) and cost (less-is-better)
    more = [c for c, v in criteria.items() if v["type"] == "benefit"]
    less = [c for c, v in criteria.items() if v["type"] != "benefit"]

    # ------------------------------------------------------------------
    # Build decision matrix
    # ------------------------------------------------------------------
    columns       = ["OTA diameter (m)"]       + list(criteria)
    uncer_columns = ["Uncer OTA diameter (m)"] + [f"Uncer {c}" for c in criteria]

    dm  = df[columns].set_index("OTA diameter (m)")
    udm = df[uncer_columns].set_index("Uncer OTA diameter (m)")

    n_alt, n_crit = dm.shape

    # ------------------------------------------------------------------
    # Normalise into fuzzy triplets (l, m, u) for every criterion
    # ------------------------------------------------------------------
    V   = []          # Statistical Variance accumulator
    C_l, C_m, C_u = [], [], []   # CRITIC accumulator

    for crit in criteria:
        x = dm[crit].to_numpy(dtype=float)
        u = udm[f"Uncer {crit}"].to_numpy(dtype=float)

        x_l, x_m, x_u = x - u, x, x + u
        x_max = x_u.max()
        x_min = x_l.min()
        denom = x_max - x_min

        if crit in more:
            l_st,  m_st,  u_st  = x_l / x_max, x_m / x_max, x_u / x_max
            l_crt, m_crt, u_crt = (x_l - x_min) / denom, (x_m - x_min) / denom, (x_u - x_min) / denom
            l_mrc, m_mrc, u_mrc = x_min / x_u, x_min / x_m, x_min / x_l
        else:
            l_st,  m_st,  u_st  = x_min / x_u, x_min / x_m, x_min / x_l
            l_crt, m_crt, u_crt = (x_max - x_u) / denom, (x_max - x_m) / denom, (x_max - x_l) / denom
            l_mrc, m_mrc, u_mrc = x_l / x_max, x_m / x_max, x_u / x_max

        dm[crit + "_fuzzy"]      = _fuzzy_column(np.column_stack([x_l,  x_m,  x_u]))
        dm[crit + "_norm_st"]    = _fuzzy_column(np.column_stack([l_st, m_st, u_st]))
        dm[crit + "_norm_CRITIC"]= _fuzzy_column(np.column_stack([l_crt, m_crt, u_crt]))
        dm[crit + "_norm_MEREC"] = _fuzzy_column(np.column_stack([l_mrc, m_mrc, u_mrc]))

    dm.to_csv(DECISION_CSV, sep="\t", index=False)
    _log_saved(DECISION_CSV.name)

    # ------------------------------------------------------------------
    # Statistical Variance weights
    # ------------------------------------------------------------------
    for crit in criteria:
        st_vals = dm[crit + "_norm_st"].values
        V.append((1 / n_alt) * sum((v - np.mean(st_vals)) ** 2 for v in st_vals))

        crt_vals = np.vstack(dm[crit + "_norm_CRITIC"].to_numpy())
        l_1, m_1, u_1 = crt_vals[:, 0].astype(float), crt_vals[:, 1].astype(float), crt_vals[:, 2].astype(float)

        corr_l, corr_m, corr_u = [], [], []
        for other in criteria:
            if other == crit:
                continue
            other_vals = np.vstack(dm[other + "_norm_CRITIC"].to_numpy())
            l_2 = other_vals[:, 0].astype(float)
            m_2 = other_vals[:, 1].astype(float)
            u_2 = other_vals[:, 2].astype(float)
            corr_l.append(pearsonr(l_1, l_2)[0])
            corr_m.append(pearsonr(m_1, m_2)[0])
            corr_u.append(pearsonr(u_1, u_2)[0])

        C_l.append(np.std(l_1) * np.nansum(1 - np.array(corr_l)))
        C_m.append(np.std(m_1) * np.nansum(1 - np.array(corr_m)))
        C_u.append(np.std(u_1) * np.nansum(1 - np.array(corr_u)))

    V = np.array(V)
    st_weight      = _fuzzy_weight_from_columns(V[:, 0], V[:, 1], V[:, 2])
    st_weight_defu = _defuzzify(st_weight)

    # ------------------------------------------------------------------
    # CRITIC weights
    # ------------------------------------------------------------------
    C = np.column_stack([C_l, C_m, C_u])
    crt_weight      = _fuzzy_weight_from_columns(C[:, 0], C[:, 1], C[:, 2])
    crt_weight_defu = _defuzzify(crt_weight)

    # ------------------------------------------------------------------
    # MEREC weights
    # ------------------------------------------------------------------
    mrc_all = np.array([dm[c + "_norm_MEREC"] for c in criteria])   # (n_crit, n_alt)

    S_component = []
    S_alternative = []    # will be (n_crit, n_alt) after transpose

    for i in range(n_alt):
        alt_vals = np.vstack(mrc_all[:, i])
        l, m, u = alt_vals[:, 0].astype(float), alt_vals[:, 1].astype(float), alt_vals[:, 2].astype(float)

        s_l = math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in l))
        s_m = math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in m))
        s_u = math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in u))
        S_component.append(np.array([s_l, s_m, s_u]))

        alt_row = []
        for j in range(n_crit):
            l_j = [v for k, v in enumerate(l) if k != j]
            m_j = [v for k, v in enumerate(m) if k != j]
            u_j = [v for k, v in enumerate(u) if k != j]
            alt_row.append(np.array([
                math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in l_j)),
                math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in m_j)),
                math.log(1 + (1 / n_crit) * sum(abs(math.log(v)) for v in u_j)),
            ]))
        S_alternative.append(np.array(alt_row))

    S_component   = np.array(S_component)          # (n_alt, 3)
    S_alternative = np.array(S_alternative)        # (n_alt, n_crit, 3)
    S_alternative = S_alternative.transpose(1, 0, 2)  # (n_crit, n_alt, 3)

    E_crit = np.array([
        np.sum(np.abs(S_alternative[j] - S_component), axis=0)
        for j in range(n_crit)
    ])                                              # (n_crit, 3)

    mrc_weight      = _fuzzy_weight_from_columns(E_crit[:, 0], E_crit[:, 1], E_crit[:, 2])
    mrc_weight_defu = _defuzzify(mrc_weight)

    # ------------------------------------------------------------------
    # Weighted normalised matrix for TOPSIS and MABAC
    # ------------------------------------------------------------------
    weights = {
        "st":  st_weight,
        "crt": crt_weight,
        "mrc": mrc_weight,
    }

    I = {}   # weighted matrices, shape (n_crit, n_alt, 3)
    for tag, w in weights.items():
        I[tag] = np.array([
            [dm[crit + "_norm_st"].values[i] * w[j]
             for i in range(n_alt)]
            for j, crit in enumerate(criteria)
        ])

    # ------------------------------------------------------------------
    # MABAC scores
    # ------------------------------------------------------------------
    for tag in weights:
        g_mat = np.prod(I[tag], axis=1) ** (1 / n_alt)          # (n_crit, 3)
        q_mat = I[tag] - g_mat[:, np.newaxis, :]                 # (n_crit, n_alt, 3)
        S     = q_mat.sum(axis=0)                                # (n_alt, 3)
        df[f"score_total_{tag}_MABAC"] = pd.Series(
            _defuzzify(S), index=range(n_alt)
        )

    # ------------------------------------------------------------------
    # TOPSIS scores
    # ------------------------------------------------------------------
    for tag in weights:
        A_star = I[tag][:, :, 2].max(axis=1)   # (n_crit,)
        A_less = I[tag][:, :, 0].min(axis=1)   # (n_crit,)

        d_star = np.array([
            _fuzzy_distance(A_star[j], I[tag][j]) for j in range(n_crit)
        ]).sum(axis=0)   # (n_alt,)

        d_less = np.array([
            _fuzzy_distance(A_less[j], I[tag][j]) for j in range(n_crit)
        ]).sum(axis=0)

        CC = d_less / (d_less + d_star)
        df[f"score_total_{tag}_TOPSIS"] = pd.Series(CC, index=range(CC.size))

    # ------------------------------------------------------------------
    # Save ranked results
    # ------------------------------------------------------------------
    df.to_csv(SCORE_CSV, sep="\t", index=False)

    for tag in weights:
        for method in ("TOPSIS", "MABAC"):
            col      = f"score_total_{tag}_{method}"
            out_path = Path(f"data/{col}.csv")
            df.sort_values(by=col, ascending=False).to_csv(out_path, sep="\t", index=False)

    _log_saved(SCORE_CSV.name)

    # ------------------------------------------------------------------
    # Find best alternative indices
    # ------------------------------------------------------------------
    best = {
        tag: int(np.argmax(df[f"score_total_{tag}_TOPSIS"].values))
        for tag in weights
    }

    print("\n\033[1m\033[4mBest configuration found:\033[0m\n")
    for col in criteria:
        val = df[col].values
        print(f"{col}: {val[best['st']]}")
    print(f"score_total_st_TOPSIS: {df['score_total_st_TOPSIS'].values[best['st']]:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_multi_criteria(df, criteria, best, st_weight_defu, crt_weight_defu, mrc_weight_defu)

    return (
        st_weight,  st_weight_defu,
        crt_weight, crt_weight_defu,
        mrc_weight, mrc_weight_defu,
        criteria,
    )


def _plot_scatter(ax, ota_mm, y_vals, n_ota, best_indices, ylabel: str):
    """Plot all alternatives as circles and best-fit ones as stars."""
    best_set = set(best_indices.values())
    for i in range(len(ota_mm)):
        if i not in best_set:
            ax.plot(ota_mm[i], y_vals[i], "o",
                    label=f"{int(n_ota[i])} OTAs of {int(ota_mm[i])} mm")

    labels = {"st": "St. Variance", "crt": "CRITIC", "mrc": "MEREC"}
    for tag, idx in best_indices.items():
        ax.plot(ota_mm[idx], y_vals[idx], "*", ms=20,
                label=f"Best {labels[tag]}: {int(n_ota[idx])} OTAs of {int(ota_mm[idx])} mm")

    ax.set_ylabel(ylabel)
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0)
    ax.legend(loc="upper right")


def _plot_multi_criteria(df, criteria, best, st_defu, crt_defu, mrc_defu):
    """Generate and save all diagnostic figures for the multi-criteria analysis."""
    ota_mm  = df["OTA diameter (m)"].values * 1000
    cost    = df["Total cost (MEUR)"].values
    n_ota   = df["Number of OTA for high efficiency"].values
    rec_mod = df["Recalculated module diameter (m)"].values
    out_core= df["Required output fiber core (microns)"].values
    cost_mod= df["Total cost for each module (MEUR)"].values

    plots = [
        ("Cost (MEUR) vs OTA diameter",         ota_mm,  cost,     "Cost (MEUR)",          "OTA diameter (mm)",    "Cost_vs_Aperture.png"),
        ("Module size (m) vs OTA diameter",      ota_mm,  rec_mod,  "Module aperture (m)",  "OTA diameter (mm)",    "Module_size_vs_Aperture.png"),
        ("PL core vs OTA diameter",              ota_mm,  out_core, "PL core (µm)",         "OTA diameter (mm)",    "Core_PL_vs_Aperture.png"),
    ]

    for title, x_data, y_data, ylabel, xlabel, fname in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_scatter(ax, x_data, y_data, n_ota, best, ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname)
        _log_saved(fname)
        plt.close()

    # Cost vs Module size (extra reference curve)
    fig, ax = plt.subplots(figsize=(8, 5))
    best_set = set(best.values())
    for i in range(len(ota_mm)):
        if i not in best_set:
            ax.plot(rec_mod[i], cost_mod[i], "o",
                    label=f"{int(n_ota[i])} OTAs of {int(ota_mm[i])} mm")
            ax.plot(rec_mod[i], 2.37 * rec_mod[i] ** 1.96, "o", mec="black",
                    label=f"Traditional cost for {rec_mod[i]:.1f} m module")
    labels = {"st": "St. Variance", "crt": "CRITIC", "mrc": "MEREC"}
    for tag, idx in best.items():
        ax.plot(rec_mod[idx], cost_mod[idx], "*", ms=20,
                label=f"Best {labels[tag]}: {int(n_ota[idx])} OTAs of {int(ota_mm[idx])} mm")
    ax.set_xlabel("Module diameter (m)")
    ax.set_ylabel("Cost (MEUR)")
    ax.set_title("Cost vs Module size")
    ax.grid(which="major", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Cost_vs_Module_size.png")
    _log_saved("Cost_vs_Module_size.png")
    plt.close()

    # Spider diagram
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})
    series        = [st_defu, crt_defu, mrc_defu]
    series_labels = ["F-St.Variance", "F-CRITIC", "F-MEREC"]
    colors        = ["tab:red", "tab:green", "tab:blue"]
    N      = len(criteria)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4"], fontsize=10)
    ax.grid(True, color="#bcbcbc", alpha=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(criteria), fontsize=11)

    for y, label, color in zip(series, series_labels, colors):
        y_closed = np.concatenate((y, [y[0]]))
        ax.plot(angles, y_closed, color=color, linewidth=2, label=label)
        ax.fill(angles, y_closed, color=color, alpha=0.15)
    ax.legend(loc=(1, 0.6))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Spider_Diagram_Weights.png")
    plt.savefig(FIGURES_DIR / "Spider_Diagram_Weights.pdf")
    _log_saved("Spider_Diagram_Weights.png")
    plt.close()

    # Correlation heat-map
    weight_df = pd.DataFrame({
        "F-St. Variance": st_defu,
        "F-CRITIC":       crt_defu,
        "F-MEREC":        mrc_defu,
    })
    fig, ax_hm = plt.subplots(figsize=(6.5, 5.8))
    sns.heatmap(
        weight_df.corr(method="pearson"),
        vmin=0.5, vmax=1.0,
        cmap="Greens",
        annot=True, fmt=".3f",
        square=True,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax_hm,
    )
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha="left")
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Correlation_Matrix_Weights.png")
    plt.savefig(FIGURES_DIR / "Correlation_Matrix_Weights.pdf")
    _log_saved("Correlation_Matrix_Weights.png")
    plt.close()


# ---------------------------------------------------------------------------
# snr_cal
# ---------------------------------------------------------------------------

def _flux_integral(l_ini: float, l_fin: float) -> float:
    """Compute ∫ λ dλ over the J band (simple analytic result)."""
    result, _ = quad(lambda x: x, l_ini, l_fin)
    return result


def _signal(module_diam_m, efficiency, QE, flux_integral) -> np.ndarray:
    """
    Compute the number of ADU electrons collected by the telescope.

    Parameters
    ----------
    module_diam_m : array-like
        Effective module diameter [m].
    efficiency : array-like
        End-to-end system throughput.
    QE : float
        Detector quantum efficiency.
    flux_integral : float
        Pre-computed ∫ λ F_obs dλ value [W/m²].
    """
    area = np.pi * (np.asarray(module_diam_m) / 2) ** 2
    return area * efficiency * _F_OBS_J * QE * flux_integral / _HC


def _n_detector(DARK, t_exp, g, R_noise, fiber_core_mm, pix_size_mm, n_modules) -> np.ndarray:
    """Compute detector noise electrons (dark + read)."""
    n_pix = n_modules * fiber_core_mm / pix_size_mm
    return n_pix * ((DARK / g) * t_exp + (R_noise / g) ** 2)


def _snr(N_adu, g, N_det) -> np.ndarray:
    """Compute SNR from signal and detector noise."""
    S = N_adu / g
    return S / np.sqrt(N_det + S)


def _sigma_snr(values: dict, sigmas: dict) -> np.ndarray:
    """
    Propagate uncertainties through the SNR formula using partial derivatives.

    Parameters
    ----------
    values / sigmas : dict
        Keys: 'ADU', 'g', 'd' (fiber core mm), 'p' (pixel mm),
              'D' (dark), 't' (exposure), 'R' (read noise).
    """
    A, g  = values["ADU"], float(values["g"])
    d, p  = values["d"], float(values["p"])
    D, t  = float(values["D"]), float(values["t"])
    R     = float(values["R"])

    N = A / g
    Y = (D * t) / g + (R ** 2) / (g ** 2)
    X = (d / p) * Y
    Q = X + N

    df_dN = (X + 0.5 * N) / (Q ** 1.5)
    df_dX = -0.5 * N / (Q ** 1.5)

    dN_dg  = -N / g
    dX_dg  = (d / p) * (-(D * t) / (g ** 2) - 2.0 * (R ** 2) / (g ** 3))

    partials = {
        "ADU": df_dN / g,
        "d":   df_dX * (X / d),
        "p":   df_dX * (-X / p),
        "D":   df_dX * ((d / p) * (t / g)),
        "t":   df_dX * ((d / p) * (D / g)),
        "R":   df_dX * ((d / p) * (2.0 * R / (g ** 2))),
        "g":   df_dN * dN_dg + df_dX * dX_dg,
    }

    var = sum((partials[k] * sigmas[k]) ** 2 for k in partials)
    return np.sqrt(var)


def snr_cal(
    archive,
    plate_scale: float,
    R_noise: float,
    g: float,
    pix_size: float,
    FoV: float,
    DARK: float,
    QE: float,
    slicer: bool,
    super_pl: bool,
) -> np.ndarray:
    """
    Compute the SNR fraction SNR_MARCOT / SNR_pseudoslit and save it to
    the results CSV.

    Parameters
    ----------
    archive : str or Path
        Path to the results TSV (currently unused; reads RESULTS_CSV directly).
    plate_scale : float
        Detector plate scale [µm/arcsec].
    R_noise : float
        Read noise [e⁻].
    g : float
        Gain [e⁻/ADU].
    pix_size : float
        Pixel size [µm/px].
    FoV : float
        Sky aperture [arcsec].
    DARK : float
        Dark current [e⁻/s × 10³].
    QE : float
        Quantum efficiency.
    slicer : bool
        Whether an image slicer is used (halves the effective fiber core).
    super_pl : bool
        Whether a super photonic lantern is used (currently unused).

    Returns
    -------
    np.ndarray
        SNR fraction array (one value per alternative).
    """
    df = pd.read_csv(RESULTS_CSV, sep="\t")

    # Convert units
    pix_size_mm  = pix_size  * 1e-3   # µm → mm
    DARK_e       = DARK      * 1e-3   # ×10³ e/s → e/s

    module_diam   = df["Module diameter (m)"].values
    efficiency    = df["Expected efficiency"].values
    fiber_out_mm  = df["Selected commercial output core (microns)"].values * 1e-3
    fiber_in_mm   = df["Selected commercial input core (microns)"].values  * 1e-3
    e_fiber_out   = df["Uncer Selected commercial output core (microns)"].values * 1e-3
    e_fiber_in    = df["Uncer Selected commercial input core (microns)"].values  * 1e-3
    n_modules     = df["Total modules"].values
    n_otas        = df["Number of OTAs"].values

    flux_int = _flux_integral(_L_INI, _L_FIN)

    N_adu = _signal(module_diam, efficiency, QE, flux_int) * _T_EXP
    e_N_adu = N_adu * np.sqrt(
        (0.1 / _T_EXP) ** 2
        + (0.001 / efficiency) ** 2
        + (2 * 0.1 / module_diam) ** 2
        + (0.01e-12 / _F_OBS_J) ** 2
        + ((2 * _L_FIN * 0.01e-6) ** 2 + (2 * _L_INI * 0.01e-6) ** 2) ** 2
          / (_L_FIN ** 2 - _L_INI ** 2) ** 2
    )

    factor = 0.5 if slicer else 1.0
    N_det_marcot = _n_detector(DARK_e, _T_EXP, g, R_noise, fiber_out_mm * factor, pix_size_mm, 1)
    N_det_pseu   = _n_detector(DARK_e, _T_EXP, g, R_noise, fiber_in_mm  * factor, pix_size_mm, n_modules * n_otas)

    SNR_MARCOT = _snr(N_adu, g, N_det_marcot)
    SNR_PSEU   = _snr(N_adu, g, N_det_pseu)

    values_m = {"ADU": N_adu, "g": g, "d": fiber_out_mm, "p": pix_size_mm, "D": DARK_e, "t": _T_EXP, "R": R_noise}
    sigmas_m = {"ADU": e_N_adu, "g": 0.1, "d": e_fiber_out, "p": 0.1e-3, "D": 0.1, "t": 0.1, "R": 0.1}
    values_p = {"ADU": N_adu, "g": g, "d": fiber_in_mm,  "p": pix_size_mm, "D": DARK_e, "t": _T_EXP, "R": R_noise}
    sigmas_p = {"ADU": e_N_adu, "g": 0.1, "d": e_fiber_in,  "p": 0.1e-3, "D": 0.1, "t": 0.1, "R": 0.1}

    snr_fraction  = SNR_MARCOT / SNR_PSEU
    e_snr_fraction = snr_fraction * np.sqrt(
        (_sigma_snr(values_m, sigmas_m) / SNR_MARCOT) ** 2
        + (_sigma_snr(values_p, sigmas_p) / SNR_PSEU) ** 2
    )

    df["SNR fraction"]       = pd.Series(snr_fraction,   index=range(snr_fraction.size))
    df["Uncer SNR fraction"] = pd.Series(e_snr_fraction, index=range(e_snr_fraction.size))
    df.to_csv(RESULTS_CSV, sep="\t", index=False)

    return snr_fraction


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _make_criteria_labels(n: int) -> list[str]:
    return _criteria_column_labels(n)


def _latex_table(
    header_line: str,
    body_lines: list[str],
    col_spec: str,
    caption: str,
    label: str,
    centering: bool = True,
    double_toprule: bool = False,
) -> str:
    """Return a complete LaTeX table* string."""
    lines = [r"\begin{table*}"]
    if centering:
        lines.append(r"  \centering")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"      \toprule")
    if double_toprule:
        lines.append(r"      \toprule")
    lines.append("       " + header_line)
    lines.append(r"      \midrule")
    lines += ["       " + bl for bl in body_lines]
    lines.append(r"      \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def tables(criteria: dict) -> None:
    """
    Write four LaTeX tables to the data/ directory:

    - ``table_dm.txt``          Decision matrix (fuzzy values).
    - ``table_weight.txt``      Fuzzy weights per method.
    - ``table_weight_defu.txt`` Defuzzified weights (%) per method.
    - ``table_ranking.txt``     TOPSIS and MABAC rankings side-by-side.
    """
    (
        st_weight, st_weight_defu,
        crt_weight, crt_weight_defu,
        mrc_weight, mrc_weight_defu,
        criteria,
    ) = multi_criteria(RESULTS_CSV, criteria)

    df = pd.read_csv(DECISION_CSV, sep="\t", header=0, index_col=0)
    col_labels = _make_criteria_labels(len(criteria))

    # ------------------------------------------------------------------
    # Decision matrix table
    # ------------------------------------------------------------------
    fuzzy_cols = [df[c + "_fuzzy"].values for c in criteria]
    header = " & ".join(rf"\textbf{{{c}}}" for c in col_labels) + r" \\"
    body   = [
        " & ".join(str(fuzzy_cols[j][i]) for j in range(len(criteria))) + r" \\"
        for i in range(len(fuzzy_cols[0]))
    ]
    Path("data/table_dm.txt").write_text(
        _latex_table(header, body, "l" + "r" * len(criteria),
                     "Decision matrix of configurations",
                     "tab:decision_matrix_inline",
                     centering=False),
        encoding="utf-8",
    )
    _log_saved("table_dm.txt")

    # ------------------------------------------------------------------
    # Weight tables (fuzzy and defuzzified)
    # ------------------------------------------------------------------
    weight_configs = [
        (
            "table_weight.txt",
            [np.round(st_weight, 3), np.round(crt_weight, 3), np.round(mrc_weight, 3)],
            _WEIGHT_METHODS,
            "Fuzzy weights determined by methodology.",
            "tab:all_weights",
            False,
        ),
        (
            "table_weight_defu.txt",
            [np.round(st_weight_defu * 100, 3), np.round(crt_weight_defu * 100, 3), np.round(mrc_weight_defu * 100, 3)],
            [r"St. Variance ($\%$)", r"CRITIC ($\%$)", r"MEREC ($\%$)"],
            "Defuzzified weights determined by methodology.",
            "tab:all_weights_defu",
            False,
        ),
    ]

    for fname, weight_list, method_labels, caption, label, _ in weight_configs:
        weights_arr = np.array(weight_list)
        header = (
            r"\textbf{Criterion} & "
            + " & ".join(rf"\textbf{{{m}}}" for m in method_labels)
            + r" \\"
        )
        body = [
            f"{c} & " + " & ".join(str(weights_arr[k, i]) for k in range(3)) + r" \\"
            for i, c in enumerate(col_labels)
        ]
        Path(f"data/{fname}").write_text(
            _latex_table(header, body, "l" + "c" * (len(method_labels) + 1),
                         caption, label, double_toprule=True),
            encoding="utf-8",
        )
        _log_saved(fname)

    # ------------------------------------------------------------------
    # Ranking table (TOPSIS + MABAC side by side)
    # ------------------------------------------------------------------
    load = lambda tag, method: pd.read_csv(
        f"data/score_total_{tag}_{method}.csv", sep="\t", header=0
    )["Alternative"].values

    topsis_cols = [load(t, "TOPSIS") for t in ("st", "crt", "mrc")]
    mabac_cols  = [load(t, "MABAC")  for t in ("st", "crt", "mrc")]
    n_rows      = len(topsis_cols[0])

    header_1 = (
        r"\multicolumn{4}{c|}{\textbf{TOPSIS}}"
        " & "
        r"\multicolumn{4}{c}{\textbf{MABAC}}"
        r"\\"
    )
    header_2 = (
        " & ".join(rf"\textbf{{{m}}}" for m in _WEIGHT_METHODS)
        + " & "
        + " & ".join(rf"\textbf{{{m}}}" for m in _WEIGHT_METHODS)
        + r" \\"
    )
    body = [
        " & ".join(str(c[i]) for c in topsis_cols)
        + " & "
        + " & ".join(str(c[i]) for c in mabac_cols)
        + r" \\"
        for i in range(n_rows)
    ]

    n = len(_WEIGHT_METHODS)
    col_spec = "c" * n + "|" + "c" * n
    table_str = (
        r"\begin{table*}" + "\n"
        + f"\\begin{{tabular}}{{{col_spec}}}\n"
        + r"\toprule" + "\n"
        + r"\toprule" + "\n"
        + header_1 + "\n"
        + r"\midrule" + "\n"
        + header_2 + "\n"
        + r"\midrule" + "\n"
        + "\n".join(body) + "\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\caption{Ranking of the alternatives depending on weight-determination method and ranking method.}" + "\n"
        + r"\label{tab:all_rankings}" + "\n"
        + r"\end{table*}" + "\n"
    )
    Path("data/table_ranking.txt").write_text(table_str, encoding="utf-8")
    _log_saved("table_ranking.txt")
