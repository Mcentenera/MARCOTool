import pandas as pd
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import ast
import re

def print_results(results):
    def print_section(title):
        print(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")

    print_section("TELESCOPE")
    keys_tel = [k for k in results if any(s in k for s in ["OTA", "Seeing", "Input", "Module", "F-number", "Focal length", "Total cost tel", "Total cost for each", "Weight", "Reduction cost factor"])]
    for k in keys_tel:
        print(f"{k:<45}: {results[k]}")

    print_section("PHOTONIC LANTERN")
    keys_pl = [k for k in results if any(s in k for s in ["Expected", "Modes", "Fibers", "Required output", "Selected", "Super-PL", "Total cost PLs"])]
    for k in keys_pl:
        print(f"{k:<45}: {results[k]}")

    print_section("SPECTROGRAPH")
    keys_instr = [k for k in results if any(s in k for s in ["Beam diameter", "Spectrograph", "Estimated cost", "Magnification", "Resolution"])]
    for k in keys_instr:
        print(f"{k:<45}: {results[k]}")

    print_section("AUXILIAR PARAMETERS")
    other_keys = [k for k in results if k not in keys_tel + keys_pl + keys_instr and k != ""]
    for k in other_keys:
        print(f"{k:<45}: {results[k]}")

def multi_criteria(archive):
    df = pd.read_csv('data/results_marcot.csv', sep='\t')

    criterios = {
        "Expected efficiency (%)": 0.9,
        "Cost telescope + PL (MEUR)": 0.5,
        "Total modules": 0,
        "Resolution with commercial fibers": 0.8,
        "Number of OTA for high efficiency": 0.2,
        "Spectrograph volume (m³)": 0.5,
        "SNR fraction": 0.9,
        "OTA diameter (m)": 0,
        "Recalculated module diameter (m)": 0,
        "Recalculated telescope aperture (m)": 0
    }

    more = ["Expected efficiency (%)", "Resolution with commercial fibers", "SNR fraction", "Recalculated telescope aperture (m)"]
    less = ["Cost telescope + PL (MEUR)", "Total modules" ,"Number of OTAs", "Spectrograph volume (m³)"]

    # Just necesary columns
    missing_cols = [col for col in criterios if col not in df.columns]
    if missing_cols:
        print(f'\033[4;93;1mWARNING: Missing columns: {missing_cols}\033[0m')
        return None

    # Pass from list to array
    list_columns = [col for col in df.columns if df[col].dtype == object and df[col].astype(str).str.startswith('[').any()]
    
    for col in list_columns:
        try:
            def parse_array(x):
                if not isinstance(x, str) or not re.search(r'\d', x):
                    return x
                x = x.strip("[] \n\t")
                return np.fromstring(x, sep=' ')
            df[col] = df[col].apply(parse_array)
        except Exception as e:
            print(f'\033[4;93;1mWARNING: Error in column {col}: {str(e)}\033[0m')
            return None

    for criterio in criterios:
        col_data = df[criterio]

    # Check if we have any array at the column
        has_array = col_data.apply(lambda x: isinstance(x, (list, np.ndarray))).any()

        if has_array:
        # Flatten arrays to calculate global min and max
            try:
                all_values = np.concatenate(col_data.apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x])))
                vmin, vmax = np.min(all_values), np.max(all_values)

                if vmax == vmin:
                    print(f"\033[4;93;1mWARNING: The criteria '{criterio}' has constant values. Normalization not applicable\033[0m")
                    df[criterio + "_norm"] = col_data.apply(lambda x: np.zeros_like(x) if isinstance(x, (list, np.ndarray)) else 0)
                elif criterio in more:
                    df[criterio + "_norm"] = col_data.apply(lambda x: (np.array(x) - vmin) / (vmax - vmin) if isinstance(x, (list, np.ndarray)) else (x - vmin) / (vmax - vmin))
                else:
                    df[criterio + "_norm"] = col_data.apply(lambda x: (vmax - np.array(x)) / (vmax - vmin) if isinstance(x, (list, np.ndarray)) else (vmax - x) / (vmax - vmin))
            except Exception as e:
                print(f"\033[4;93;1mWARNING: Error normalizing {criterio}: {e}\033[0m")
                df[criterio + "_norm"] = 0

        else:
        # Scalar values only
            vmin, vmax = col_data.min(), col_data.max()

            if vmax == vmin:
                print(f"\033[4;93;1mWARNING: The criteria '{criterio}' has constant values. Normalization not applicable\033[0m")
                df[criterio + "_norm"] = 0
            elif criterio in more:
                df[criterio + "_norm"] = (col_data - vmin) / (vmax - vmin)
            else:
                df[criterio + "_norm"] = (vmax - col_data) / (vmax - vmin)

    # Calculate score total
    df["score_total"] = df[[c + "_norm" for c in criterios]].apply(
    lambda row: sum(row[c + "_norm"] * criterios[c] for c in criterios),
    axis=1)
    
    
    df_sorted = df.sort_values(by="score_total", ascending=False)
    
    df_sorted.to_csv("data/score_total.csv", sep = '\t', index = False)
    print("\033[1;4;32mFile 'score_total.csv' was saved successfully\033[0m")
    
    scores = df["score_total"].iloc[0]

    best_index = np.argmax(scores)

    print("\n\033[1m\033[4mBest configuration found:\033[0m\n")
    
#########################################################################
# PLOTS
#########################################################################
    ota_diam_mm = (df["OTA diameter (m)"].iloc[0]) * 1000
    cost = df["Total cost (MEUR)"].iloc[0]
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        
    for i in range(0, len(cost)):
    
        if i == best_index:
            pass
        else:
            ax.plot(ota_diam_mm[i], cost[i], 'o', label = f'{ota_diam_mm[i]} mm')
            
    ax.plot(ota_diam_mm[best_index], cost[best_index], 'b*', ms = 20, label = f'Best fit: {ota_diam_mm[best_index]}')
    plt.title('Cost (MEUR) vs OTA\'s diameter')
    plt.xlabel('OTA\'s diameter (mm)')
    plt.ylabel('Cost (MEUR)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="upper right")
    
    plt.savefig("Figures/Cost_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Cost_vs_Aperture.png' was saved successfully\033[0m")

    plt.close()
    
    for col in criterios.keys():
        val = df[col].iloc[0]
        if isinstance(val, (list, np.ndarray)):
            print(f"{col}: {val[best_index]}")
        else:
            print(f"{col}: {val}")

    print(f"score_total: {scores[best_index]}")

    
def snr_cal(archive):
    # Detector paremeters (CARMENES-VIS)
    df = pd.read_csv('data/results_marcot.csv', sep = '\t')
    
    try:
        def parse_array(x):
            if not isinstance(x, str) or not re.search(r'\d', x):
                return x
            x = x.strip("[] \n\t")
            return np.fromstring(x, sep=' ')
        df['Selected commercial output core (microns)'] = df['Selected commercial output core (microns)'].apply(parse_array)
    except Exception as e:
        print(f'\033[4;93;1mWARNING: Error in column {col}: {str(e)}\033[0m')
        return None
    
    module_diameter_m = df['Module diameter (m)'].iloc[0]
    efi_sys_MARCOT = 0.9
    # CAMBIAR A UN VALOR MEJOR CUANDO EL stimator.py CALCULE AL EFICIENCIA DLE SISTEMA DE MARCOT
    fiber_core_mm_MARCOT = (df['Selected commercial output core (microns)'].iloc[0]) * 1e-3
    # CUIDADO QUE EL VALOR QUE COJO DE df ESTÁ EN um Y NO EN mm
    
    t_exp = 100 # Exposure time [s]
    plate_scale = 0.169 # Plate scale [mm/"]
    R_noise = 5 # Read noise [e-]
    g = 1 # Ganancia [e-/ADU]
    pix_size = 15e-3 # Pixel size [mm]
    FoV = 1.5 # FoV of the fibre ["]
    DARK = 3 # DARK current [e-/s]
    QE = 0.92 # Quatum efficiency


    def N_det(DARK, t_exp, g, R_noise, fiber_core_mm):
        n_pix = fiber_core_mm / pix_size
        return n_pix * ((DARK / g) * t_exp + (R_noise / g)**2)

    # Parameters for a random object J02530+168

    l_ini = 1.1e-6 # Beginning od the J band [m]
    l_fin = 1.4e-6 # End of the J band [m]
    hc = 1.98644586e-25 # Speed light and Planck cte [J/m]
    F_obs_J = 1.277494916846618e-12 # Observational magnitude [W/m2/um]


    # Define the function inside the integral
    def function(x):
        return x

    def signal(module_diameter_m, efi_sys, hc, QE, l_ini, l_fin, F_obs_J):
        result, error = quad(function, l_ini, l_fin)
        n_adu = (np.pi * ((module_diameter_m / 2) ** 2) * efi_sys) / hc * F_obs_J * QE * result
        return n_adu

    N_ADU_TRAD = signal(module_diameter_m, 0.94, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_TRAD = (N_ADU_TRAD / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, 100 * 1e-3) + N_ADU_TRAD / g)
    
    N_ADU_MARCOT = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_MARCOT = (N_ADU_MARCOT / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT) + N_ADU_MARCOT / g)
    
    df['SNR fraction'] = None

    df.at[0, 'SNR fraction'] = SNR_MARCOT / SNR_TRAD

    df.to_csv('data/results_marcot.csv', sep='\t', index=False)
        
    return SNR_MARCOT / SNR_TRAD
    



