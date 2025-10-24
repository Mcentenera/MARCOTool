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

def multi_criteria(results_csv_path: str = None, criteria: dict = None):
    # Si no se pasa ruta, usamos el CSV por defecto
    if results_csv_path is None:
        results_csv_path = RES("data", "results_marcot.csv")
    df = pd.read_csv(results_csv_path, sep='\t')

    if criteria is None:
        criteria = {
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

    more = ["Expected efficiency (%)", "Resolution with commercial fibers", "SNR fraction",
            "SNR PL vs pseu fraction", "Recalculated telescope aperture (m)"]
    less = ["Cost telescope + PL (MEUR)", "Total modules" ,"Number of OTAs", "Spectrograph volume (m³)"]

    # --- Conversión de columnas con arrays ---
    list_columns = [col for col in df.columns if df[col].dtype == object and df[col].astype(str).str.startswith('[').any()]
    for col in list_columns:
        def parse_array(x):
            if not isinstance(x, str) or not re.search(r'\d', x):
                return x
            x = x.strip("[] \n\t")
            return np.fromstring(x, sep=' ')
        df[col] = df[col].apply(parse_array)

    # --- Normalización ---
    for criterio in criteria:
        if criterio not in df.columns:
            print(f"[WARNING] Missing column: {criterio}")
            continue

        col_data = df[criterio]
        has_array = col_data.apply(lambda x: isinstance(x, (list, np.ndarray))).any()

        if has_array:
            all_values = np.concatenate(col_data.apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x])))
            vmin, vmax = np.min(all_values), np.max(all_values)
            if vmax == vmin:
                df[criterio + "_norm"] = 0
            elif criterio in more:
                df[criterio + "_norm"] = col_data.apply(lambda x: (np.array(x) - vmin) / (vmax - vmin))
            else:
                df[criterio + "_norm"] = col_data.apply(lambda x: (vmax - np.array(x)) / (vmax - vmin))
        else:
            vmin, vmax = col_data.min(), col_data.max()
            if vmax == vmin:
                df[criterio + "_norm"] = 0
            elif criterio in more:
                df[criterio + "_norm"] = (col_data - vmin) / (vmax - vmin)
            else:
                df[criterio + "_norm"] = (vmax - col_data) / (vmax - vmin)

      # Calculate score total
    df["score_total"] = df[[c + "_norm" for c in criteria]].apply(
    lambda row: sum(row[c + "_norm"] * criteria[c] for c in criteria),
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
    
    for col in criteria.keys():
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


    
def snr_cal_2(archive):
    # Detector paremeters (CARMENES-VIS)
    df = pd.read_csv('data/results_marcot.csv', sep = '\t')
    try:
        def parse_array(x):
            if not isinstance(x, str) or not re.search(r'\d', x):
                return x
            x = x.strip("[] \n\t")
            return np.fromstring(x, sep=' ')
        df['Selected commercial output core (microns)'] = df['Selected commercial output core (microns)'].apply(parse_array)
        df['Selected commercial input core (microns)'] = df['Selected commercial input core (microns)'].apply(parse_array)
        df['Selected commercial output core 2-stage (microns)'] = df['Selected commercial ouput core 2-stage (microns)'].apply(parse_array)
        df['Selected commercial output core (microns)'] = df['Selected commercial ouput core (microns)'].apply(parse_array)
        df['Magnification factor'] = df['Magnification factor'].apply(parse_array)
        df['Focal length (mm)'] = df['Focal length (mm)'].apply(parse_array)
        df['Pixel size (microns)'] = df['Pixel size (microns)'].apply(parse_array)
        df['Number of OTA for high efficiency'] = df['Number of OTA for high efficiency'].apply(parse_array)
        df['Total modules'] = df['Total modules'].apply(parse_array)
        
        # ESTO HAY QUE CAMBIARLO PARA QUE NO HAYA QUE METER LOS PARÁMETROS UNO A UNO, SINO QUE LE LOOP PASE POR TODAS
    except Exception as e:
        print(f'\033[4;93;1mWARNING: Error in column: {str(e)}\033[0m')
        return None
    
    module_diameter_m = df['Module diameter (m)'].iloc[0]
    com_core_in = df['Selected commercial input core (microns)'].iloc[0] * 1e-6
    efi_sys_MARCOT = 0.9
    # CAMBIAR A UN VALOR MEJOR CUANDO EL stimator.py CALCULE AL EFICIENCIA DLE SISTEMA DE MARCOT
    fiber_core_m_MARCOT = (df['Selected commercial output core (microns)'].iloc[0]) * 1e-6
    # CUIDADO QUE EL VALOR QUE COJO DE df ESTÁ EN um Y NO EN mm
    magnification_factor = df['Magnification factor'].iloc[0]
    f_cam = df['Focal length camera (mm)'].iloc[0]
    n_fiber_total = df['Number of OTA for high efficiency'].iloc[0]
    total_modules = df['Total modules'].iloc[0]
    com_core_out_2 = (df['Selected commercial output core 2-stage (microns)'].iloc[0]) * 1e-6
    com_core_out = (df['Selected commercial output core (microns)'].iloc[0]) * 1e-6
        
    t_exp = 100 # Exposure time [s]
    plate_scale = 0.169 # Plate scale [mm/"]
    R_noise = 5 # Read noise [e-]
    g = 1 # Ganancia [e-/ADU]
    pix_size = df['Pixel size (microns)'].iloc[0] # Pixel size [mm]
    DARK = 3 # DARK current [e-/s]
    QE = 0.92 # Quatum efficiency
    R = 94600
    
    def n_pix(magnification_factor, pix_size, fiber_core_mm, f_cam, R, theta_i, grating):
        Theta_B = np.arctan(grating)
        theta_d = 2 * Theta_B - theta_i
        n_spa = 1.5 * ((magnification_factor * fiber_core_mm / 2) / (pix_size * 1e-6)) # Pixel in spatial direction. Divide by 2 because we have an image slicer
        n_dis = (f_cam * 1e-3 / (pix_size * 1e-6 * R)) * (np.sin(theta_i) / np.cos(theta_d) + np.tan(theta_d))# Pixel in dispertion direction
        return n_spa * n_dis


    def N_det(DARK, t_exp, g, R_noise, magnification_factor, n_fiber_total, pix_size, fiber_core_mm, f_cam, R, theta_i, grating):
        return n_fiber_total * n_pix(magnification_factor, pix_size, fiber_core_mm, f_cam, R, theta_i, grating) * ((DARK / g) * t_exp + (R_noise / g) ** 2)

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

    N_ADU_PSEU = signal(module_diameter_m, 0.94, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_PSEU = (N_ADU_PSEU / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, n_fiber_total, pix_size, com_core_in, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_PSEU / g)
        
    N_ADU_PL = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_PL = (N_ADU_PL / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, 1, pix_size, fiber_core_m_MARCOT, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_PL / g)
    
    N_ADU_TRAD = signal(module_diameter_m, 0.94, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_TRAD = (N_ADU_TRAD / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, 1, pix_size, 100e-3, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_TRAD / g)
    
    N_ADU_MARCOT = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_MARCOT = (N_ADU_MARCOT / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, total_modules, pix_size, fiber_core_mm_MARCOT, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_MARCOT / g)
    
    N_ADU_PSEU_2 = signal(module_diameter_m, 0.94, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_PSEU_2 = (N_ADU_PSEU / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, n_modules_total, pix_size, com_core_out, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_PSEU / g)
        
    N_ADU_PL_2 = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_PL_2 = (N_ADU_PL / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, magnification_factor, 1, pix_size, com_core_out_2, f_cam, R, 75.2 * np.pi / 180, 4) + N_ADU_PL / g)
    
    df['SNR fraction'] = None

    df.at[0, 'SNR fraction'] = SNR_MARCOT / SNR_TRAD

    #df.to_csv('data/results_marcot.csv', sep='\t', index=False)
    
    df['SNR PL vs pseu fraction'] = None

    df.at[0, 'SNR PL vs pseu fraction'] = SNR_PL / SNR_PSEU

    # df.to_csv('data/results_marcot.csv', sep='\t', index=False)
    
    df['SNR PL vs pseu fraction 2-stage'] = None

    df.at[0, 'SNR PL vs pseu fraction 2-stage'] = SNR_PL_2 / SNR_PSEU_2

    df.to_csv('data/results_marcot.csv', sep='\t', index=False)
        
    return SNR_PL / SNR_PSEU, SNR_MARCOT / SNR_TRAD, SNR_PL_2 / SNR_PSEU_2
