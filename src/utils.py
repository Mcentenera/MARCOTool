import pandas as pd
import numpy as np
from scipy.integrate import quad

def print_results(results, module_diameter_m, com_core_out_mm):
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
    
    print_section("SCIENTIFIC CRITERIA")
    fraction = snr_cal(module_diameter_m, 0.9, com_core_out_mm )
    # HAY QUE DEFINIR UN VALOR MÁS REALISTA DE LA EFICIENCIA DEL SISTEMA DE MARCOT Y NO PONER UN 0.9 POR DEFECTO.
    
    # HAY QUE CAMBIAR EL module_diameter_m, POR PARÁMETROS n_otas Y diameter_ota_m PARA QUE CALCULA LA APERTURA EFECTIVA
    
    print(f"MARCOT telescope has a {np.round(fraction, 3)} * SNR in comparison to a traditional design")

def multi_criteria(archive):

    # Import results from estimator.py
    df = pd.DataFrame('data/results_marcot.txt')

    # Define criteria
    criterios = {
    "Expected efficiency": 0.9,
    "Estimated cost": 0.9,
    "Resolution": 0.8,
    "Total fibers": 0.2,
    "Spectrograph volume": 0.5
    }
    
    # Divide in groups more is better or less is better
    more = ["Expected efficiency", "Resolution"]
    less = ["Estimated cost", "Total fibers", "Spectrograph volume"]
    
    # Calculated normalized value
    for criterio in criterios:
        if criterio in more:
            df[criterio + "_norm"] = (df[criterio] - df[criterio].min()) / (df[criterio].max() - df[criterio].min())
        else:
            df[criterio + "_norm"] = (df[criterio].max() - df[criterio]) / (df[criterio].max() - df[criterio].min())
            
    # Calculate over the ponderated score
    df["score_total"] = sum(df[c + "_norm"] * w for c, w in criterios.items())
    
    df_sorted = df.sort_values(by="total_score", ascending=False)
    print(df_sorted[["Module diameter (m)", "Use tip/tilt?", "total_score"] + list(criterios.keys())])
    
def snr_cal(module_diameter_m, efi_sys_MARCOT, fiber_core_mm_MARCOT):
    # Detector paremeters (CARMENES-VIS)
    
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
        n_adu = (np.pi * (module_diameter_m / 2) * efi_sys) / hc * F_obs_J * QE * result
        return n_adu

    N_ADU_TRAD = signal(module_diameter_m, 0.94, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_TRAD = (N_ADU_TRAD / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, 100 * 1e-3) + N_ADU_TRAD / g)
    
    N_ADU_MARCOT = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    SNR_MARCOT = (N_ADU_MARCOT / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT) + N_ADU_MARCOT / g)
    
    return SNR_MARCOT / SNR_TRAD







