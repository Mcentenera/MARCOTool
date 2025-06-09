import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#############################################################################

def marcot_hr_estimator(
    module_diameter_m = 5,
    ota_diameter_m = 0.406,
    seeing_fwhm_arcsec = 0.85,
    use_tip_tilt = False,
    target_encircled_energy = 0.95,
    wavelength_min_nm = 500,
    wavelength_max_nm = 1000,
    magnification_factor = 1.2,
    pseudoslit = True,
    super_pl = False
):

    #########################################################################
    # 1. TELESCOPE MODULE
    #########################################################################

    com_ota_diam_mm, com_ota_f_number, com_ota_focal_length_mm, com_ota_tube_diam_mm, com_ota_tube_length_mm, com_ota_weight, com_ota_cost = np.loadtxt("data/Commercial_OTA.txt", usecols=(0, 1, 2, 3, 4, 5, 6),unpack=True)
    archive = open("data/Com_ota_choosen.txt", "w")
    
    for i in com_ota_diam_mm:
        if i >= ota_diameter_m * 1000:
            first_commercial = i
        break
        
    for i in range(0, len(com_ota_diam_mm)):
        if i == 0:
            archive.write(f'# Diameter(mm) f_number Focal_length(mm) Tube_Diameter(mm) Tube_length(mm) Weight(kg) Cost(€/u)\n')
        if com_ota_diam_mm[i] == first_commercial:
            archive.write(f'{com_ota_diam_mm[i]} {com_ota_f_number[i]} {com_ota_focal_length_mm[i]} {com_ota_tube_diam_mm[i]} {com_ota_tube_length_mm[i]} {com_ota_weight[i]} {com_ota_cost[i]}\n')
            
    archive.close()
    
    df = pd.read_csv("data/Com_ota_choosen.txt", sep="\t")

    df.to_csv("data/Com_ota_choosen.txt", sep="\t", index=False)
    
    com_ota_diam_mm, com_ota_f_number, com_ota_focal_length_mm, com_ota_tube_diam_mm, com_ota_tube_length_mm, com_ota_weight, com_ota_cost = np.loadtxt("data/Com_ota_choosen.txt", usecols=(0, 1, 2, 3, 4, 5, 6),unpack=True)

    module_area = math.pi * (module_diameter_m / 2)**2
    ota_area = math.pi * (com_ota_diam_mm * 1e-3 / 2)**2
    n_otas = math.ceil(module_area / ota_area)

    plate_scale_arcsec_per_mm = 206265 / com_ota_focal_length_mm

    effective_seeing_arcsec = seeing_fwhm_arcsec * (0.7 if use_tip_tilt else 1.0)
    psf_fwhm_mm = effective_seeing_arcsec / plate_scale_arcsec_per_mm
    fiber_core_mm = 1.5 * psf_fwhm_mm  # 1.5xFWHM to capture 90–95% energy
    fiber_core_microns = fiber_core_mm * 1000
    
    com_ota_weight, com_ota_cost = np.loadtxt("data/Com_ota_choosen.txt", usecols=(5,6),unpack=True)
    
    cost_tel = n_otas * com_ota_cost
    
    cost_trad = 2.37 * ( module_diameter_m ** 1.96 ) # Equation to estimated cost of a telescope with the same diamter with traditional design
    
    frac_cost = cost_trad / ( cost_tel / 1e6 )
    
    weight_tel = n_otas * (com_ota_weight + 10) # We sum 10 kg considering post-focus intrumentation

    #########################################################################
    # 2. PHOTONIC LANTERN
    #########################################################################

    fiber_core_m = fiber_core_microns * 1e-6

    commercial_core, e_commercial_core, commercial_NA, e_commercial_NA, cost_eur_m = np.loadtxt("data/Commercial_fibers.txt", usecols=(0, 1, 2, 3, 4),unpack=True)
    archive = open("data/Com_fiber_in.txt", "w")
    # df = pd.read_csv("Com_fiber_in.txt", sep="\t")

    for i in commercial_core:
      if i > fiber_core_microns:
        first_commercial = i
        break

    for i in range(0, len(commercial_core)):
      if i == 0:
        archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
      if commercial_core[i] == first_commercial:
        # df.loc[:, 'Core_size(um)'] = commercial_core[i]
        archive.write(f'{commercial_core[i]} {e_commercial_core[i]} {commercial_NA[i]} {e_commercial_NA[i]} {cost_eur_m[i]}\n')

    archive.close()

    df = pd.read_csv("data/Com_fiber_in.txt", sep="\t")

    df.to_csv("data/Com_fiber_in.txt", sep="\t", index=False)

    com_core_in, e_com_core_in, com_NA_in, e_com_NA_in, cost_eur_m_in = np.loadtxt("data/Com_fiber_in.txt", usecols=(0, 1, 2, 3, 4),unpack=True)

    wavelengths_m = [wavelength_min_nm * 1e-9, wavelength_max_nm * 1e-9]

    def modes(l, d, NA):
      return (1 / 2) * ((((2 * np.pi * d) / (2 * l)) * NA) ** 2)

    # modes_total_per_module = [n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, NA_in)]
    modes_total_per_module = [n_otas * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)]

    wl_short = wavelengths_m[0]
    V_required = np.sqrt(2 * modes_total_per_module[0])
    a_required_m = (V_required * wl_short) / (2 * math.pi * np.array([0.1, 0.22, 0.39, 0.5])) # We choose as NA an array of possibles NA numbers
    core_required_microns = 2 * a_required_m * 1e6


    archive = open("data/Com_fiber_out.txt", "w")

    first_commercial = []
    for i in core_required_microns:
      for j in commercial_core:
        if j > i:
          first_commercial.append(j)
          break

    for i in first_commercial:
      for j in range(0, len(commercial_core)):
        if i == 0:
          archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
        if commercial_core[j] == i:
          archive.write(f'{commercial_core[j]} {e_commercial_core[j]} {commercial_NA[j]} {e_commercial_NA[j]} {cost_eur_m[j]}\n')

    archive.close()

    com_core_out, e_com_core_out, com_NA_out, e_com_NA_out, cost_eur_m_out = np.loadtxt("data/Com_fiber_out.txt", usecols=(0, 1, 2, 3, 4),unpack=True)

    loss_comm = 10 * np.log10( n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in) / modes(wavelength_min_nm * 1e-9, com_core_out * 1e-6, com_NA_out))
    efficiency_comm = 1 - 10 ** (loss_comm / 10)

    # Force output of 100 microns (como CARMENES)
    # forced_output_core_microns = 100
    # modes_per_forced_output_fiber = modes(wl_short, forced_output_core_microns * 1e-6, NA_out)
    modes_per_output_fiber = modes(wl_short, com_core_out * 1e-6, com_NA_out)

    fibers_per_module_output = modes_total_per_module[0] / modes_per_output_fiber
    # modes_per_each_100_micron_fiber = modes_total_per_module[0] / fibers_per_module_output

    total_modules = (math.pi * (15 / 2)**2) / module_area

    if pseudoslit:
        total_fibers_pseudoslit = total_modules * fibers_per_module_output
    elif super_pl:
        total_modes_all_modules = total_modules * modes_total_per_module[0]
        num_super_output_fibers = math.ceil(total_modes_all_modules / modes_per_output_fiber)
        total_fibers_pseudoslit = num_super_output_fibers
    else:
        total_fibers_pseudoslit = total_modules * fibers_per_module_output

    cost_eur_m_in = np.loadtxt("data/Com_fiber_in.txt", usecols=(4),unpack=True)
    cost_eur_m_out = np.loadtxt("data/Com_fiber_out.txt", usecols=(4),unpack=True)
        
    cost_pl = n_otas * 2 * cost_eur_m_in + 3 * cost_eur_m_out
    
    #########################################################################
    # 3. SPECTROGRAPH / INSTRUMENT
    #########################################################################

    def resolution(n_order, lines, beam_diameter_mm, forced_output_core_microns, magnification_factor):
      N = lines * beam_diameter_mm
      w = lines * 0.5 * forced_output_core_microns * 1e-3 * magnification_factor
      return ((n_order * N) / w) * 0.66666666666

    fiber_core_mm_entry = fiber_core_microns * 1e-3
    beam_diameter_mm = 2 * 455 * np.tan(np.asin(com_NA_out)) * magnification_factor

    base_volume_m3 = 14  # ESPRESSO baseline
    base_beam_mm = 200
    volume_scale = (beam_diameter_mm / base_beam_mm)**3

    spectrograph_volume_m3 = base_volume_m3 * volume_scale
    spectrograph_weight_kg = 4000 * volume_scale
    cost_estimate_meur = 6 * volume_scale

    #########################################################################
    # PLOTS
    #########################################################################

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    ax.plot(com_NA_out, efficiency_comm, 'o', label = '200 um core')

    plt.title('PL\'s efficiency vs Numerical Aperture')
    plt.xlabel('Numerical Aperture (NA)')
    plt.ylabel('Efficiency (%)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="lower right")
    
    plt.savefig("Images/Efficiency_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Efficiency_vs_Aperture.png' was saved successfully\033[0m")

    plt.show(block=False)

    #########################################################################
    # RESULTS
    #########################################################################

    results = {
        # Telescope
        "Module diameter (m)": module_diameter_m,
        "OTA diameter (m)": com_ota_diam_mm * 1e-3,
        "Number of OTAs": n_otas,
        "F-number OTA": com_ota_f_number,
        "Focal length (mm)": com_ota_focal_length_mm,
        "Seeing FWHM (arcsec)": seeing_fwhm_arcsec,
        "Effective seeing (arcsec)": effective_seeing_arcsec if use_tip_tilt else seeing_fwhm_arcsec,
        "Input fiber core (microns)": np.round(fiber_core_microns,3),
        "Selected commercial input core (microns)": np.round(com_core_in, 3),
        "Selected commercial input NA": np.round(com_NA_in, 3),
        "Total cost for each module (MEUR)": np.round(cost_tel * 1e-6, 3),
        "Total cost (MEUR)": np.round((cost_tel * total_modules) * 1e-6, 3),
        "Reduction cost factor": np.round(frac_cost, 3),
        "Weight supported by the mount (kg)": np.round(weight_tel, 3),

        # Photonic Lantern
        "": None,
        "Modes per fiber [500nm,1000nm]": np.array([np.round(modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0),np.round(modes(wavelength_max_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        "Total modes per module": np.array([np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0), np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        "Required output fiber core (microns)": np.round(core_required_microns,3),
        "Selected commercial output core (microns)": np.round(com_core_out, 3),
        "Selected commercil output NA": np.round(com_NA_out, 3),
        "Expected efficiency (%)": np.round(efficiency_comm * 100,3),
        # "Fibers per module @100µm": np.round(fibers_per_module_output,3),
        # "Modes per 100µm fiber": math.ceil(modes_per_each_100_micron_fiber),
        "Total modules": total_modules,
        "Total fibers at pseudorendija": total_fibers_pseudoslit,
        "Total cost PLs (MEUR)": np.round((cost_pl * total_modules) * 1e-6, 3),

        # Spectrograph
        "": None,
        "Beam diameter at spectrograph (mm)": np.round(beam_diameter_mm,3),
        "Spectrograph volume (m³)": np.round(spectrograph_volume_m3,3),
        "Spectrograph weight (kg)": np.round(spectrograph_weight_kg,3),
        "Estimated cost (MEUR)": np.round(cost_estimate_meur,3),
        "Resolution with commercial fibers": np.round(resolution(55, 31.6, beam_diameter_mm, com_core_out, magnification_factor)),
        # "S/N improvement factor": np.round(4.409 , 3),
        # If we are going to use a super PL, the S/N improvment factor is 1.6105 for 5m, 2.9463 for 10m and 4.4170 for 15m
        # If we use pseudoslit, this must be calculated
        # Fuse S/N calculator with this pipeline
        "Reduction factor in R (%)": np.round(resolution(55, 31.6, beam_diameter_mm, com_core_out, magnification_factor) / 94600 * 100,3),

        # Auxiliar parameters
        "Use tip/tilt?": use_tip_tilt,
        "Magnification factor": magnification_factor,
        "Pseudoslit": pseudoslit,
        "Super-PL": super_pl
    }
    return results, module_diameter_m, com_core_out * 1e-3
        

