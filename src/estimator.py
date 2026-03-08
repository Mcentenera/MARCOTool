import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#############################################################################

def marcot_hr_estimator(
    module_diameter_m,
    f_number_out,
    f_number_out_locked,
    d_core_out,
    d_core_out_locked,
    telescope_aperture_m,
    seeing_fwhm_arcsec,
    sky_aperture,
    sky_aperture_locked,
    use_tip_tilt,
    target_encircled_energy,
    wavelength_min_nm,
    wavelength_max_nm,
    pseudoslit,
    super_pl,
    grooves_mm,
    resolution,
    magnification_factor,
    beam_diameter_mm,
    pixel_size,
    rel_element,
    f_cam_mm,
    f_cam_mm_locked,
    nir_arm
):

    #########################################################################
    # 1. TELESCOPE MODULE
    #########################################################################

    com_ota_diam_mm, com_ota_f_number, com_ota_focal_length_mm, com_ota_tube_diam_mm, com_ota_effi, com_ota_tube_length_mm, com_ota_weight, com_ota_cost = np.loadtxt("data/Commercial_OTA.txt", usecols=(0, 1, 2, 3, 4, 5, 6, 7),unpack=True)
    
    e_module_diameter_m = 0.1
    module_area = math.pi * (module_diameter_m / 2) ** 2
    e_module_area = (math.pi / 2) * module_diameter_m * e_module_diameter_m
    
    e_com_ota_diam = 0.1 * 1e-3
    ota_area = math.pi * (com_ota_diam_mm * 1e-3 / 2) ** 2
    e_ota_area = (math.pi / 2) * com_ota_diam_mm * 1e-3 * e_com_ota_diam
    n_otas = np.round(module_area / ota_area, 0)
    e_n_otas = n_otas * np.sqrt((e_module_area / module_area) ** 2 + (e_ota_area / ota_area) ** 2)

    plate_scale_arcsec_per_mm = 206265 / com_ota_focal_length_mm
    new_plate_scale = 206265 / (com_ota_diam_mm * 5) # Recalculate the plate scale, using a focal adapter to pass to f/5

    effective_seeing_arcsec = seeing_fwhm_arcsec * (0.8 if use_tip_tilt else 1.0)
    psf_fwhm_mm = effective_seeing_arcsec / new_plate_scale
    
    
    # Calculate the diameter which encircled the target_encricled_energy
    if sky_aperture_locked:
        fiber_core_mm = sky_aperture / new_plate_scale
        fiber_core_microns = fiber_core_mm * 1000
    else:
        fiber_core_mm = (effective_seeing_arcsec * np.sqrt(- np.log(1 - target_encircled_energy) / np.log(2))) / new_plate_scale
        fiber_core_microns = fiber_core_mm * 1000

    #########################################################################
    # 2. PHOTONIC LANTERN
    #########################################################################

    fiber_core_m = fiber_core_microns * 1e-6

    commercial_core, e_commercial_core, commercial_NA, e_commercial_NA, cost_eur_m = np.loadtxt("data/Commercial_fibers.txt", usecols=(0, 1, 2, 3, 4),unpack=True)
    archive = open("data/Com_fiber_in.txt", "w")
    
    first_commercial = []

    for j in fiber_core_microns:
        for i in commercial_core:
            if i > j:
                first_commercial.append(i)
                break
    first_commercial = np.array(first_commercial)
        
    for j in first_commercial:
        for i in range(0, len(commercial_core)):
            if i == 0:
                archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
            
            if commercial_core[i] == j:
                archive.write(f'{commercial_core[i]} {e_commercial_core[i]} {commercial_NA[i]} {e_commercial_NA[i]} {cost_eur_m[i]}\n')
                break

    archive.close()

    df = pd.read_csv("data/Com_fiber_in.txt", sep="\t")

    df.to_csv("data/Com_fiber_in.txt", sep="\t", index=False)

    com_core_in, e_com_core_in, com_NA_in, e_com_NA_in, cost_eur_m_in = np.loadtxt("data/Com_fiber_in.txt", usecols=(0, 1, 2, 3, 4),unpack=True)
    
    wavelengths_m = [wavelength_min_nm * 1e-9, wavelength_max_nm * 1e-9]

    def modes(l, d, NA):
      return (1 / 2) * ((((np.pi * d) / l) * NA) ** 2)
      
    e_l = 0.1 * 1e-9 # we set the uncertain on wavelength to 1 nm
      
    def e_modes(l, e_l, d, e_d, NA, e_NA):
        return modes(l, d, NA) * np.sqrt((2 * e_d / d) ** 2 + (2 * e_NA / NA) ** 2 + (2 * e_l / l) ** 2)
      
    # Define a funtion to calculate the NA needed for the capillary, acording to fiber and PL NA
    
    # THIS DEPENDS ON THE INPUT FIBERS MATERIALS, IS CLADDING FIBERS ARE SILICA, NA_PL=NA_cap
    
    def NA_cap_estimator(NA_PL, com_NA_in):
        return np.sqrt(NA_PL ** 2 - com_NA_in **2)
        
    def NA_PL_estimator(NA_cap, com_NA_in):
        return np.sqrt(NA_cap ** 2 - com_NA_in ** 2)

    archive = open("data/Com_fiber_out.txt", "w")
    archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
    
    # We are going to recalculate the number of fiber for higher efficiency
    
    n_fiber_total = []
    def n_fiber(n):
        return 1 + 3 * n * (n + 1)
        
    for j in range(0, len(n_otas)):
        for i in range(1, 1000):
            if n_otas[j] > 27 and n_fiber(i) > n_otas[j]:
                n_fiber_total.append(n_otas[j])
                break
            elif n_otas[j] < 27 and n_fiber(i) > n_otas[j]:
                n_fiber_total.append(n_fiber(i)) # If we want to be under the optimal fibre we have to put n_fiber(i-1)
                break

    def e_n_fiber_total(module_diameter_m, com_ota_diam_mm, n):
        return n * np.sqrt((2 * 0.1 / module_diameter_m) ** 2 + (2 * 0.1 * 1e-3 / (com_ota_diam_mm * 1e-3)) ** 2)
    
    modes_total_per_module = n_fiber_total * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)
    
    # Set an array with possible output diameters between 0 and 1000 um
    
    core_diam_um = np.array(np.linspace(0,1000,1001))
    
    for i in modes_total_per_module:
        if f_number_out_locked:
            for j in range(0,len(core_diam_um)):
                NA_PL = np.array(np.linspace(1 / (2 * f_number_out), 1 / (2 * f_number_out), len(core_diam_um)))
                modes_out_PL = modes(wavelength_min_nm * 1e-9, core_diam_um[j] * 1e-6, NA_PL[j])
                if modes_out_PL >= i:
                    if d_core_out_locked:
                        archive.write(f'{d_core_out} {0} {np.round(NA_PL[j], 3)} {0} {0}\n')
                    else:
                        archive.write(f'{core_diam_um[j]} {0} {np.round(NA_PL[j], 3)} {0} {0}\n')
                    break
        else:
            found = False
            for j in range(0,len(core_diam_um)):
                # If we do not lock NA_PL, we choose the best option for the commercial list
                # modes_out_PL = modes(wavelength_min_nm * 1e-9, commercial_core[j] * 1e-6, commercial_NA[j])
                for jj in com_NA_in:
                    if modes(wavelength_min_nm * 1e-9, core_diam_um[j] * 1e-6, (NA_PL_estimator(0.22, jj))) >= i:
                        if d_core_out_locked:
                            archive.write(f'{d_core_out} {0} {np.round(NA_PL_estimator(0.22, jj), 3)} {0} {0}\n')
                        else:
                            archive.write(f'{core_diam_um[j]} {0} {np.round(NA_PL_estimator(0.22, jj), 3)} {0} {0}\n')
                        found = True
                        break
                if found:
                    break


    archive.close()

    com_core_out, e_com_core_out, com_NA_out, e_com_NA_out, cost_eur_m_out = np.loadtxt("data/Com_fiber_out.txt", usecols=(0, 1, 2, 3, 4),unpack=True)
    
    efficiency_comm = []
    e_efficiency_comm = []
    for i in range(0,len(com_NA_out)):
        efi = ((modes(wavelength_min_nm * 1e-9, com_core_out[i] * 1e-6, com_NA_out[i])) / (n_fiber_total[i] * modes(wavelength_min_nm * 1e-9, com_core_in[i] * 1e-6, com_NA_in[i]))) * 100
        
        efficiency_comm.append(efi * com_ota_effi[i] ** 2)
        e_efficiency_comm.append(efi * np.sqrt((e_modes(wavelength_min_nm * 1e-9, e_l, com_core_out[i]* 1e-6, e_commercial_core[i]* 1e-6, com_NA_out[i], e_com_NA_out[i]) / modes(wavelength_min_nm * 1e-9, com_core_out[i]* 1e-6, com_NA_out[i])) ** 2 + (e_modes(wavelength_min_nm * 1e-9, e_l, com_core_in[i]* 1e-6, e_com_core_in[i]* 1e-6, com_NA_in[i], e_com_NA_in[i])/ modes(wavelength_min_nm * 1e-9, com_core_in[i]* 1e-6, com_NA_in[i]) ) ** 2 + ( e_n_fiber_total(module_diameter_m, com_ota_diam_mm[i], n_fiber_total[i]) / n_fiber_total[i] ) ** 2))

        if efficiency_comm[i] > 100:
            efficiency_comm[i] == 100
    
    
    fibers_per_module_output = modes_total_per_module / modes(wavelength_min_nm * 1e-9, com_core_out, com_NA_out)
    
    def evanescent(com_core_in, com_core_out, com_NA_out, n_otas, com_ota_diam_mm):
    
        taper_ratio_total = []
    
        for j in range(0, len(n_otas)):
            for i in range(1, 1000):
                if n_fiber(i) > n_otas[j]:
                
                    taper_ratio = ((1 + 2 * i) * 125e-6) / (com_core_out[j] * 1e-6)
                    taper_ratio_total.append(taper_ratio)
                    break
        recal_module_diam = np.sqrt(n_fiber_total) * com_ota_diam_mm * 1e-3
        
        def alpha(r, r_f):
            return np.arctan((r-r_f)/0.04)

        def radius(r,lon,alpha):
            return r-np.tan(alpha)*lon

        def modes(l, d, NA):
            return (1/4)*((((2*np.pi*d)/(2*l))*NA)**2)
    
        return taper_ratio_total, recal_module_diam, n_fiber_total
        
    taper_ratio_total, recal_module_diam, n_fiber_total = evanescent(com_core_in, com_core_out, com_NA_out, n_otas, com_ota_diam_mm)
    
    module_area = math.pi * (recal_module_diam / 2)**2
    
    total_modules = np.round((math.pi * (telescope_aperture_m / 2)**2) / module_area)
    
    # We are going to recalculate the number of modules for higher efficiency if we use a PL os second stage
    
    n_modules_total = []

    modes_total_per_module = n_fiber_total * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)

    if pseudoslit:
        n_modules_total = total_modules
    elif super_pl:
        for j in range(0, len(n_otas)):
            for i in range(1, 1000):
                if n_fiber(i) > total_modules[j]:
                    n_modules_total.append(n_fiber(i)) # If we want to be under the best fibre number, we have to change to n_fiber(i-1)
                    break
    else:
        total_fibers_pseudoslit = total_modules * fibers_per_module_output
        
        
    # We are going to calculate the output fiber required for a PL of second stage
    
    # if f_number_out_locked:
    #     NA_PL = np.array(np.linspace(1 / (2 * f_number_out), 1 / (2 * f_number_out), len(core_diam_um)))
    #     modes_module = n_modules_total * modes(wavelength_min_nm * 1e-9, core_diam_um * 1e-6, NA_PL)
    # else:
    modes_module = n_modules_total * modes(wavelength_min_nm * 1e-9, com_core_out * 1e-6, com_NA_out)
            
    com_core_out_2 = []
    com_NA_out_2 = []
    
    for i in modes_module:
        if f_number_out_locked:
            for j in range(0,len(core_diam_um)):
                NA_PL = np.array(np.linspace(1 / (2 * f_number_out), 1 / (2 * f_number_out), len(core_diam_um)))
                modes_out_PL = modes(wavelength_min_nm * 1e-9, core_diam_um[j] * 1e-6, NA_PL[j])
                if modes_out_PL >= i:
                    com_core_out_2.append(core_diam_um[j])
                    com_NA_out_2.append(NA_PL[j])
                    break
        else:
            for j in range(0,len(commercial_NA)):
                modes_out_PL = modes(wavelength_min_nm * 1e-9, commercial_core[j] * 1e-6, commercial_NA[j])
                if modes_out_PL >= i:
                    com_core_out_2.append(commercial_core[j])
                    com_NA_out_2.append(commercial_NA[j])
                    break
                    
    com_core_out_2 = np.array(com_core_out_2)
    com_NA_out_2 = np.array(com_NA_out_2)
    
    recal_telescope_aperture = np.sqrt(n_modules_total) * recal_module_diam
    
    cost_eur_m_in = np.loadtxt("data/Com_fiber_in.txt", usecols=(4),unpack=True)
    cost_eur_m_out = np.loadtxt("data/Com_fiber_out.txt", usecols=(4),unpack=True)
        
    cost_pl = np.array(n_fiber_total) * 2 * np.array(cost_eur_m_in) + 3 * np.array(cost_eur_m_out)
    
    # Equation to estimated cost of a telescope with the same diameter with traditional design
    def cost_estimator(com_ota_diam_mm):
        return (2.37 * com_ota_diam_mm ** 1.96) / 3
    
    cost_tel = np.array(n_fiber_total) * np.array(com_ota_cost)
    
    e_com_ota_weight = 0.1
    weight_tel = np.array(n_fiber_total) * (com_ota_weight + 10) # We sum 10 kg considering post-focus intrumentation
    e_weight_tel = []
    for i in range(0,len(com_ota_diam_mm)):
        e_weight_tel.append(weight_tel[i] * ((e_n_fiber_total(module_diameter_m, com_ota_diam_mm[i], n_fiber_total[i]) / n_fiber_total[i]) ** 2 + ( e_com_ota_weight / (com_ota_weight[i] + 10)) ** 2))
    e_weight_tel = np.array(e_weight_tel)

    #########################################################################
    # 3. SPECTROGRAPH / INSTRUMENT
    #########################################################################

    # This formula is to calculate the new resolution
    def resolution(n_order, lines, beam_diameter_mm, forced_output_core_microns, magnification_factor):
      N = lines * beam_diameter_mm
      w = lines * 0.5 * forced_output_core_microns * 1e-3 * magnification_factor
      # the value 0.5 is because in CARMENES we use slicer to divide in two parts the spot
      return ((n_order * N) / w) * 0.66666666666
      
    def e_resolution(n_order, lines, beam_diameter_mm, forced_output_core_microns, e_forced_output_core_microns, magnification_factor):
        return resolution(n_order, lines, beam_diameter_mm, forced_output_core_microns, magnification_factor) * np.sqrt((0.1 / n_order) ** 2 + (0.1 / beam_diameter_mm) ** 2 + (e_forced_output_core_microns / forced_output_core_microns) ** 2 + (0.01 / magnification_factor) ** 2 )

    beam_diameter_mm = 2 * 455 * np.tan(np.asin(com_NA_out)) * magnification_factor
    # f_col_mm = np.array(beam_diameter_mm / (2 * np.tan(np.arcsin(com_NA_out))))
    f_col_mm = 455
    # Calculate detector dimensions units in [mm]
    theta_B = np.arctan(4) # Blaze angle using R4 grating
    theta_i = 75.2 * np.pi / 180 # Incident angle in [rad] taken from CARMENES
    theta_d = 2 * theta_B - theta_i # Dispersion angle [rad]
    
    if not f_cam_mm_locked:
        f_cam_mm = np.array(f_col_mm / magnification_factor)
    else:
        pass
        
    
    wavelength_range_m = np.array(np.linspace(wavelength_min_nm, wavelength_max_nm, wavelength_max_nm - wavelength_min_nm + 1))
    
    def detector_length(grooves_mm, f_cam_mm , theta_d, wavelength_max_nm, wavelength_min_nm):
        return (grooves_mm * 1e3 * (wavelength_max_nm * 1e-9 - wavelength_min_nm * 1e-9) * f_cam_mm * 1e-3) / np.cos(theta_d)
        
    def m(wavelength_range_m, grooves_mm, theta_i, theta_d):
        return (np.sin(theta_i) + np.sin(theta_d)) / (wavelength_range_m * grooves_mm)
    
   # def dispersion(wavelength_range_m, grooves_mm, theta_d, f_cam_mm):
   #     return np.cos(theta_d)  / (m(wavelength_range_m, grooves_mm, theta_i, theta_d) * f_cam_mm * grooves_mm)
        
    dispersion_nm_um = np.cos(theta_d) * (1e6 / grooves_mm) / (53 * np.array(f_cam_mm) ) # 53 is the number od orders in VIS
            
    resolution_element = 1 / (dispersion_nm_um * pixel_size / (wavelength_max_nm / resolution(53, 31.6, beam_diameter_mm, com_core_out, magnification_factor)))
        
    # === Baseline for visible spectograph of high stability such as ESPRESSO ===
    BASE = {
        "beam_mm": 200.0,            # reference pupil collimated
        "vol_m3": 14.0,              # total volume
        "mass_kg": 4000.0,           # reference total mass
        "cost_meur": 6.0,            # cost of visible
    }

    def cubic_scale(beam_mm, base_beam_mm=BASE["beam_mm"]):
        return (beam_mm / base_beam_mm)**3

    def spectrograph_estimate(beam_mm,
                          nir_arm=False,       # sincase of NIR branch
                          ultra_stable=True,   # termal control ~0.01 ºC and vacum
                          extras={}):
        if nir_arm == True:
            n_arms = 2
        else:
            n_arms = 1
            
        scale = cubic_scale(beam_mm)

    # 1) Optical core (per branch)
        core_vol = BASE["vol_m3"] * scale
        core_mass = BASE["mass_kg"] * scale
        core_cost = BASE["cost_meur"] * scale
        e_core_cost = core_cost * np.sqrt((0.1 / BASE["cost_meur"]) ** 2 + ((beam_mm * 0.01) / (BASE["beam_mm"])) ** 2)
        e_core_vol = core_vol * np.sqrt((0.01 / BASE["vol_m3"]) ** 2 + ((beam_mm * 0.01) / (BASE["beam_mm"])) ** 2)

    # 2) Additional branch (if n_arms > 1)
        vol = core_vol * n_arms
        mass = core_mass * n_arms
        cost = core_cost * n_arms

    # 3) Add factor for including NIR branch (detector HxRG + cryogenics + coaters IR)
        if nir_arm:
            nir_factor_cost = 0.35   # +35% of the VIS branch cost
            nir_factor_vol  = 0.20   # +20% extra volume
            vol  += core_vol * nir_factor_vol
            cost += core_cost * nir_factor_cost

    # 4) Ultra-stability
        if ultra_stable:
            stab_cost_factor = 0.20  # +20% cost
            stab_vol_factor  = 0.10  # +10% volume
            vol  *= (1.0 + stab_vol_factor)
            cost *= (1.0 + stab_cost_factor)

    # 5) Extras
        for k, v in (extras or {}).items():
            if k == "laser_comb":
                cost += 0.7          # M€ typical of purchase/integ.
            if k == "multi_UT":
                cost += 0.5 * v      # M€ for additional canal
            if k == "image_slicer":
                cost += 0.2

        return vol, e_core_vol, mass, cost, e_core_cost, scale
        
    volume_m3_spec, e_volume_m3_spec, mass_kg_spec, cost_meur_spec, e_core_cost_spec, scale_spec = spectrograph_estimate(beam_diameter_mm,
                          nir_arm=False,       # sincase of NIR branch
                          ultra_stable=True,   # termal control ~0.01 ºC and vacum
                          extras={})
                          
    if nir_arm == True:
        frac_cost = (cost_estimator(module_diameter_m) + 12) / ((cost_tel * 1e-6) + cost_meur_spec)
        e_frac_cost = np.sqrt((((2.37 * 1.96 * module_diameter_m ** 0.96) / (3 * (cost_tel * 1e-6) + cost_meur_spec)) * 0.1) ** 2 + (((3 * (2.37 * module_diameter_m ** 1.96 + 12)) / (3 * (cost_tel * 1e-6) + cost_meur_spec) ** 2) * 0.1) ** 2 + (((2.37 * module_diameter_m ** 1.96 + 12) / (3 * (cost_tel * 1e-6) + cost_meur_spec) ** 2) * e_core_cost_spec) ** 2)
    else:
        frac_cost = (cost_estimator(module_diameter_m) + 6) / ((cost_tel * 1e-6) + cost_meur_spec)
        e_frac_cost = np.sqrt((((2.37 * 1.96 * module_diameter_m ** 0.96) / (3 * (cost_tel * 1e-6) + cost_meur_spec)) * 0.1) ** 2 + (((3 * (2.37 * module_diameter_m ** 1.96 + 6)) / (3 * (cost_tel * 1e-6) + cost_meur_spec) ** 2) * 0.1) ** 2 + (((2.37 * module_diameter_m ** 1.96 + 6) / (3 * (cost_tel * 1e-6) + cost_meur_spec) ** 2) * e_core_cost_spec) ** 2)
    #########################################################################
    # PLOTS
    #########################################################################

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    
    for i in range(0, len(com_NA_out)):
        ax.plot(com_NA_out[i], efficiency_comm[i], 'o', label = f'{com_core_out[i]} um core')

    plt.title('PL\'s efficiency vs Numerical Aperture')
    plt.xlabel('Numerical Aperture (NA)')
    plt.ylabel('Efficiency')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="upper left")
    
    plt.savefig("Figures/Efficiency_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Efficiency_vs_Aperture.png' was saved successfully\033[0m")

    plt.close()

    #########################################################################
    # RESULTS
    #########################################################################

    results = {
        # Telescope
        "Module diameter (m)": module_diameter_m,
        "OTA diameter (m)": com_ota_diam_mm * 1e-3,
        "Uncer OTA diameter (m)": np.linspace(0.001, 0.001, len(com_ota_diam_mm)),
        "Number of OTAs": n_otas,
        "Uncer Number of OTAs": e_n_otas,
        "F-number OTA": com_ota_f_number,
        "Focal length (mm)": com_ota_focal_length_mm,
        "Seeing FWHM (arcsec)": seeing_fwhm_arcsec,
        "Effective seeing (arcsec)": effective_seeing_arcsec if use_tip_tilt else seeing_fwhm_arcsec,
        "Input fiber core (microns)": np.round(fiber_core_microns,3),
        "Selected commercial input core (microns)": np.round(com_core_in, 3),
        "Selected commercial input NA": np.round(com_NA_in, 3),
        "Sky aperture (arcsec)": np.round(sky_aperture, 3),
        "Total cost for each module (MEUR)": np.round((cost_tel * 1e-6), 3),
        "Total cost (MEUR)": np.round(((cost_tel * 1e-6) * total_modules + cost_pl * 1e-6), 3),
        "Reduction cost factor": np.round(frac_cost, 3),
        "Uncer Reduction cost factor": np.round(e_frac_cost, 3),
        "Weight supported by the mount (kg)": np.round(weight_tel, 3),
        "Uncer Weight supported by the mount (kg)": np.round(e_weight_tel, 3),

        # Photonic Lantern
        "": None,
        #"Modes per fiber [500nm,1000nm]": np.array([np.round(modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0),np.round(modes(wavelength_max_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        #"Total modes per module": np.array([np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0), np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        "Required output fiber core (microns)": np.round(com_core_out,3),
        "Selected commercial output core (microns)": np.round(com_core_out, 3),
        "Uncer Selected commercial output core (microns)": np.round(e_com_core_out, 3),
        "Selected commercial output NA": np.round(com_NA_out, 3),
        "Expected efficiency": np.round(efficiency_comm,3),
        "Uncer Expected efficiency": e_efficiency_comm,
        "Total modules": n_modules_total,
        #"Total fibers at pseudorendija": total_fibers_pseudoslit,
        "Total cost PLs (MEUR)": np.round((cost_pl * total_modules) * 1e-6, 3),
        "Taper Ratio": np.array(taper_ratio_total),
        "Number of OTA for high efficiency": np.array(n_fiber_total),
        "Uncer Number of OTA for high efficiency": np.array(e_n_fiber_total(module_diameter_m, com_ota_diam_mm, n_fiber_total)),
        "Recalculated module diameter (m)": np.array(recal_module_diam),
        "Recalculated telescope aperture (m)": np.round(np.array(recal_telescope_aperture),3),

        # Spectrograph
        "": None,
        "Beam diameter at spectrograph (mm)": np.round(beam_diameter_mm,3),
        "Spectrograph volume (m³)": np.round(volume_m3_spec,3),
        "Uncer Spectrograph volume (m³)":np.round(e_volume_m3_spec, 3),
        "Spectrograph weight (kg)": np.round(mass_kg_spec,3),
        "Estimated cost spectograph (MEUR)": np.round(cost_meur_spec,3),
        "Resolution with commercial fibers": np.round(resolution(53, 31.6, beam_diameter_mm, com_core_out, magnification_factor)),
        "Uncer Resolution with commercial fibers": np.round(e_resolution(53, 31.6, beam_diameter_mm, com_core_out, e_com_core_out, magnification_factor), 3),
        "Detector size (mm)": np.round((np.array(detector_length(grooves_mm, f_cam_mm, theta_d, wavelength_max_nm, wavelength_min_nm))) * 1e3, 3),
        "Pixel size (microns)": pixel_size,
        "Spot size on detector (um)": np.round(com_core_out * magnification_factor / 3, 3),
        "Resolution element (px)": np.round(resolution_element, 3),
        # "S/N improvement factor": np.round(4.409 , 3),
        # If we are going to use a super PL, the S/N improvment factor is 1.6105 for 5m, 2.9463 for 10m and 4.4170 for 15m
        # If we use pseudoslit, this must be calculated
        # Fuse S/N calculator with this pipeline
        "Reduction factor in R (%)": np.round((1 - (resolution(53, 31.6, beam_diameter_mm, com_core_out, magnification_factor) / 94600 )) * 100,3),
        "Focal length camera (mm)": np.round(f_cam_mm, 3),

        # Auxiliar parameters
        "Cost telescope + PL (MEUR)": np.round(((cost_tel + cost_pl) * total_modules) * 1e-6, 3),
        "Use tip/tilt?": use_tip_tilt,
        "Magnification factor": magnification_factor,
        "Pseudoslit": pseudoslit,
        "Super-PL": super_pl,
        "Selected commercial output core 2-stage (microns)": np.round(com_core_out_2, 3),
        "Selected commercial output NA 2-stage": np.round(com_NA_out_2, 3)
    }
    # return results, module_diameter_m, com_core_out * 1e-3, com_NA_out, com_core_in * 1e-3, n_otas, efficiency_comm
    return results
        

