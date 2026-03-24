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
    focal_adapter,
    focal_adapter_locked,
    seeing_fwhm_arcsec,
    sky_aperture,
    sky_aperture_locked,
    use_tip_tilt,
    maximixe_PL,
    target_encircled_energy,
    wavelength_min_nm,
    wavelength_max_nm,
    pseudoslit,
    super_pl,
    grooves_mm,
    resolution_power,
    beam_diameter_mm,
    beam_diameter_mm_locked,
    pixel_size,
    rel_element,
    slicer,
    f_cam_mm,
    f_cam_mm_locked,
    f_coll_mm,
    incident_angle,
    echelle,
    nir_arm
):

    #########################################################################
    # 1. TELESCOPE MODULE
    #########################################################################

    name, com_ota_diam_mm, com_ota_f_number, com_ota_focal_length_mm, com_ota_tube_diam_mm, com_ota_effi, com_ota_tube_length_mm, com_ota_weight, com_ota_cost = np.genfromtxt("data/Commercial_OTA.txt", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8),dtype=None, unpack=True)
    
    e_module_diameter_m = 0.1
    module_area = math.pi * (module_diameter_m / 2) ** 2
    e_module_area = (math.pi / 2) * module_diameter_m * e_module_diameter_m
    
    e_com_ota_diam = 0.1 * 1e-3
    ota_area = math.pi * (com_ota_diam_mm * 1e-3 / 2) ** 2
    e_ota_area = (math.pi / 2) * com_ota_diam_mm * 1e-3 * e_com_ota_diam
   
    n_otas = np.round(module_area / ota_area, 0)
    e_n_otas = n_otas * np.sqrt((e_module_area / module_area) ** 2 + (e_ota_area / ota_area) ** 2)

    plate_scale_arcsec_per_mm = 206265 / com_ota_focal_length_mm
    new_plate_scale = 206265 / (com_ota_diam_mm * focal_adapter) # Recalculate the plate scale, using a focal adapter to pass to f/5

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
            if (i + i * 0.2) > j:
                first_commercial.append(i)
                break
    first_commercial = np.array(first_commercial)
        
    for j in first_commercial:
        if focal_adapter_locked == True:
            for i in range(0, len(commercial_core)):
                if i == 0:
                    archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
                if commercial_core[i] == j:
                    if commercial_NA[i] > (1/(2 * focal_adapter)):
                        archive.write(f'{commercial_core[i]} {e_commercial_core[i]} {commercial_NA[i]} {e_commercial_NA[i]} {cost_eur_m[i]}\n')
                    else:
                        archive.write(f'{commercial_core[i]} {e_commercial_core[i]} {1 / (2 * focal_adapter)} {e_commercial_NA[i]} {cost_eur_m[i] + 2500}\n')
                    break
        else:
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
        
    if maximixe_PL == True:
        for j in range(0, len(n_otas)):
            for i in range(1, 1000):
                if n_otas[j] > 37 and n_fiber(i) > n_otas[j]:
                    n_fiber_total.append(n_otas[j])
                    break
                elif n_otas[j] < 37 and n_fiber(i) > n_otas[j]:
                    n_fiber_total.append(n_fiber(i)) # If we want to be under the optimal fibre we have to put n_fiber(i-1)
                    break
    else:
        n_fiber_total = n_otas

    def e_n_fiber_total(module_diameter_m, com_ota_diam_mm, n):
        return n * np.sqrt((2 * 0.1 / module_diameter_m) ** 2 + (2 * 0.1 * 1e-3 / (com_ota_diam_mm * 1e-3)) ** 2)
    
    modes_total_per_module = n_fiber_total * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)
    
    # Set an array with possible output diameters between 0 and 1000 um
    
    core_diam_um = np.array(np.linspace(0,1000,1001))
    
    for i in modes_total_per_module:
        if f_number_out_locked:
            found = False
            for j in range(0,len(core_diam_um)):
                NA_PL = np.array(np.linspace(1 / (2 * f_number_out), 1 / (2 * f_number_out), len(core_diam_um)))
                modes_out_PL = modes(wavelength_min_nm * 1e-9, core_diam_um[j] * 1e-6, NA_PL[j])
                if modes_out_PL >= i:
                    archive.write(f'{core_diam_um[j]} {0} {np.round(NA_PL[j], 3)} {0} {0}\n')
                    found = True
                    break
                if found:
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
    
    n_total_modules = np.round((math.pi * (telescope_aperture_m / 2)**2) / module_area)
    
    # We are going to recalculate the number of modules for higher efficiency if we use a PL os second stage
    
    #n_modules_total = []

    #modes_total_per_module = n_fiber_total * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)

    #if pseudoslit:
    #n_modules_total = total_modules
    #elif super_pl:
    #    for j in range(0, len(n_otas)):
    #        for i in range(1, 1000):
    #            if n_fiber(i) > total_modules[j]:
    #                n_modules_total.append(n_fiber(i)) # If we want to be under the best fibre number, we have to change to n_fiber(i-1)
    #                break
    #else:
    #    n_modules_total = total_modules
    
    total_fibers_pseudoslit = n_total_modules * n_otas * 2 # We multiply by two because we have science and calibration fibres
        
        
    # We are going to calculate the output fiber required for a PL of second stage
            
    #com_core_out_2 = []
    #com_NA_out_2 = []
    
    #modes_total = modes_total_per_module * n_modules_total
    
    #for i in modes_total:
    #    if f_number_out_locked:
    #        found = False
    #        for j in range(0,len(core_diam_um)):
    #            NA_PL = np.array(np.linspace(1 / (2 * f_number_out), 1 / (2 * f_number_out), len(core_diam_um)))
    #            modes_out_PL = modes(wavelength_min_nm * 1e-9, core_diam_um[j] * 1e-6, NA_PL[j])
    #            if modes_out_PL >= i:
    #                com_core_out_2.append(core_diam_um[j])
    #                com_NA_out_2.append(NA_PL[j])
    #                found = True
    #                break
    #    else:
    #        for j in range(0,len(commercial_NA)):
    #            modes_out_PL = modes(wavelength_min_nm * 1e-9, commercial_core[j] * 1e-6, commercial_NA[j])
    #            if modes_out_PL >= i:
    #                com_core_out_2.append(commercial_core[j])
    #                com_NA_out_2.append(commercial_NA[j])
    #                break
                    
    #com_core_out_2 = np.array(com_core_out_2)
    #com_NA_out_2 = np.array(com_NA_out_2)
    
    #recal_telescope_aperture = np.sqrt(n_modules_total) * recal_module_diam
    
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
    
    if beam_diameter_mm_locked == False:
        if super_pl == True:
            beam_diameter_mm = 2 * f_coll_mm * np.tan(np.asin(com_NA_out)) + com_core_out * 1e-3
        else:
            beam_diameter_mm = 2 * f_coll_mm * np.tan(np.asin(com_NA_in * 0.9)) + com_core_in * 1e-3 # We multiply 0.9 in com_NA_in to take into account the FRD of 10%

    magnification_factor = f_cam_mm / f_coll_mm
    e_magnification_factor = magnification_factor * np.sqrt((0.01 / f_cam_mm) ** 2 + (0.01 / f_coll_mm) ** 2)
    
    # Calculate detector dimensions units in [mm]
    if echelle:
        theta_B = np.arctan(4) # Blaze angle using R4 grating
        theta_i = incident_angle * np.pi / 180 # Incident angle in [rad]
        theta_d = 2 * theta_B - theta_i # Dispersion angle [rad]
    else:
        theta_i = incident_angle * np.pi / 180
        theta_d = (90 * np.pi / 180) - theta_i
    
    # For computing the order number
    def m(wavelength, grooves_mm, theta_i, theta_d):
        order = ((np.sin(theta_i) + np.sin(theta_d)) / (wavelength * 1e-9 * grooves_mm * 1e3))
        return order
    
    def e_m(wavelength, grooves_mm, theta_i, theta_d):
        return m(wavelength, grooves_mm, theta_i, theta_d) * np.sqrt(((np.cos(theta_i) * 0.01) / (np.sin(theta_i) + np.sin(theta_d))) ** 2 + ((np.cos(theta_d) * 0.01) / (np.sin(theta_i) + np.sin(theta_d))) ** 2 + (0.1 * 1e-9 / wavelength * 1e-9) ** 2 + (0.01 * 1e3/ (grooves_mm * 1e3)) ** 2)
        
    # This formula is to calculate the new resolution
    def resolution(f_coll_mm, grooves_mm, forced_output_core_microns, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i):
    
        if slicer == True:
            width = magnification_factor * forced_output_core_microns * 0.5 * 1e-6
        else:
            width = magnification_factor * forced_output_core_microns * 1e-6
            
        dlambda = np.cos(theta_d) / (grooves_mm * 1e3 * m(wavelength_max_nm, grooves_mm, theta_i, theta_d) * f_coll_mm * 1e-3)
      # the value 0.5 is because in CARMENES we use slicer to divide in two parts the spot
        return wavelength_max_nm * 1e-9 / (width * dlambda)
      
    def e_resolution(f_coll_mm, grooves_mm, forced_output_core_microns, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i, e_forced_output_core_microns):
        return resolution(f_coll_mm, grooves_mm, forced_output_core_microns, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i) * np.sqrt(((0.01 * 1e3) / (grooves_mm * 1e3) ** 2 + (e_m(wavelength_max_nm, grooves_mm, theta_i, theta_d) / m(wavelength_max_nm, grooves_mm, theta_i, theta_d)) ** 2 + (0.01 / f_coll_mm) ** 2 + (0.1 * 1e-9 / ( wavelength_max_nm * 1e-9)) ** 2 + (np.tan(theta_d) * 0.01) ** 2 + (e_magnification_factor / magnification_factor) ** 2 + (e_forced_output_core_microns / forced_output_core_microns) ** 2))

    if super_pl:
        reduction_on_R = (resolution(f_coll_mm, grooves_mm, com_core_out, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i) / resolution_power) * 100
    else:
        reduction_on_R = (resolution(f_coll_mm, grooves_mm, com_core_in, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i) / resolution_power) * 100
        
    def detector_length(grooves_mm, f_cam_mm , theta_d, wavelength_max_nm, wavelength_min_nm):
        if pseudoslit:
            detector_size_mm = total_fibers_pseudoslit * (com_core_in + 81) * 1e-3 * 2
        else:
            detector_size_mm = (grooves_mm * 1e3 * (wavelength_max_nm * 1e-9 - wavelength_min_nm * 1e-9) * f_cam_mm * 1e-3) / np.cos(theta_d)
        return detector_size_mm
        
    #Calculate the projected fibre spot on the detector
        
    if slicer == True:
        if super_pl == True:
            spot = (com_core_out * magnification_factor) / 2
        else:
            spot = (com_core_in * magnification_factor) / 2
    else:
        if super_pl == True:
            spot = com_core_out * magnification_factor
        else:
            spot = com_core_in * magnification_factor
                        
    dispersion_nm_um = np.cos(theta_d) * (1e6 / grooves_mm) / ((m(wavelength_min_nm, grooves_mm, theta_i, theta_d) - m(wavelength_max_nm, grooves_mm, theta_i, theta_d)) * np.array(f_cam_mm))
            
    resolution_element = 1 / (dispersion_nm_um * pixel_size / (wavelength_max_nm / resolution(f_coll_mm, grooves_mm, com_core_out, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i)))
    
    if super_pl == True:
        final_resolution = resolution(f_coll_mm, grooves_mm, com_core_out, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i)
        e_final_resolution = e_resolution(f_coll_mm, grooves_mm, com_core_out, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i, e_com_NA_out)
    else:
        final_resolution = resolution(f_coll_mm, grooves_mm, com_core_in, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i)
        e_final_resolution = e_resolution(f_coll_mm, grooves_mm, com_core_in, magnification_factor, slicer, theta_d, wavelength_max_nm, theta_i, e_com_NA_in)
        
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
        "Alternative": name,
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
        "Uncer Selected commercial input core (microns)": np.round(e_com_core_in, 3),
        "Selected commercial input NA": np.round(com_NA_in, 3),
        "Sky aperture (arcsec)": np.round(sky_aperture, 3),
        "Total cost for each module (MEUR)": np.round((cost_tel * 1e-6), 3),
        "Total cost (MEUR)": np.round(((cost_tel * 1e-6) * n_total_modules + cost_pl * 1e-6), 3),
        "Reduction cost factor": np.round(frac_cost, 3),
        "Uncer Reduction cost factor": np.round(e_frac_cost, 3),
        "Weight supported by the mount (kg)": np.round(weight_tel, 3),
        "Uncer Weight supported by the mount (kg)": np.round(e_weight_tel, 3),

        # Photonic Lantern
        "": None,
        "Required output fiber core (microns)": np.round(com_core_out,3),
        "Selected commercial output core (microns)": np.round(com_core_out, 3),
        "Uncer Selected commercial output core (microns)": np.round(e_com_core_out, 3),
        "Selected commercial output NA": np.round(com_NA_out, 3),
        "Expected efficiency": np.round(efficiency_comm,3),
        "Uncer Expected efficiency": e_efficiency_comm,
        "Total modules": n_total_modules,
        #"Total fibers at pseudorendija": total_fibers_pseudoslit,
        "Total cost PLs (MEUR)": np.round((cost_pl * n_total_modules) * 1e-6, 3),
        #"Taper Ratio": np.array(taper_ratio_total),
        "Number of OTA for high efficiency": np.array(n_fiber_total),
        "Uncer Number of OTA for high efficiency": np.array(e_n_fiber_total(module_diameter_m, com_ota_diam_mm, n_fiber_total)),
        "Recalculated module diameter (m)": np.array(recal_module_diam),
        #"Recalculated telescope aperture (m)": np.round(np.array(recal_telescope_aperture),3),

        # Spectrograph
        "": None,
        "Beam diameter at spectrograph (mm)": np.round(beam_diameter_mm,3),
        "Spectrograph volume (m³)": np.round(volume_m3_spec,3),
        "Uncer Spectrograph volume (m³)":np.round(e_volume_m3_spec, 3),
        "Spectrograph weight (kg)": np.round(mass_kg_spec,3),
        "Estimated cost spectograph (MEUR)": np.round(cost_meur_spec,3),
        "Resolution with commercial fibers": np.round(final_resolution),
        "Uncer Resolution with commercial fibers": np.round(e_final_resolution, 3),
        "Detector size (mm)": np.round(detector_length(grooves_mm, f_cam_mm , theta_d, wavelength_max_nm, wavelength_min_nm), 3),
        "Spot size on detector (um)": np.round(spot, 3),
        "Resolution element (px)": np.round(resolution_element, 3),
        # "S/N improvement factor": np.round(4.409 , 3),
        # If we are going to use a super PL, the S/N improvment factor is 1.6105 for 5m, 2.9463 for 10m and 4.4170 for 15m
        # If we use pseudoslit, this must be calculated
        # Fuse S/N calculator with this pipeline
        "Reduction factor in R (%)": np.round(reduction_on_R,3),
        "Focal length camera (mm)": np.round(f_cam_mm, 3),

        # Auxiliar parameters
        "Cost telescope + PL (MEUR)": np.round(((cost_tel + cost_pl) * n_total_modules) * 1e-6, 3),
        "Use tip/tilt?": use_tip_tilt,
        "Magnification factor": magnification_factor,
        #"Pseudoslit": pseudoslit,
        #"Super-PL": super_pl,
        #"Selected commercial output core 2-stage (microns)": np.round(com_core_out_2, 3),
        #"Selected commercial output NA 2-stage": np.round(com_NA_out_2, 3)
    }
    # return results, module_diameter_m, com_core_out * 1e-3, com_NA_out, com_core_in * 1e-3, n_otas, efficiency_comm
    return results
        

