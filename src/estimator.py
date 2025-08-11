import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#############################################################################

def marcot_hr_estimator(
    module_diameter_m,
    telescope_aperture_m,
    seeing_fwhm_arcsec,
    use_tip_tilt,
    target_encircled_energy,
    wavelength_min_nm,
    wavelength_max_nm,
    grooves_mm,
    magnification_factor,
    beam_diameter_mm,
    pixel_size,
    f_cam_mm,
    pseudoslit = True,
    super_pl = False
):

    #########################################################################
    # 1. TELESCOPE MODULE
    #########################################################################

    com_ota_diam_mm, com_ota_f_number, com_ota_focal_length_mm, com_ota_tube_diam_mm, com_ota_tube_length_mm, com_ota_weight, com_ota_cost = np.loadtxt("data/Commercial_OTA.txt", usecols=(0, 1, 2, 3, 4, 5, 6),unpack=True)

    module_area = math.pi * (module_diameter_m / 2) ** 2
    ota_area = math.pi * (com_ota_diam_mm * 1e-3 / 2) ** 2
    n_otas = np.round(module_area / ota_area, 0)

    plate_scale_arcsec_per_mm = 206265 / com_ota_focal_length_mm
    new_plate_scale = 206265 / (com_ota_diam_mm * 5) # Recalculate the plate scale, using a focal adapter to pass to f/5

    effective_seeing_arcsec = seeing_fwhm_arcsec * (0.8 if use_tip_tilt else 1.0)
    psf_fwhm_mm = effective_seeing_arcsec / new_plate_scale
    fiber_core_mm = 1.5 * psf_fwhm_mm  # 1.5xFWHM to capture 90–95% energy
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
    
    print(f'first_commercial {first_commercial}')
    print(f'commercial_core {commercial_core}')
        
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
    
    print(f'com_core_in {com_core_in}')
    
    wavelengths_m = [wavelength_min_nm * 1e-9, wavelength_max_nm * 1e-9]

    def modes(l, d, NA):
      return (1 / 2) * ((((2 * np.pi * d) / (2 * l)) * NA) ** 2)

    archive = open("data/Com_fiber_out.txt", "w")
    archive.write(f'# Core_size(um) eCore_size(um) NA eNA Cost(€/m)\n')
    print(f'Número de OTAs {n_otas}')
    print(f'com_core_in {com_core_in}')
    print(f'com_NA_in {com_NA_in}')
    
    # We are going to recalculate the number of fiber for higher efficiency
    
    n_fiber_total = []
    def n_fiber(n):
        return 1 + 3 * n * (n + 1)
        
    for j in range(0, len(n_otas)):
        for i in range(1, 1000):
            if n_fiber(i) > n_otas[j]:
                n_fiber_total.append(n_fiber(i-1))
                break
    modes_total_per_module = n_fiber_total * modes(wavelength_min_nm * 1e-9, com_core_in * 1e-6, com_NA_in)
    
    for i in modes_total_per_module:
        for j in range(0,len(commercial_core)):
            modes_out_PL = modes(wavelength_min_nm * 1e-9, commercial_core[j] * 1e-6, commercial_NA[j])
            if modes_out_PL >= i:
                archive.write(f'{commercial_core[j]} {e_commercial_core[j]} {commercial_NA[j]} {e_commercial_NA[j]} {cost_eur_m[j]}\n')
                break

    archive.close()

    com_core_out, e_com_core_out, com_NA_out, e_com_NA_out, cost_eur_m_out = np.loadtxt("data/Com_fiber_out.txt", usecols=(0, 1, 2, 3, 4),unpack=True)

    loss_comm = 10 * np.log10( n_fiber_total * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in) / modes(wavelength_min_nm * 1e-9, com_core_out * 1e-6, com_NA_out))
    efficiency_comm = 1 - 10 ** (loss_comm / 10)

    fibers_per_module_output = modes_total_per_module / modes(wavelength_min_nm * 1e-9, com_core_out, com_NA_out)
    
    def evanescent(com_core_in, com_core_out, com_NA_out, n_otas, com_ota_diam_mm):
    
        # n_fiber_total = []
        taper_ratio_total = []
    
        # def n_fiber(n):
        #     return 1 + 3 * n * (n + 1)

        for j in range(0, len(n_otas)):
            for i in range(1, 1000):
                if n_fiber(i) > n_otas[j]:
                    # n_fiber_total.append(n_fiber(i))
                
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
    
        # lon = np.linspace(0,0.04,100)

        # fig,ax=plt.subplots()
    
        # for i in range(0, len(com_core_out)):
    
            # ax.plot(lon*10**2, (radius(125e-6 / 2, lon, alpha(125e-6 / 2, 125e-6 / (2 * taper_ratio_total[i]))) - radius(com_core_in * 1e-6 / 2, lon, alpha(com_core_in * 1e-6 / 2, com_core_in * 1e-6 / (2 * taper_ratio_total[i]))))*10**6,'orange')
    
            # ax.plot(lon*10**2, np.linspace(3.5,3.5,100),'--r')

        # for i in range(0,len(com_core_out)):
            # for j in range(0, len(lon)):
        
                # if radius(125e-6 / 2, lon[j], alpha(125e-6 / 2, 125e-6 / (2 * taper_ratio_total[i]))) - radius(com_core_in * 1e-6 / 2, lon[j], alpha(com_core_in * 1e-6 / 2, com_core_in * 1e-6 / (2 * taper_ratio_total[i])))<3.5e-6:
                
                    # print(f'La luz se empieza a mezclar a partir de los {np.round(lon[i],3)*10**2} cm')
                
                    # print(f'A esa altura, el diámetro de la fibra es de {np.round(radius(com_core_in * 1e-3 / 2,lon[j],alpha(com_core_in * 1e-3 / 2, com_core_in * 1e-3 / (2 * taper_ratio_total[i])))*2*10**6,2)} um')
                
                    # print(f'Una fibra con esas dimensiones es capáz de llevar {np.round(modes(0.655e-6,radius(com_core_in * 1e-3 / 2,lon[i],alpha(com_core_in * 1e-3 / 2, com_core_in * 1e-3 / (2 * taper_ratio_total[i]))),com_NA_out),2)} modos')
                
                # HAY QUE MEJORAR ESTO ANTES DE IMPRIMIR NINGÚN TEXTO
                
                    # ax.plot(lon[j]*10**2, (radius(125e-6 / 2, lon[j], alpha(125e-6 / 2, 125e-6 / (2 * taper_ratio_total[i]))) - radius(com_core_in * 1e-3 / 2, lon[j], alpha(com_core_in * 1e-3 / 2, com_core_in * 1e-3 / (2 * taper_ratio_total[i]))))*10**(6), 'b*')
                    # break

        # ax.set(xlabel='Length (cm)', ylabel='Thickness (um)')
        # ax.grid(which='minor', alpha=0)
        # ax.grid(which='major', alpha=0.5)
        # ax.tick_params(which='major', axis='x', direction='in', length=10, top=False)
        # ax.tick_params(which='major', axis='y', direction='in', length=10, right=False)
        # ax.tick_params(which='minor', axis='x', direction='in', length=6, top=False)
        # ax.tick_params(which='minor', axis='y', direction='in', length=6, right=False)
        # ax.set_title("Thickness along the tapering process")

        
        # plt.savefig("Images/Taper_length.png")
    
        # print("\033[1;4;32mFigure 'Taper_length.png' was saved successfully\033[0m")

        # plt.show(block=False)
        return taper_ratio_total, recal_module_diam, n_fiber_total
        
    taper_ratio_total, recal_module_diam, n_fiber_total = evanescent(com_core_in, com_core_out, com_NA_out, n_otas, com_ota_diam_mm)
    
    module_area = math.pi * (recal_module_diam / 2)**2
    
    total_modules = np.round((math.pi * (telescope_aperture_m / 2)**2) / module_area)
    recal_telescope_aperture = np.sqrt(total_modules) * recal_module_diam

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
        
    cost_pl = np.array(n_fiber_total) * 2 * np.array(cost_eur_m_in) + 3 * np.array(cost_eur_m_out)
    
    
    com_ota_weight, com_ota_cost = np.loadtxt("data/Com_ota_choosen.txt", usecols=(5,6),unpack=True)
    
    cost_tel = n_fiber_total * com_ota_cost
    
    cost_trad = 2.37 * ( module_diameter_m ** 1.96 ) # Equation to estimated cost of a telescope with the same diameter with traditional design
    
    frac_cost = cost_trad / ( cost_tel / 1e6 )
    
    weight_tel = n_fiber_total * (com_ota_weight + 10) # We sum 10 kg considering post-focus intrumentation
    
    #########################################################################
    # 3. SPECTROGRAPH / INSTRUMENT
    #########################################################################

    # This formula is to calculate the new resolution
    def resolution(n_order, lines, beam_diameter_mm, forced_output_core_microns, magnification_factor):
      N = lines * beam_diameter_mm
      w = lines * 0.5 * forced_output_core_microns * 1e-3 * magnification_factor
      # the value 0.5 is because in CARMENES we use slicer to divide in two parts the spot
      return ((n_order * N) / w) * 0.66666666666

    # beam_diameter_mm = 2 * 455 * np.tan(np.asin(com_NA_out)) * magnification_factor
    
    f_col_mm = beam_diameter_mm / (2 * np.tan(np.asin(com_NA_out)))
    
    # Calculate detector dimensions units in [mm]
    theta_B = np.atan(4) # Blaze angle using R4 grating
    theta_i = 75.2 * np.pi / 180 # Incident angle in [rad] taken from CARMENES
    theta_d = 2 * theta_B - theta_i # Dispersion angle [rad]
    
    f_cam_mm = f_col_mm / magnification_factor
    
    wavelength_range_m = np.linspace(wavelength_min_nm, wavelength_max_nm, wavelength_max_nm - wavelength_min_nm + 1)
    
    def detector_length(grooves_mm, f_cam_mm , theta_d, wavelength_max_nm, wavelength_min_nm):
        return (grooves_mm * 1e3 * (wavelength_max_nm * 1e-9 - wavelength_min_nm * 1e-9) * f_cam_mm * 1e-3) / np.cos(theta_d)
        
    def m (wavelength_range_m, grooves_mm, theta_i, theta_d):
        return (np.sin(theta_i) + np.sin(theta_d)) / (wavelength_range_m * grooves_mm)
    
   # def dispersion(wavelength_range_m, grooves_mm, theta_d, f_cam_mm):
   #     return np.cos(theta_d)  / (m(wavelength_range_m, grooves_mm, theta_i, theta_d) * f_cam_mm * grooves_mm)
        
    dispersion_nm_um = np.cos(theta_d) * (1e6 / grooves_mm) / (53 * f_cam_mm )
        
    resolution_element = 1 / (dispersion_nm_um * pixel_size / (wavelength_max_nm / resolution(53, 31.6, beam_diameter_mm, com_core_out, magnification_factor)))
        
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
        "Number of OTAs": n_otas,
        "F-number OTA": com_ota_f_number,
        "Focal length (mm)": com_ota_focal_length_mm,
        "Seeing FWHM (arcsec)": seeing_fwhm_arcsec,
        "Effective seeing (arcsec)": effective_seeing_arcsec if use_tip_tilt else seeing_fwhm_arcsec,
        "Input fiber core (microns)": np.round(fiber_core_microns,3),
        "Selected commercial input core (microns)": np.round(com_core_in, 3),
        "Selected commercial input NA": np.round(com_NA_in, 3),
        "Total cost for each module (MEUR)": np.round(cost_tel* 1e-6, 3),
        "Total cost (MEUR)": np.round((cost_tel * total_modules + cost_pl) * 1e-6, 3),
        "Reduction cost factor": np.round(frac_cost, 3),
        "Weight supported by the mount (kg)": np.round(weight_tel, 3),

        # Photonic Lantern
        "": None,
        "Modes per fiber [500nm,1000nm]": np.array([np.round(modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0),np.round(modes(wavelength_max_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        "Total modes per module": np.array([np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0), np.round(n_otas * modes(wavelength_min_nm * 1e-9, fiber_core_m, com_NA_in), 0)]),
        "Required output fiber core (microns)": np.round(com_core_out,3),
        "Selected commercial output core (microns)": np.round(com_core_out, 3),
        "Selected commercil output NA": np.round(com_NA_out, 3),
        "Expected efficiency (%)": np.round(efficiency_comm * 100,3),
        "Total modules": total_modules,
        "Total fibers at pseudorendija": total_fibers_pseudoslit,
        "Total cost PLs (MEUR)": np.round((cost_pl * total_modules) * 1e-6, 3),
        "Taper Ratio": np.array(taper_ratio_total),
        "Number of OTA for high efficiency": np.array(n_fiber_total),
        "Recalculated module diameter (m)": np.array(recal_module_diam),
        "Recalculated telescope aperture (m)": np.round(np.array(recal_telescope_aperture),3),

        # Spectrograph
        "": None,
        "Beam diameter at spectrograph (mm)": np.round(beam_diameter_mm,3),
        "Spectrograph volume (m³)": np.round(spectrograph_volume_m3,3),
        "Spectrograph weight (kg)": np.round(spectrograph_weight_kg,3),
        "Estimated cost (MEUR)": np.round(cost_estimate_meur,3),
        "Resolution with commercial fibers": np.round(resolution(55, 31.6, beam_diameter_mm, com_core_out, magnification_factor)),
        "Detector size (mm)": np.round((detector_length(grooves_mm, f_cam_mm, theta_d, wavelength_max_nm, wavelength_min_nm)) * 1e3, 3),
        "Spot size on detector (um)": np.round(com_core_out * magnification_factor / 3, 3),
        "Resolution element (px)": np.round(resolution_element, 3),
        # "S/N improvement factor": np.round(4.409 , 3),
        # If we are going to use a super PL, the S/N improvment factor is 1.6105 for 5m, 2.9463 for 10m and 4.4170 for 15m
        # If we use pseudoslit, this must be calculated
        # Fuse S/N calculator with this pipeline
        "Reduction factor in R (%)": np.round(resolution(55, 31.6, beam_diameter_mm, com_core_out, magnification_factor) / 94600 * 100,3),

        # Auxiliar parameters
        "Cost telescope + PL (MEUR)": np.round(((cost_tel + cost_pl) * total_modules) * 1e-6, 3),
        "Use tip/tilt?": use_tip_tilt,
        "Magnification factor": magnification_factor,
        "Pseudoslit": pseudoslit,
        "Super-PL": super_pl
    }
    # return results, module_diameter_m, com_core_out * 1e-3, com_NA_out, com_core_in * 1e-3, n_otas, efficiency_comm
    return results
        

