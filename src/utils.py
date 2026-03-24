import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr
from scipy.integrate import quad
import matplotlib.pyplot as plt
import ast
import re
import seaborn as sns

from pathlib import Path


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

def multi_criteria(results_csv_path, criteria):
    # In case no path, we use a default path6
    if results_csv_path is None:
        results_csv_path = RES("data", "results_marcot.csv")
        
    df = pd.read_csv(results_csv_path, sep='\t')
    #if criteria is None:
    #    print('hola, no hay criteria otra vez')
    #    criteria = [
    #        ("Reduction cost factor", "benefit"),
    #        ("Weight supported by the mount (kg)", "cost"),
    #        ("Selected commercial output core (microns)", "cost"),
    #        ("Expected efficiency", "benefit"),
    #        ("Resolution with commercial fibers", "benefit"),
    #        ("SNR fraction", "benefit"),
    #        ("Number of OTA for high efficiency", "cost")
    #    ]
    #
    #    criteria = {
    #        name: {"type": kind}
    #        for name, kind in criteria
    #    }

    
    more = []
    less = []
    for c, type_str in criteria.items():
        if type_str['type'] == "benefit":
            more.append(c)
        else:
            less.append(c)
        
    # --- Creating decision matrix ---
    column = []
    uncer_column = []
    column.append('OTA diameter (m)')
    uncer_column.append('Uncer OTA diameter (m)')
    
    for crit in criteria:
        column.append(crit)
        uncer_column.append(f"Uncer {crit}")
        
            
    df_decision_matrix = df[column]
    df_uncer_decision_matrix = df[uncer_column]
    df_decision_matrix = df_decision_matrix.set_index('OTA diameter (m)')
    df_uncer_decision_matrix = df_uncer_decision_matrix.set_index('Uncer OTA diameter (m)')
    
    # --- Weight Determination ---
    len_alternative, len_criteria = np.shape(df_decision_matrix)
    # --- Statitical Variance & CRITIC methods---
    V = []
    C_l = []
    C_m = []
    C_u = []
    
    # First, we need to normalize decision matrix
    for crit in criteria:
        x = df_decision_matrix[crit].to_numpy(dtype=float)
        u = df_uncer_decision_matrix[f"Uncer {crit}"].to_numpy(dtype=float)

        x_l = x - u
        x_m = x
        x_u = x + u
        x_star_u = x_u.max()
        x_less_l = x_l.min()

        if crit in more:
            # For st method
            l_st = x_l / x_star_u
            m_st = x_m / x_star_u
            uu_st = x_u / x_star_u
            
            # For CRITIC method
            l_crt = (x_l - x_less_l) / (x_star_u - x_less_l)
            m_crt = (x_m - x_less_l) / (x_star_u - x_less_l)
            uu_crt = (x_u - x_less_l) / (x_star_u - x_less_l)
            
            # For MEREC method
            l_mrc = x_less_l / x_u
            m_mrc = x_less_l / x_m
            uu_mrc = x_less_l / x_l
        else:
            # For st method
            l_st = x_less_l / x_u
            m_st = x_less_l / x_m
            uu_st = x_less_l / x_l
            
            # For CRITIC method
            l_crt = (x_star_u - x_u) / (x_star_u - x_less_l)
            m_crt = (x_star_u - x_m) / (x_star_u - x_less_l)
            uu_crt = (x_star_u - x_l) / (x_star_u - x_less_l)
            
            # For MEREC method
            l_mrc = x_l / x_star_u
            m_mrc = x_m / x_star_u
            uu_mrc = x_u / x_star_u
            
        df_decision_matrix[crit + "_fuzzy"] = list(map(np.array, np.column_stack([x_l, x_m, x_u])))
        
        df_decision_matrix[crit + "_norm_st"] = list(map(np.array, np.column_stack([l_st, m_st, uu_st])))
        
        df_decision_matrix[crit + "_norm_CRITIC"] = list(map(np.array, np.column_stack([l_crt, m_crt, uu_crt])))
        
        df_decision_matrix[crit + "_norm_MEREC"] = list(map(np.array, np.column_stack([l_mrc, m_mrc, uu_mrc])))
        
    df_decision_matrix.to_csv("data/decision_matrix.csv", sep="\t", index=False)
    
        
    for criterio in criteria:
        #all_values = df_decision_matrix[criterio].values
        st_fuzzy_values = df_decision_matrix[criterio + "_norm_st"].values
        crt_fuzzy_values = df_decision_matrix[criterio + "_norm_CRITIC"]
        
        cuadratic = []
        
        # This is to calculating V in the st method
        for i in st_fuzzy_values:
            cuadratic.append((i - np.mean(st_fuzzy_values)) ** 2)
        V.append((1 / len_alternative) * sum(cuadratic))
        
        # This is for calculating the correlation in CRITIC method
        correlation_l = []
        correlation_m = []
        correlation_u = []
        for j, second_criterio in enumerate(criteria):
            second_all_values = df_decision_matrix[second_criterio + "_norm_CRITIC"]
            
            if criterio == second_criterio:
                pass
            else:
                crt_values = np.vstack(crt_fuzzy_values.to_numpy())
                l_1 = crt_values[:, 0].astype(float)
                m_1 = crt_values[:, 1].astype(float)
                u_1 = crt_values[:, 2].astype(float)
                
                crt_values_2 = np.vstack(second_all_values.to_numpy())
                l_2 = crt_values_2[:, 0].astype(float)
                m_2 = crt_values_2[:, 1].astype(float)
                u_2 = crt_values_2[:, 2].astype(float)
                
                corr_l, p_value = pearsonr(l_1, l_2)
                correlation_l.append(corr_l)
                
                corr_m, p_value = pearsonr(m_1, m_2)
                correlation_m.append(corr_m)
                
                corr_u, p_value = pearsonr(u_1, u_2)
                correlation_u.append(corr_u)
                
        correlation_l = np.array(correlation_l)
        correlation_m = np.array(correlation_m)
        correlation_u = np.array(correlation_u)

        C_l.append(np.std(l_1) * np.nansum(1 - correlation_l))
        C_m.append(np.std(m_1) * np.nansum(1 - correlation_m))
        C_u.append(np.std(u_1) * np.nansum(1 - correlation_u))
        
    V = np.array(V)
    
    st_variance_weight = []
    st_variance_weight.append(V[:,0] / np.nansum(V[:,2]))
    st_variance_weight.append(V[:,1] / np.nansum(V[:,1]))
    st_variance_weight.append(V[:,2] / np.nansum(V[:,0]))
    st_variance_weight = np.array(st_variance_weight).T
    st_variance_weight = np.sort(st_variance_weight)
    st_variance_weight_defu = (st_variance_weight[:,0] + 4 * st_variance_weight[:,1] + st_variance_weight[:,2]) / 6

    C = np.array([np.array(C_l), np.array(C_m), np.array(C_u)])
    C = C.T
    CRITIC_weight = []
    CRITIC_weight.append(C[:,0] / np.nansum(C[:,2]))
    CRITIC_weight.append(C[:,1] / np.nansum(C[:,1]))
    CRITIC_weight.append(C[:,2] / np.nansum(C[:,0]))
    CRITIC_weight = np.array(CRITIC_weight).T
    CRITIC_weight = np.sort(CRITIC_weight)
    CRITIC_weight_defu = (CRITIC_weight[:,0] + 4 * CRITIC_weight[:,1] + CRITIC_weight[:,2]) / 6
    
    # --- MEREC method ---
    E = []
    S_component_l = []
    S_component_m = []
    S_component_u = []
    S_alternative_l = []
    S_alternative_m = []
    S_alternative_u = []
    mcr_values = []
    
    for crit in criteria:
        mrc_fuzzy_values = df_decision_matrix[crit + "_norm_MEREC"]
        mcr_values.append(mrc_fuzzy_values)
    
    mcr_values = np.array(mcr_values)

    for i in range(0, len_alternative):
        alternative_values= mcr_values[:, i]
    
        all_values = np.vstack(alternative_values)
        l = all_values[:, 0].astype(float)
        m = all_values[:, 1].astype(float)
        u = all_values[:, 2].astype(float)
    
        S_component_l.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in l)))
        S_component_m.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in m)))
        S_component_u.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in u)))
        
        S_al_l = []
        S_al_m = []
        S_al_u = []
        for j, criterio in enumerate(criteria):
            new_values_l = [x for i, x in enumerate(l) if i != j]
            S_al_l.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in new_values_l)))
            new_values_m = [x for i, x in enumerate(m) if i != j]
            S_al_m.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in new_values_m)))
            new_values_u = [x for i, x in enumerate(u) if i != j]
            S_al_u.append(math.log(1 + (1 / len_criteria) * sum(abs(math.log(x)) for x in new_values_u)))
            
        S_al_l = np.array(S_al_l)
        S_alternative_l.append(S_al_l)
        S_al_m = np.array(S_al_m)
        S_alternative_m.append(S_al_m)
        S_al_u = np.array(S_al_u)
        S_alternative_u.append(S_al_u)
        
    S_alternative_l = np.array(S_alternative_l)
    S_alternative_l = S_alternative_l.T
    S_alternative_m = np.array(S_alternative_m)
    S_alternative_m = S_alternative_m.T
    S_alternative_u = np.array(S_alternative_u)
    S_alternative_u = S_alternative_u.T
    
    S_component_l = np.array(S_component_l)
    S_component_m = np.array(S_component_m)
    S_component_u = np.array(S_component_u)
    
    E_crit_l = []
    E_crit_m = []
    E_crit_u = []
    for i in range(0, len_criteria):
        
        all_values_l = S_alternative_l[i]
        all_values_m = S_alternative_m[i]
        all_values_u = S_alternative_u[i]
                        
        E_crit_l.append(sum(abs(all_values_l - S_component_l)))
        E_crit_m.append(sum(abs(all_values_m - S_component_m)))
        E_crit_u.append(sum(abs(all_values_u - S_component_u)))
        

    E_crit = [np.array(E_crit_l), np.array(E_crit_m), np.array(E_crit_u)]
    E_crit = np.array(E_crit)
    E_crit = E_crit.T
    
    MEREC_weight = []
    MEREC_weight.append(E_crit[:,0] / np.nansum(E_crit[:,2]))
    MEREC_weight.append(E_crit[:,1] / np.nansum(E_crit[:,1]))
    MEREC_weight.append(E_crit[:,2] / np.nansum(E_crit[:,0]))
    MEREC_weight = np.array(MEREC_weight).T
    MEREC_weight = np.sort(MEREC_weight)
    MEREC_weight_defu = (MEREC_weight[:,0] + 4 * MEREC_weight[:,1] + MEREC_weight[:,2]) / 6

    #---- TOPSIS ---
    
    I_criteria_st = []
    I_criteria_crt = []
    I_criteria_mcr = []
        
    for j, crit in enumerate(criteria):
        st_fuzzy_values = df_decision_matrix[crit + "_norm_st"].values
        st_fuzzy_values = np.vstack(st_fuzzy_values)
        st_fuzzy_values_defu = (st_fuzzy_values[:,0] + 4 * st_fuzzy_values[:,1] + st_fuzzy_values[:,2]) / 6
        
        new_st_fuzzy_values = np.vstack(st_fuzzy_values)

        I_alternative_st = []
        I_alternative_crt = []
        I_alternative_mcr = []

        for i in range(0, len_alternative):
            I_alternative_st.append(st_fuzzy_values[i] * st_variance_weight[j,:])
            I_alternative_crt.append(st_fuzzy_values[i] * CRITIC_weight[j,:])
            I_alternative_mcr.append(st_fuzzy_values[i] * MEREC_weight[j,:])

        I_criteria_st.append(np.array(I_alternative_st))
        I_criteria_crt.append(np.array(I_alternative_crt))
        I_criteria_mcr.append(np.array(I_alternative_mcr))


    I_criteria_st = np.array(I_criteria_st)
    I_criteria_crt = np.array(I_criteria_crt)
    I_criteria_mcr = np.array(I_criteria_mcr)
    
    A_star_st = np.array(np.max(I_criteria_st[:, :, 2], axis = 1))
    A_less_st = np.array(np.min(I_criteria_st[:, :, 0], axis = 1))
    A_star_crt = np.array(np.max(I_criteria_crt[:, :, 2], axis = 1))
    A_less_crt = np.array(np.min(I_criteria_crt[:, :, 0], axis = 1))
    A_star_mrc = np.array(np.max(I_criteria_mcr[:, :, 2], axis = 1))
    A_less_mrc = np.array(np.min(I_criteria_mcr[:, :, 0], axis = 1))
    
    # Calculate teh border approximation area (MABAC)
    g_st = (np.prod(I_criteria_st, axis=1)) ** (1 / len_alternative)
    g_crt = (np.prod(I_criteria_crt, axis=1)) ** (1 / len_alternative)
    g_mrc = (np.prod(I_criteria_mcr, axis=1)) ** (1 / len_alternative)
    
    # Calculate the distances (MABAC)
    q_st = I_criteria_st - g_st[:, np.newaxis, :]
    q_crt = I_criteria_crt - g_crt[:, np.newaxis, :]
    q_mrc = I_criteria_mcr - g_mrc[:, np.newaxis, :]
    
    # Calculate the total score (MABAC)
    S_st = sum(q_st)
    S_st = (S_st[:,0] + 4 * S_st[:,1] + S_st[:,2]) / 6
    S_crt = sum(q_crt)
    S_crt = (S_crt[:,0] + 4 * S_crt[:,1] + S_crt[:,2]) / 6
    S_mrc = sum(q_mrc)
    S_mrc = (S_mrc[:,0] + 4 * S_mrc[:,1] + S_mrc[:,2]) / 6

    df["score_total_st_MABAC"] = pd.Series(S_st, index=range(S_st.size))
    
    df["score_total_crt_MABAC"] = pd.Series(S_crt, index=range(S_crt.size))
    
    df["score_total_mrc_MABAC"] = pd.Series(S_mrc, index=range(S_mrc.size))
            
    # Calculate the ideal and anti-ideal distances (TOPSIS)
    d_star_st = []
    d_less_st = []
    d_star_crt = []
    d_less_crt = []
    d_star_mrc = []
    d_less_mrc = []
    
    for j, crit in enumerate(criteria):
        i_st = I_criteria_st[j]
        a_star_st = A_star_st[j]
        a_less_st = A_less_st[j]
        
        i_crt = I_criteria_crt[j]
        a_star_crt = A_star_crt[j]
        a_less_crt = A_less_crt[j]
        
        i_mrc = I_criteria_mcr[j]
        a_star_mrc = A_star_mrc[j]
        a_less_mrc = A_less_mrc[j]
        
        d_star_st.append(np.sqrt(1/3 * ((a_star_st - i_st[:,0]) ** 2 + (a_star_st - i_st[:,1]) ** 2 + (a_star_st - i_st[:,2]) ** 2)))
        d_star_crt.append(np.sqrt(1/3 * ((a_star_crt - i_crt[:,0]) ** 2 + (a_star_crt - i_crt[:,1]) ** 2 + (a_star_crt - i_crt[:,2]) ** 2)))
        d_star_mrc.append(np.sqrt(1/3 * ((a_star_mrc - i_mrc[:,0]) ** 2 + (a_star_mrc - i_mrc[:,1]) ** 2 + (a_star_mrc - i_mrc[:,2]) ** 2)))
        
        d_less_st.append(np.sqrt(1/3 * ((a_less_st - i_st[:,0]) ** 2 + (a_less_st - i_st[:,1]) ** 2 + (a_less_st - i_st[:,2]) ** 2)))
        d_less_crt.append(np.sqrt(1/3 * ((a_less_crt - i_crt[:,0]) ** 2 + (a_less_crt - i_crt[:,1]) ** 2 + (a_less_crt - i_crt[:,2]) ** 2)))
        d_less_mrc.append(np.sqrt(1/3 * ((a_less_mrc - i_mrc[:,0]) ** 2 + (a_less_mrc - i_mrc[:,1]) ** 2 + (a_less_mrc - i_mrc[:,2]) ** 2)))
        
        
    D_star_st = sum(d_star_st)
    D_less_st = sum(d_less_st)
    D_star_crt = sum(d_star_crt)
    D_less_crt = sum(d_less_crt)
    D_star_mrc = sum(d_star_mrc)
    D_less_mrc = sum(d_less_mrc)

    
    CC_st = D_less_st / (D_less_st + D_star_st)
    CC_crt = D_less_crt / (D_less_crt + D_star_crt)
    CC_mrc = D_less_mrc / (D_less_mrc * D_star_mrc)

    df["score_total_st_TOPSIS"] = pd.Series(CC_st, index=range(CC_st.size))
    
    df["score_total_crt_TOPSIS"] = pd.Series(CC_crt, index=range(CC_crt.size))
    
    df["score_total_mrc_TOPSIS"] = pd.Series(CC_mrc, index=range(CC_mrc.size))
        
    df.to_csv("data/score_total.csv", sep = '\t', index = False)

    
    df_sorted_st_TOPSIS = df.sort_values(by="score_total_st_TOPSIS", ascending=False)
    df_sorted_crt_TOPSIS = df.sort_values(by="score_total_crt_TOPSIS", ascending=False)
    df_sorted_mrc_TOPSIS = df.sort_values(by="score_total_mrc_TOPSIS", ascending=False)

    df_sorted_st_MABAC = df.sort_values(by="score_total_st_MABAC", ascending=False)
    df_sorted_crt_MABAC = df.sort_values(by="score_total_crt_MABAC", ascending=False)
    df_sorted_mrc_MABAC = df.sort_values(by="score_total_mrc_MABAC", ascending=False)
    
    df_sorted_st_TOPSIS.to_csv("data/score_total_st_TOPSIS.csv", sep = '\t', index = False)
    df_sorted_crt_TOPSIS.to_csv("data/score_total_crt_TOPSIS.csv", sep = '\t', index = False)
    df_sorted_mrc_TOPSIS.to_csv("data/score_total_mrc_TOPSIS.csv", sep = '\t', index = False)

    df_sorted_st_MABAC.to_csv("data/score_total_st_MABAC.csv", sep = '\t', index = False)
    df_sorted_crt_MABAC.to_csv("data/score_total_crt_MABAC.csv", sep = '\t', index = False)
    df_sorted_mrc_MABAC.to_csv("data/score_total_mrc_MABAC.csv", sep = '\t', index = False)

    print("\033[1;4;32mFile 'score_total.csv' was saved successfully\033[0m")
    
    scores_st = df["score_total_st_TOPSIS"].values

    best_index_st = np.argmax(scores_st)
    
    scores_crt = df["score_total_crt_TOPSIS"].values

    best_index_crt = np.argmax(scores_crt)
    
    scores_mrc = df["score_total_mrc_TOPSIS"].values

    best_index_mrc = np.argmax(scores_mrc)
        
    print("\n\033[1m\033[4mBest configuration found:\033[0m\n")
    
#########################################################################
# PLOTS
#########################################################################
    ota_diam_mm = (df["OTA diameter (m)"].values) * 1000
    cost = df["Total cost (MEUR)"].values
    n_fiber_total = df['Number of OTA for high efficiency'].values
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        
    for i in range(0, len(cost)):
    
        if i in (best_index_st, best_index_crt, best_index_mrc):
            pass
        else:
            ax.plot(ota_diam_mm[i], cost[i], 'o', label = f'{int(n_fiber_total[i])} OTAs of {int(ota_diam_mm[i])} mm')
            
    ax.plot(ota_diam_mm[best_index_st], cost[best_index_st], '*', ms = 20, label = f'Best fit St. Variance: {int(n_fiber_total[best_index_st])} OTAs of {int(ota_diam_mm[best_index_st])} mm')
    ax.plot(ota_diam_mm[best_index_crt], cost[best_index_crt], '*', ms = 20, label = f'Best fit CRITIC: {int(n_fiber_total[best_index_crt])} OTAs of {int(ota_diam_mm[best_index_crt])} mm')
    ax.plot(ota_diam_mm[best_index_mrc], cost[best_index_mrc], '*', ms = 20, label = f'Best fit MEREC: {int(n_fiber_total[best_index_mrc])} OTAs of {int(ota_diam_mm[best_index_mrc])} mm')
    
    plt.title('Cost (MEUR) vs OTA\'s diameter')
    plt.xlabel('OTA\'s diameter (mm)')
    plt.ylabel('Cost (MEUR)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="upper right")
    
    plt.savefig("Figures/Cost_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Cost_vs_Aperture.png' was saved successfully\033[0m")

    plt.close()

    recalculated_module_diameter_m = df["Recalculated module diameter (m)"].values
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        
    for i in range(0, len(cost)):
    
        if i in (best_index_st, best_index_crt, best_index_mrc):
            pass
        else:
            ax.plot(ota_diam_mm[i], recalculated_module_diameter_m [i], 'o', label = f'{int(n_fiber_total[i])} OTAs of {int(ota_diam_mm[i])} mm')
            
    ax.plot(ota_diam_mm[best_index_st], recalculated_module_diameter_m [best_index_st], '*', ms = 20, label = f'Best fit St. Variance: {int(n_fiber_total[best_index_st])} OTAs of {int(ota_diam_mm[best_index_st])} mm')
    ax.plot(ota_diam_mm[best_index_crt], recalculated_module_diameter_m [best_index_crt], '*', ms = 20, label = f'Best fit CRITIC: {int(n_fiber_total[best_index_crt])} OTAs of {int(ota_diam_mm[best_index_crt])} mm')
    ax.plot(ota_diam_mm[best_index_mrc], recalculated_module_diameter_m [best_index_mrc], '*', ms = 20, label = f'Best fit MEREC: {int(n_fiber_total[best_index_mrc])} OTAs of {int(ota_diam_mm[best_index_mrc])} mm')
    
    plt.title('Module size (m) vs OTA\'s diameter')
    plt.xlabel('OTA\'s diameter (mm)')
    plt.ylabel('Module aperture (m)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="upper right")
    
    plt.savefig("Figures/Module_size_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Module_size_vs_Aperture.png' was saved successfully\033[0m")

    plt.close()
    
    out_core_PL = df['Required output fiber core (microns)'].values
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        
    for i in range(0, len(cost)):
    
        if i in (best_index_st, best_index_crt, best_index_mrc):
            pass
        else:
            ax.plot(ota_diam_mm[i], out_core_PL[i], 'o', label = f'{int(n_fiber_total[i])} OTAs of {int(ota_diam_mm[i])} mm')
            
    ax.plot(ota_diam_mm[best_index_st], out_core_PL[best_index_st], '*', ms = 20, label = f'Best fit St. Variance: {int(n_fiber_total[best_index_st])} OTAs of {int(ota_diam_mm[best_index_st])} mm')
    ax.plot(ota_diam_mm[best_index_crt], out_core_PL[best_index_crt], '*', ms = 20, label = f'Best fit CRITIC: {int(n_fiber_total[best_index_crt])} OTAs of {int(ota_diam_mm[best_index_crt])} mm')
    ax.plot(ota_diam_mm[best_index_mrc], out_core_PL[best_index_mrc], '*', ms = 20, label = f'Best fit MEREC: {int(n_fiber_total[best_index_mrc])} OTAs of {int(ota_diam_mm[best_index_mrc])} mm')
    
    plt.title('PL core vs OTA\'s diameters')
    plt.xlabel('OTA\'s diameter (mm)')
    plt.ylabel(f'PL core diameter (microns)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    lgnd = plt.legend(loc="upper right")
    
    plt.savefig("Figures/Core_PL_vs_Aperture.png")
    
    print("\033[1;4;32mFigure 'Core_PL_vs_Aperture.png' was saved successfully\033[0m")

    plt.close()
    
    cost_module = df['Total cost for each module (MEUR)'].values
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        
    for i in range(0, len(cost)):
    
        if i in (best_index_st, best_index_crt, best_index_mrc):
            pass
        else:
            ax.plot(recalculated_module_diameter_m[i], cost_module[i], 'o', label = f'{int(n_fiber_total[i])} OTAs of {int(ota_diam_mm[i])} mm')
            ax.plot(recalculated_module_diameter_m[i], 2.37 * (recalculated_module_diameter_m[i]) ** 1.96, 'o', mec = 'black', label = f'Traditional cost for {int(recalculated_module_diameter_m[i])} m module')
            
    ax.plot(recalculated_module_diameter_m[best_index_st], cost_module[best_index_st], '*', ms = 20, label = f'Best fit St. Variance: {int(n_fiber_total[best_index_st])} OTAs of {int(ota_diam_mm[best_index_st])} mm')
    ax.plot(recalculated_module_diameter_m[best_index_crt], cost_module[best_index_crt], '*', ms = 20, label = f'Best fit CRTIC: {int(n_fiber_total[best_index_crt])} OTAs of {int(ota_diam_mm[best_index_crt])} mm')
    ax.plot(recalculated_module_diameter_m[best_index_mrc], cost_module[best_index_mrc], '*', ms = 20, label = f'Best fit MEREC: {int(n_fiber_total[best_index_mrc])} OTAs of {int(ota_diam_mm[best_index_mrc])} mm')
    
    plt.title('Cost vs Module size')
    plt.xlabel('Module diameter (m)')
    plt.ylabel(f'Cost (MEUR)')
    plt.grid(which='minor', alpha=0)
    plt.grid(which='major', alpha=0.5)
    
    plt.savefig("Figures/Cost_vs_Module_size.png")
    
    print("\033[1;4;32mFigure 'Cost_vs_Module_size.png' was saved successfully\033[0m")

    plt.close()
    
    for col in criteria.keys():
        val = df[col].values
        if isinstance(val, (list, np.ndarray)):
            print(f"{col}: {val[best_index_st]}")
        else:
            print(f"{col}: {val}")

    print(f"score_total_st: {scores_st[best_index_st]}")
    
    
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={'projection': 'polar'})
    
    series = [st_variance_weight_defu, CRITIC_weight_defu, MEREC_weight_defu]
    series_labels = ["F-St.Variance", "F-CRITIC", "F-MEREC"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    
    N = len(criteria)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    ax.set_ylim(0, 0.5)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4"], fontsize=10)
    ax.grid(True, color="#bcbcbc", alpha=0.6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=11)

    for y, label, color in zip(series, series_labels, colors):
        y_closed = np.concatenate((y, [y[0]]))
        ax.plot(angles, y_closed, color=color, linewidth=2, label=label)
        ax.fill(angles, y_closed, color=color, alpha=0.15)

    #ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15))
    ax.legend(loc=(1, 0.6))
    plt.tight_layout()
    
    plt.savefig("Figures/Spider_Diagram_Weights.png")
    plt.savefig("Figures/Spider_Diagram_Weights.pdf")
    
    print("\033[1;4;32mFigure 'Spider_Diagram_Weights.png' was saved successfully\033[0m")

    plt.close()
    
    df = pd.DataFrame({
    
        "F-St. Variance": st_variance_weight_defu,
        "F-CRITIC":       CRITIC_weight_defu,
        "F-MEREC":        MEREC_weight_defu
    })

    corr = df.corr(method="pearson")

    plt.figure(figsize=(6.5, 5.8))
    ax = sns.heatmap(
        corr,
        vmin=0.5, vmax=1.0,  # rango parecido al de tu figura (ajusta si quieres)
        cmap="Greens",       # paleta en verdes
        annot=True, fmt=".3f",
        square=True,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    
    plt.savefig("Figures/Correlation_Matrix_Weights.png")
    plt.savefig("Figures/Correlation_Matrix_Weights.pdf")
    
    print("\033[1;4;32mFigure 'Correlation_Matrix_Weights.png' was saved successfully\033[0m")

    plt.close()
    
    return st_variance_weight, st_variance_weight_defu, CRITIC_weight, CRITIC_weight_defu, MEREC_weight, MEREC_weight_defu, criteria
    
def snr_cal(archive, plate_scale, R_noise, g, pix_size, FoV, DARK, QE, slicer, super_pl):
    # Detector paremeters (CARMENES-VIS)
    df = pd.read_csv('data/results_marcot.csv', sep = '\t')
    
    #try:
    #    def parse_array(x):
    #        if not isinstance(x, str) or not re.search(r'\d', x):
    #            return x
    #        x = x.strip("[] \n\t")
    #        return np.fromstring(x, sep=' ')
    #        df['Selected commercial output core (microns)'] = df['Selected commercial output core (microns)'].apply(parse_array)
    #except Exception as e:
    #    print(f'\033[4;93;1mWARNING: Error in column {col}: {str(e)}\033[0m')
    #   return None
    
    module_diameter_m = df['Module diameter (m)'].values
    efi_sys_MARCOT = df['Expected efficiency'].values
    

    fiber_core_mm_MARCOT = (df['Selected commercial output core (microns)'].values) * 1e-3

    fiber_core_mm_MARCOT_2 =(df['Selected commercial output core 2-stage (microns)'].values) * 1e-3
    
    e_fiber_core_mm_MARCOT = (df['Uncer Selected commercial output core (microns)'].values) * 1e-3
    
    n_modules = df['Total modules'].values
    
    t_exp = 100 # Exposure time [s]
    pix_size = pix_size * 1e-3
    DARK = DARK * 1e-3
    plate_scale = plate_scale * 1e-3

    def N_det(DARK, t_exp, g, R_noise, fiber_core_mm, n_modules):
        n_pix = n_modules * fiber_core_mm / pix_size
        return n_pix * ((DARK / g) * t_exp + (R_noise / g)**2)

    # Parameters for a random object J02530+168

    l_ini = 1.1e-6 # Beginning of the J band [m]
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

    N_ADU_PSEU = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    
    e_N_ADU_PSEU = N_ADU_PSEU * np.sqrt((0.1 / t_exp) ** 2 + (0.001 / efi_sys_MARCOT) ** 2 + (2 * 0.1 / module_diameter_m) ** 2 + (0.01 * 1e-12/ F_obs_J) ** 2 + ((2 * l_fin * 0.01 * 1e-6) ** 2 + (2 * l_ini * 0.01 * 1e-6) ** 2) ** 2 / ((l_fin ** 2 - l_ini ** 2) ** 2))
    
    
    N_ADU_MARCOT = signal(module_diameter_m, efi_sys_MARCOT, hc, QE, l_ini, l_fin, F_obs_J) * t_exp
    
    e_N_ADU_MARCOT = N_ADU_MARCOT * np.sqrt((0.1 / t_exp) ** 2 + (0.001 / efi_sys_MARCOT) ** 2 + (2 * 0.1 / module_diameter_m) ** 2 + (0.01 * 1e-12/ F_obs_J) ** 2 + ((2 * l_fin * 0.01 * 1e-6) ** 2 + (2 * l_ini * 0.01 * 1e-6) ** 2) ** 2 / ((l_fin ** 2 - l_ini ** 2) ** 2))
    
    if slicer == True:
        SNR_MARCOT = (N_ADU_MARCOT / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT_2 * 0.5, 1) + N_ADU_MARCOT / g)
        SNR_PSEU = (N_ADU_PSEU / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT * 0.5, n_modules) + N_ADU_PSEU / g)
    else:
        SNR_MARCOT = (N_ADU_MARCOT / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT_2, 1) + N_ADU_MARCOT / g)
        SNR_PSEU = (N_ADU_PSEU / g) / np.sqrt(N_det(DARK, t_exp, g, R_noise, fiber_core_mm_MARCOT, n_modules) + N_ADU_PSEU / g)

    
    
    values_MARCOT = {'ADU': N_ADU_MARCOT, 'g': g, 'd': fiber_core_mm_MARCOT_2, 'p': pix_size, 'D': DARK, 't': t_exp, 'R': R_noise}
    sigmas_MARCOT = {'ADU': e_N_ADU_MARCOT, 'g': 0.1, 'd': e_fiber_core_mm_MARCOT, 'p': 0.1e-3, 'D': 0.1, 't': 0.1, 'R': 0.1}
    
    
    values_PSEU = {'ADU': N_ADU_PSEU, 'g': g, 'd': fiber_core_mm_MARCOT, 'p': pix_size, 'D': DARK, 't': t_exp, 'R': R_noise}
    sigmas_PSEU = {'ADU': e_N_ADU_PSEU, 'g': 0.1, 'd': e_fiber_core_mm_MARCOT, 'p': 0.1e-3, 'D': 0.1, 't': 0.1, 'R': 0.1}
        

    def sigma_snr(values, sigmas):

    # Variables
        A = values['ADU']
        g = float(values['g'])
        d = values['d']
        p = float(values['p'])
        D = float(values['D'])
        t = float(values['t'])
        R = float(values['R'])

    # Intermedios
        N = A / g
        Y = (D * t) / g + (R**2) / (g**2)
        X = (d / p) * Y
        Q = X + N

    # Derivadas base
        df_dN = (X + 0.5 * N) / (Q**1.5)
        df_dX = -0.5 * N / (Q**1.5)

    # Derivadas parciales respecto a variables originales
        partials = {}

    # ADU
        partials['ADU'] = df_dN * (1.0 / g)

    # d, p
        partials['d'] = df_dX * (X / d)
        partials['p'] = df_dX * (-X / p)

    # D, t
        partials['D'] = df_dX * ((d / p) * (t / g))
        partials['t'] = df_dX * ((d / p) * (D / g))

    # R
        partials['R'] = df_dX * ((d / p) * (2.0 * R / (g**2)))

    # g (afecta a N y a X)
        dN_dg = -N / g
        dX_dg = (d / p) * (-(D * t) / (g**2) - 2.0 * (R**2) / (g**3))
        partials['g'] = df_dN * dN_dg + df_dX * dX_dg
    
    # Propagación (independencia)
        var_f = 0.0
        for k in ['ADU','g','d','p','D','t','R']:
            var_f += (partials[k] * sigmas[k])**2

        return np.sqrt(var_f)
    
    #target_len = len(df['OTA diameter (m)'])
    
    df['SNR fraction'] = None
    df['Uncer SNR fraction'] = None
    
    
    value_to_save = (SNR_MARCOT / SNR_PSEU)
    uncer_to_save = value_to_save * np.sqrt((sigma_snr(values_MARCOT, sigmas_MARCOT) / SNR_MARCOT) ** 2 + (sigma_snr(values_PSEU, sigmas_PSEU) / SNR_PSEU) ** 2)
    #value_to_save = np.asarray(value_to_save).ravel()
    
    df['SNR fraction'] = pd.Series(value_to_save, index=range(value_to_save.size))
    
    df['Uncer SNR fraction'] = pd.Series(uncer_to_save, index=range(uncer_to_save.size))
    
    df.to_csv('data/results_marcot.csv', sep='\t', index=False)
        
    return value_to_save

    
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
    
#def tables(self):
#    df_decision_matrix = pd.read_csv('data/decision_matrix.csv', sep='\t')
#    df_results = pd.read_csv('data/results_marcot.csv', sep='\t')
#    # df_table = pd.read_csv('data/table_dm.txt', sep='\t')
    
#    shape = np.shape(df_decision_matrix)
  
#    output_file = "data/table_dm.txt"
    
    latex_code = []
    
    latex_code.append("\\begin{table}")
    latex_code.append("\\centering")
    # latex_code.append("\\begin{tabular}{|" + " | ".join(["c"] * shape[1]) + "|}")
    latex_code.append("\\begin{tabular}{" + "|".join(["c"] * (shape[1] - 1)) + "}")
    latex_code.append("\\hline")
    
    # --- Conversión de columnas con arrays ---
    list_columns = [col for col in df_decision_matrix.columns if df_decision_matrix[col].dtype == object and df_decision_matrix[col].astype(str).str.startswith('[').any()]
    
    latex_code.append("\\textbf{" + "} & \\textbf{".join(list_columns) + "} \\\\")
    
    for i in range(0, shape[0]):
        df_decision_matrix.iloc[i]
        def parse_array(x):
            if not isinstance(x, str) or not re.search(r'\d', x):
                return x
            x = x.strip("[] \n\t")
            return np.fromstring(x, sep=' ')
            df_decision_matrix[col] = df_decision_matrix[col].apply(parse_array)
            
    latex_code.append("\\hline")
    
    value_matrix = []
    
    # --- Read the columns ---
    for col in list_columns:
        col_data = df_decision_matrix[col]
        has_array = col_data.apply(lambda x: isinstance(x, (list, np.ndarray))).any()
        if has_array:
            all_values = np.concatenate(col_data.apply(lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else np.array([x])))
            value_matrix.append(all_values[0])
            # latex_code.append(" & ".join(all_values) + " \\\\")
        else:
            value_matrix.append(col_data[0])
            # latex_code.append(" & ".join(col_data) + " \\\\")
            
    value_matrix = np.array([np.fromstring(row.strip('[]'), sep=' ') for row in value_matrix])
    value_matrix = value_matrix.T
    value_matrix = value_matrix.astype(str)
        
    shape = np.shape(value_matrix)
    
    for i in range(0, shape[0]):
        latex_code.append(" & ".join(value_matrix[i]) + " \\\\")
    
    latex_code.append("\\hline")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\caption{Decision matrix}")
    latex_code.append("\\label{tab:decision_matrix}")
    latex_code.append("\\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in latex_code:
            f.write(line + "\n")
    

def tables(criteria):

    st_weight, st_weight_defu, crt_weight, crt_weight_defu, mrc_weight, mrc_weight_defu, criteria = multi_criteria('data/results_marcot.csv', criteria)


    csv_path = Path("data/decision_matrix.csv")  # ajusta si tu ruta es distinta
    txt_out = Path("data/table_dm.txt")

    df = pd.read_csv(csv_path, sep='\t', header=0, index_col=0)
    
    cols = []
    for crit in criteria:
        cols.append(df[crit + "_fuzzy"].values)
        
    column = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    if len(criteria) > 7:
        for i in range(0, len(criteria) - 7):
            column.append(f'C{7 + i}')

    def fmt(x):
        try:
            return f"{float(x):.3g}"  # 3 cifras significativas
        except Exception:
            return str(x)

    index_name = df.index.name or ""
    #cols = df.columns.tolist()

    header_cells = [rf"\textbf{{{c}}}" for c in column]
    header_line = " & ".join(header_cells) + r" \\"

    body_lines = []
    for i, crit in enumerate(criteria):
        vals = []
        for idx, vector in enumerate(cols):
            vals.append(vector[i])
        body_lines.append(f" & ".join(vals) + r" \\")

    table = (
        r"\begin{table*}" + "\n"
        f"  \\begin{{tabular}}{{l{'r' * len(cols)}}}\n"
        r"      \toprule" + "\n"
        "       " + header_line + "\n"
        r"      \midrule" + "\n"
        "       " + "\n".join(body_lines) + "\n"
        r"      \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"  \caption{Decision matrix of configurations}" + "\n"
        r"  \label{tab:decision_matrix_inline}" + "\n"
        r"\end{table*}" + "\n"
    )

    txt_out.write_text(table, encoding="utf-8")

    print("\033[1;4;32mFile 'table_dm.txt' was saved successfully\033[0m")
    
    
    
    column = []
    for crit in criteria:
        column.append(crit)
    column = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    if len(criteria) > 7:
        for i in range(0, len(criteria) - 7):
            column.append(f'C{7 + 1 + i}')
        
    txt_out = Path("data/table_weight.txt")

    index_name = ['St. Variance', 'CRITIC', 'MEREC']
    weights = []
    weights.append(np.round(st_weight, 3))
    weights.append(np.round(crt_weight, 3))
    weights.append(np.round(mrc_weight, 3))
    weights = np.array(weights)
    
    header_cells = [rf"\textbf{{Criterion}}"] + [rf"\textbf{{{c}}}" for c in index_name]
    header_line = " & ".join(header_cells) + r" \\"
    
    body_lines = []
    for i, c in enumerate(column):
        vals = []
        for idx, vector in enumerate(weights):
            vals.append(vector[i])
        body_lines.append(f"{c} & " + " & ".join(str(cc) for cc in vals) + r" \\")

    table = (
        r"\begin{table*}" + "\n"
        r"  \centering" + "\n"
        f"  \\begin{{tabular}}{{l{'c' * (len(index_name) + 1)}}}\n"
        r"      \toprule" + "\n"
        r"      \toprule" + "\n"
        "       " + header_line + "\n"
        r"      \midrule" + "\n"
        "       " + "\n".join(body_lines) + "\n"
        r"      \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"  \caption{Fuzzy weights determined by methodology.}" + "\n"
        r"  \label{tab:all_weights}" + "\n"
        r"\end{table*}" + "\n"
    )

    txt_out.write_text(table, encoding="utf-8")

    print("\033[1;4;32mFile 'table_weight.txt' was saved successfully\033[0m")
    
    column = []
    for crit in criteria:
        column.append(crit)
    column = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    if len(criteria) > 7:
        for i in range(0, len(criteria) - 7):
            column.append(f'C{7 + 1 + i}')
        
    txt_out = Path("data/table_weight_defu.txt")

    index_name = ['St. Variance ($\%$)', 'CRITIC ($\%$)', 'MEREC ($\%$)']
    weights = []
    weights.append(np.round(st_weight_defu * 100, 3))
    weights.append(np.round(crt_weight_defu * 100, 3))
    weights.append(np.round(mrc_weight_defu * 100, 3))
    weights = np.array(weights)
    
    header_cells = [rf"\textbf{{Criterion}}"] + [rf"\textbf{{{c}}}" for c in index_name]
    header_line = " & ".join(header_cells) + r" \\"
    
    body_lines = []
    for i, c in enumerate(column):
        vals = []
        for idx, vector in enumerate(weights):
            vals.append(vector[i])
        body_lines.append(f"{c} & " + " & ".join(str(cc) for cc in vals) + r" \\")

    table = (
        r"\begin{table*}" + "\n"
        r"  \centering" + "\n"
        f"  \\begin{{tabular}}{{l{'c' * (len(index_name) + 1)}}}\n"
        r"      \toprule" + "\n"
        r"      \toprule" + "\n"
        "       " + header_line + "\n"
        r"      \midrule" + "\n"
        "       " + "\n".join(body_lines) + "\n"
        r"      \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"  \caption{Defuzzy weights determined by methodology.}" + "\n"
        r"  \label{tab:all_weights_defu}" + "\n"
        r"\end{table*}" + "\n"
    )

    txt_out.write_text(table, encoding="utf-8")

    print("\033[1;4;32mFile 'table_weight_defu.txt' was saved successfully\033[0m")
    
    df_st_TOPSIS = pd.read_csv("data/score_total_st_TOPSIS.csv", sep='\t', header=0)
    
    df_crt_TOPSIS = pd.read_csv("data/score_total_crt_TOPSIS.csv", sep='\t', header=0)
    
    df_mrc_TOPSIS = pd.read_csv("data/score_total_mrc_TOPSIS.csv", sep='\t', header=0)
    
    df_st_MABAC = pd.read_csv("data/score_total_st_MABAC.csv", sep='\t', header=0)
    
    df_crt_MABAC = pd.read_csv("data/score_total_crt_MABAC.csv", sep='\t', header=0)
    
    df_mrc_MABAC = pd.read_csv("data/score_total_mrc_MABAC.csv", sep='\t', header=0)
    
    txt_out = Path("data/table_ranking.txt")
    
    columns_TOPSIS = []
    columns_TOPSIS.append(df_st_TOPSIS['Alternative'].values)
    columns_TOPSIS.append(df_crt_TOPSIS['Alternative'].values)
    columns_TOPSIS.append(df_mrc_TOPSIS['Alternative'].values)
    
    columns_MABAC = []
    columns_MABAC.append(df_st_MABAC['Alternative'].values)
    columns_MABAC.append(df_crt_MABAC['Alternative'].values)
    columns_MABAC.append(df_mrc_MABAC['Alternative'].values)
    
    index_name = ['St. Variance', 'CRITIC', 'MEREC']
    
    header_cells = [rf"\textbf{{{c}}}" for c in index_name]
        
    header_line_1 = r"\multicolumn{4}{c|}{\textbf{TOPSIS}}" + " & " + r"\multicolumn{4}{c}{\textbf{MABAC}}" + r"\\"
    
    header_line_2 = " & ".join(header_cells) + " & " + " & ".join(header_cells) + r" \\"
    
    body_lines = []
    for i in range(0, len(columns_TOPSIS[0])):
        body_lines.append(f" & ".join(str(c[i]) for c in columns_TOPSIS) + " & " + f" & ".join(str(c[i]) for c in columns_MABAC) + r" \\")
        
        
    table = (
        r"\begin{table*}" + "\n"
        f"\\begin{{tabular}}{{{'c' * len(index_name)}|{'c' * len(index_name) }}}\n"
        r"\toprule" + "\n"
        r"\toprule" + "\n"
        + header_line_1 + "\n"
        r"\midrule" + "\n"
        + header_line_2 + "\n"
        r"\midrule" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\caption{Ranking of the alternatives depending on weight-determination method and ranking method.}" + "\n"
        r"\label{tab:all_rankings}" + "\n"
        r"\end{table*}" + "\n"
    )

    txt_out.write_text(table, encoding="utf-8")

    print("\033[1;4;32mFile 'table_ranking.txt' was saved successfully\033[0m")
    
    

