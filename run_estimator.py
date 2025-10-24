import argparse
import pandas as pd
from src.estimator import marcot_hr_estimator
from src.utils import print_results
from src.utils import multi_criteria
from src.utils import snr_cal
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("params", nargs=23)
args = parser.parse_args()


typed = [
    float(args.params[0]), float(args.params[1]), args.params[2], float(args.params[3]), args.params[4], float(args.params[5]), float(args.params[6]), float(args.params[7]), args.params[8], args.params[9], float(args.params[10]), int(args.params[11]), int(args.params[12]), args.params[13], args.params[14], float(args.params[15]), float(args.params[16]), float(args.params[17]), float(args.params[18]), float(args.params[19]), float(args.params[20]), float(args.params[21]), args.params[22]
    ]


results_list = []

results = marcot_hr_estimator(*typed)
print_results(results)

filter_data = {k: v for k, v in results.items() if isinstance(v, (int, float, np.ndarray)) and k != ""}

results_list.append(filter_data)

df_results = pd.DataFrame(results_list)

# df = pd.DataFrame.from_dict(filter_data, orient = 'index', columns=["# Value"])

df_results.to_csv("data/results_marcot.csv", sep = '\t', index = False)
print("\033[1;4;32mFile 'results_marcot.csv' was saved successfully\033[0m")

def print_section(title):
    print(f"\n{'=' * 60}\n{title:^60}\n{'=' * 60}")

print_section("SCIENTIFIC CRITERIA")
# fraction = snr_cal('data/results_marcot.csv')
    # HAY QUE DEFINIR UN VALOR MÁS REALISTA DE LA EFICIENCIA DEL SISTEMA DE MARCOT Y NO PONER UN 0.9 POR DEFECTO
    
    # HAY QUE CAMBIAR EL module_diameter_m, POR PARÁMETROS n_otas Y diameter_ota_m PARA QUE CALCULA LA APERTURA EFECTIVA

fraction = snr_cal('data/results_marcot.csv')

# print(f"MARCOT telescope has a {np.round(fraction, 3)} * SNR in comparison to a traditional design")
    
# print(f"Thanks to the use of PL, the SNR is improved by a factor of 3 {np.round(fraction_pl, 3)}")

# print(f"Thanks to the use of super PL, the SNR is improved by a factor of 3 {np.round(fraction_super_pl, 3)}")

MCM = multi_criteria('data/results_marcot.csv')
print(MCM)





