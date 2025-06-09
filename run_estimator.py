import argparse
import pandas as pd
from src.estimator import marcot_hr_estimator
from src.utils import print_results
from src.utils import multi_criteria
from src.utils import snr_cal
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("params", nargs=10)
args = parser.parse_args()


typed = [
    float(args.params[0]), float(args.params[1]), float(args.params[2]), args.params[3] == "False",
    float(args.params[4]), int(args.params[5]), int(args.params[6]), float(args.params[7]),
    args.params[8] == "True", args.params[9] == "False"
    ]

results, module_diameter_m, com_core_out_mm = marcot_hr_estimator(*typed)
print_results(results, module_diameter_m, com_core_out_mm)

filter_data = {k: v for k, v in results.items() if isinstance(v, (int, float, np.ndarray)) and k != ""}

df = pd.DataFrame.from_dict(filter_data, orient = 'index', columns=["# Value"])
df.to_csv("data/results_marcot.txt", sep = '\t', index = False)
print("\033[1;4;32mFile 'results_marcot.txt' was saved successfully\033[0m")

MCM = multi_criteria('data/results_marcot.txt')
print(MCM)



