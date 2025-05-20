import argparse
from src.estimator import marcot_hr_estimator
from src.utils import print_results

parser = argparse.ArgumentParser()
parser.add_argument("params", nargs=14)
args = parser.parse_args()


typed = [
    float(args.params[0]), float(args.params[1]), float(args.params[2]), float(args.params[3]),
    args.params[4] == "True", float(args.params[5]), float(args.params[6]), float(args.params[7]),
    int(args.params[8]), int(args.params[9]), int(args.params[10]), float(args.params[11]),
    args.params[12] == "True", args.params[13] == "True"
]

results = marcot_hr_estimator(*typed)
print_results(results)

