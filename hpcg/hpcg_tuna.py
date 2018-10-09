import os
import pickle

import jinja2

from subprocess import Popen, PIPE, check_output
from hyperopt import hp, tpe, fmin
from hyperopt.base import Trials

import numpy as np

TEMPLATE_FILE = "hpcg_jinja_template.dat"
HPCG_CONFIG_FILE = "hpcg.dat"

def main():
    search_space = {}
    search_space["problem_size"] = hp.quniform("problem_size", 1, 64, 1)
    search_space["num_ranks"] = hp.quniform("num_ranks", 1, 8, 1)

    trials = Trials()

    best = fmin(fn=run_hpcg,
                space=search_space,
                algo=tpe.suggest, 
                max_evals = 50,
                trials=trials)

    print("------------------------------------------------")

    for trial in trials:
        print("{1}x({2}, {3}, {4}) = {0} GF".format(-1.0*trial["result"]["loss"],
                                                    trial["misc"]["vals"]["num_ranks"][0],
                                                    trial["misc"]["vals"]["problem_size"][0],
                                                    trial["misc"]["vals"]["problem_size"][0],
                                                    trial["misc"]["vals"]["problem_size"][0]))

    print("Saving pkl....")
    with open('trials.pkl', 'wb') as output:
        pickle.dump(trials, output)
    print("... done")

def run_hpcg(search_space):
    problem_size = int(search_space["problem_size"]) * 8
    num_ranks = str(int(search_space["num_ranks"]))

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(TEMPLATE_FILE)
    output_text = template.render(dimx=problem_size,
                                  dimy=problem_size,
                                  dimz=problem_size)

    with open(HPCG_CONFIG_FILE, "w") as file_handle:
        file_handle.write(output_text)

    try:
        hpcg_output = check_output(["mpirun",
                                    "-n", num_ranks,
                                    "./xhpcg-3.1_gcc_485_cuda90176_ompi_1_10_2_sm_35_sm_50_sm_60_sm_70_ver_10_8_17"""
                                   ],
                                   universal_newlines=True)

        perf = float(hpcg_output.split("\n")[-5].split()[2])
    except Exception as error:
        print(error)
        perf = -100.

    print("HPCG run, {4}x({0}, {1}, {2}): {3} GF".format(problem_size, problem_size, problem_size, perf, num_ranks))

    return -1.0 * perf

if __name__ == '__main__':
    main()
