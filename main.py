# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:52:40 2020

@author: Cory Kromer-Edwards

To run full code for complete output:
python main.py --threaded -ai --output-csv
"""
from argparse import ArgumentParser

import random
import numpy as np

# For parallel computing
import multiprocessing
import signal
import time

# Imports for the algorithms
import GA as GA
import GSA as GSA
import RAN as RAN
import PSO as PSO

# For options dictionary
import itertools
import time
# For graphing
import matplotlib.pyplot as plt

# For outputing csv
import csv

from progress_bar import print_progress_bar
from grid_search import grid_search

# for optimization
from convex_quadratic_opt import generate_input as gi, generate_solution_nonconvex
from convex_quadratic_opt import nonconvex_generate_input as gnci
from convex_quadratic_opt import f_vect


def build_parser():
    """
  Builds the parser based on input variables from the command line.
  """
    parser = ArgumentParser()

    # Arguments for the Genetic Algorithm
    # =====================================================================================
    parser.add_argument('-a', '--alpha', dest='alpha', type=float, default=0.9,
                        help='Alpha used for blending individuals when mating',
                        metavar='A')

    parser.add_argument('-t', '--tournament-size', dest='tournsize', type=int, default=4,
                        help='The number of individuals per tournament during selection',
                        metavar='T')

    parser.add_argument('-i', '--indpb', dest='indpb', type=float, default=0.9,
                        help='Probability that a gene will be mutated in a mutating individual',
                        metavar='I')

    parser.add_argument('-g', '--num-gen', dest='number_generations', type=int, default=1000,
                        help='Number of generations to run through in algorithm',
                        metavar='G')

    parser.add_argument('-c', '--cxpb', dest='cxpb', type=float, default=0.5,
                        help='Percentage chance that 2 individuals will be mated',
                        metavar='C')

    parser.add_argument('-m', '--mutpb', dest='mutpb', type=float, default=0.5,
                        help='Percentage chance that an individual will be mutated',
                        metavar='M')

    parser.add_argument('-mu', '--gausian-mean', dest='mu', type=float, default=0,
                        help='The mean to use in the gausian distrobution when mutating genes',
                        metavar='MU')

    parser.add_argument('-si', '--gausian-sigma', dest='sigma', type=float, default=0.005,
                        help='The standard deviation to use in the gausian distrobution when mutating genes',
                        metavar='SI')
    # =====================================================================================

    # Arguments for the Gravity Search Algorithm
    # =====================================================================================
    parser.add_argument('-gc', '--grav-constant', dest='gc', type=float, default=.7,
                        help='The gravitiational Constant',
                        metavar='GI')
    parser.add_argument('-gd', '--grav-decay', dest='gd', type=float, default=.9,
                        help='The gravitiational decay',
                        metavar='GD')
    # =====================================================================================

    # Arguments for the Random Search Algorithm
    # =====================================================================================
    parser.add_argument('-rsn', '--random-search-number', dest='rsn', type=int, default=5,
                        help='The number of random points to sample ')
    # =====================================================================================

    # Arguments for the PSO
    # =====================================================================================
    parser.add_argument('-vw', '--vel_weight', dest='vw', type=float, default=.9,
                        help='The previous velocity weight',
                        metavar='VW')
    parser.add_argument('-sw', '--social_weight', dest='sw', type=float, default=.7,
                        help='The social particle weighting',
                        metavar='SW')
    parser.add_argument('-cw', '--cognitive_weight', dest='cw', type=float, default=.5,
                        help='The cognitive particle weighting',
                        metavar='CW')

    # General arguments for problems
    # =====================================================================================
    parser.add_argument('-n', '--size', dest='size', type=int, default=5,
                        help='The size of the square matrices',
                        metavar='N')
    # =====================================================================================

    # Arguments for Quadratic Optimization problem
    # =====================================================================================
    parser.add_argument('-k', '--condition-number', dest='k', type=int, default=5,
                        help='The condition number that we want to approximate for A matrix',
                        metavar='K')
    # =====================================================================================

    # Arguments for Nonconvex Optimization problem
    # =====================================================================================
    parser.add_argument('-ncm', '--ncm', dest='ncm', type=int, default=5,
                        help='Number of local minimizers (3, 10)',
                        metavar='ncm')
    parser.add_argument('-ncb', '--ncb', dest='ncb', type=int, default=5,
                        help='Upper limit for random numbers generates for beta',
                        metavar='ncb')
    parser.add_argument('-ncM', '--ncM', dest='ncM', type=int, default=5,
                        help='Upper limit for random numbers generates for alpha',
                        metavar='ncM')
    # =====================================================================================

    # Output arguments
    # =====================================================================================
    parser.add_argument('-v', '--verbose', dest='debug', type=int, default=-1,
                        help='The log level for the algorithm. Values are [-1, 0, 1, 2]. -1 = no output',
                        metavar='S')
    parser.add_argument('-cl', '--contor-level', dest='cl', type=int, default=10,
                        help='the number of contor levels for gif interpolation',
                        metavar='CL')
    parser.add_argument('--output-csv', dest='is_csv_exported', action='store_true',
                        help='If given, csv file(s) will be generated  from runs.')

    parser.add_argument('--output-plot', dest='is_plot_exported', action='store_true',
                        help='If given, plot file(s) will be generated  from runs.')
    # =====================================================================================

    # Misc arguments
    # =====================================================================================
    parser.add_argument('-s', '--seed', dest='seed', type=int, default=1234,
                        help='The random seed for the algorithm',
                        metavar='S')

    parser.add_argument('-p', '--num-particles', dest='num_particles', type=int, default=50,
                        help='Number of particles for algorithms to use',
                        metavar='P')

    parser.add_argument('-r', '--problem-runs', dest='problems', type=int, default=2,
                        help='Which problems should be run. 0=just quad, 1=just non-convex, 2=both',
                        metavar='R')

    parser.add_argument('-ai', '--all-inputs', dest='use_pred_inputs', action='store_true',
                        help='If given, all problem inputs will be ignored and each chosen problem will go over all preset inputs')

    parser.add_argument('-gs', '--grid-search', dest='perform_grid_search', action='store_true',
                        help='If given, grid search will be performed and then the program will exit.')

    parser.add_argument('--threaded', dest='is_threaded', action='store_true',
                        help='If given, a process pool will be generated, and multiple algorithm runs will be done in parallel.')
    # =====================================================================================

    return parser


def run_alg(problem, options, alg_class):
    """
  Runs a given algorithm based on options.

  Parameters
  ----------
  problem : Int
    The problem to be performed [0, 1].
  options : NameSpace
    All user given or default values for setup.
  alg_class : Algorithm
    The Algorithm class that is within the algorithm modules.

  Returns
  -------
  logbook : Dictionary
    A dictionary of arrays for iterations, min, max, average, and std. dev. for each iteration.

  """
    if problem == 1:
        alg = alg_class(problem=1, **options)
        _, _, _, loss_values, value = alg.run()
    elif problem == 0:
        alg = alg_class(problem=0, **options)
        _, _, _, loss_values, value = alg.run()

    return loss_values, value


# Data output methods
# ==============================================================================================================
def plot_multi_data(num_particles, data_dict, alg_import):
    for key, values in data_dict.items():
        plt.plot(num_particles, values, 'b-', label=key)

    legend = plt.legend(ncol=3, title='Key:', bbox_to_anchor=(1.04, 1))
    plt.xlabel('Particle Number')
    plt.ylabel("Error")
    plt.savefig(alg_import.to_string() + '.svg', bbox_extra_artists=(legend,), bbox_inches='tight')


def save_csv_multi(num_particles, output_dict, alg_import, seed, problem, key_header):
    with open(f'csvs/{alg_import.to_string()}_seed_{seed}_prob_{problem + 1}_all.csv',
              'w+', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(key_header + [f"loss p{i + 1}" for i in range(num_particles)])
        for key, loss_values in output_dict.items():
            writer.writerow(key.split(',') + loss_values)


def save_csv_single(loss_values, options, alg_import, key, problem):
    if alg_import.to_string() == "GSA":
        filename = "GSA_prob_{}_pop_{}_{}_gc_{}_gd_{}_iter_{}.csv".format(problem,
                                                                          options.num_particles, key,
                                                                          options.gc, options.gd,
                                                                          options.number_generations)
    if alg_import.to_string() == "PSO":
        filename = "PSO_prob_{}_pop_{}_{}_sw_{}_cw_{}_vw_{}_iter_{}.csv".format(problem,
                                                                                options.number_generations)

    if alg_import.to_string() == "GA":
        filename = "GA_prob_{}_pop_{}_{}.csv".format(problem, options.num_particles, key)

    if alg_import.to_string() == "RAN":
        filename = "RAN_prob_{}_pop_{}_{}.csv".format(problem, options.num_particles, key)

    with open(r'csvs/' + filename, 'w+', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(loss_values)


# ==============================================================================================================

def setup_alg(options, alg_import):
    """
  Perform setup for a given algorithm.

  Parameters
  ----------
  options : NameSpace
    All user given or default values for setup.
  alg_import : Algorithm import
    The import module of an algorithm file that will be used.

  Returns
  -------
  None.

  """
    if options.perform_grid_search:
        # Perform Grid search to find best hyperparameters to set as default
        grid_search(options.is_threaded, options.k, options.size, alg_import)
    else:
        if options.problems == 0:
            problem_runs = [0]
        elif options.problems == 1:
            problem_runs = [1]
        elif options.problems == 2:
            problem_runs = [0, 1]
        else:
            print("Problems given is not of [0, 1, 2]")
            exit(1)

        for problem in problem_runs:
            print(f"\tRunning problem: {problem + 1}")
            print(f"\tThreading is: {'Enabled' if options.is_threaded else 'Disabled'}")

            if options.use_pred_inputs:
                run_num = 1
                # for steps in [100, 1000, 10000, 100000]:
                for steps in [10]:
                    options.number_generations = steps
                    log_dict = dict()
                    dis_dict = dict()
                    if options.is_threaded:
                        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                        executor = multiprocessing.Pool(8)
                        signal.signal(signal.SIGINT, original_sigint_handler)
                        submitted = 1
                        futures = dict()

                    try:
                        # for k in [3, 10, 30, 100, 300, 1000]:
                        #  for n in [2, 5, 10, 20, 50, 100]:
                        k_values = [5, 10, 50]
                        n_values = [5, 10, 50]
                        b_values = [5, 10, 50]
                        m_values = [5, 10, 50]
                        M_values = [5, 10, 50]

                        if problem == 0:
                            total = len(k_values) * len(n_values)
                        else:
                            total = len(b_values) * len(m_values) * len(M_values) * len(n_values)

                        for n in n_values:
                            if problem == 0:
                                for k in k_values:
                                    key = f"{n},{k}"
                                    key_header = ["n", "k"]

                                    if options.is_threaded:
                                        futures[executor.apply_async(run_alg, (
                                            problem, vars(options), alg_import.Algorithm))] = key
                                        print(f'\rSubmitted {submitted} / {total} jobs', end='\r')
                                        submitted += 1
                                    else:
                                        loss_values, distance_from_sol = run_alg(problem, vars(options),
                                                                                 alg_import.Algorithm)
                                        dis_dict[key] = distance_from_sol
                                        log_dict[key] = loss_values
                                        print_progress_bar(run_num, total)
                                        run_num += 1
                            else:
                                for m in m_values:
                                    for M in M_values:
                                        for b in b_values:
                                            options.ncm = m
                                            options.ncM = M
                                            options.ncb = b
                                            options.size = n
                                            key = f"{n},{m},{M},{b}"
                                            key_header = ["n", "m", "M", "b"]

                                            if options.is_threaded:
                                                futures[executor.apply_async(run_alg, (
                                                    problem, vars(options), alg_import.Algorithm))] = key
                                                print(f'\rSubmitted {submitted} / {total} jobs', end='\r')
                                                submitted += 1
                                            else:
                                                loss_values, distance_from_sol = run_alg(problem, vars(options),
                                                                                         alg_import.Algorithm)
                                                dis_dict[key] = distance_from_sol
                                                log_dict[key] = loss_values
                                                print_progress_bar(run_num, total)
                                                run_num += 1

                        if options.is_threaded:
                            for future in futures:
                                key = futures[future]
                                try:
                                    loss_values, distance_from_sol = future.get()
                                except Exception as exc:
                                    print('%r generated an exception: %s' % (key, exc))
                                    raise exc
                                else:
                                    dis_dict[key] = distance_from_sol
                                    log_dict[key] = loss_values
                                    print_progress_bar(run_num, total)
                                    run_num += 1
                    except KeyboardInterrupt:
                        if options.is_threaded:
                            executor.terminate()
                    else:
                        if options.is_threaded:
                            executor.close()

                    if options.is_threaded:
                        executor.join()

                    if options.is_plot_exported:
                        plot_multi_data(options.num_particles, log_dict, alg_import)

                    if options.is_csv_exported:
                        save_csv_multi(options.num_particles, log_dict, alg_import, str(options.seed), problem,
                                       key_header)
            else:
                loss_values, distance_from_sol = run_alg(problem, vars(options), alg_import.Algorithm)
                if options.is_csv_exported:
                    if problem == 0:
                        key = key = f"k={options.k}, n={options.size}"
                        dis_dict[key] = distance_from_sol
                        log_dict[key] = loss_values
                    else:
                        key = f"m={options.ncm}, M={options.ncM}, b={options.ncb}, n={options.size}"
                        dis_dict[key] = distance_from_sol
                        log_dict[key] = loss_values

                    save_csv_single(loss_values, options, alg_import, key, problem)
            w = csv.writer(open("type_{}_prob_{}_output.csv".format(alg_import.to_string(), problem), "w"))
            for key, val in dis_dict.items():
                w.writerow([key, sorted(val)])


def generate_dic(options):
    k_values = [5, 10, 50]
    n_values = [5, 10, 50]
    b_values = [5, 10, 50]
    m_values = [5, 10, 50]
    M_values = [5, 10, 50]

    if options.problems == 0:
        problem_runs = [0]
    elif options.problems == 1:
        problem_runs = [1]
    elif options.problems == 2:
        problem_runs = [0, 1]
    else:
        print("Problems given is not of [0, 1, 2]")
        exit(1)

    options_dict_values = {}
    solution_file = open("solution_file_seed_{}.txt".format(options.seed), "w+")
    for k_in in k_values:
        for n_in in n_values:
            for b in b_values:
                for m in m_values:
                    for M in M_values:
                        key_problem1 = "0_k{}_n{}_b{}_m{}_M{}".format(k_in, n_in, b, m, M)
                        key_problem2 = "1_k{}_n{}_b{}_m{}_M{}".format(k_in, n_in, b, m, M)
                        if key_problem2 not in options_dict_values.keys() or key_problem2 not in \
                                options_dict_values.keys():
                            # problem 1 parameters
                            alpha, beta, solution = gi(k_in, n_in, options.debug)
                            solution_loc = [i[0] for i in solution]
                            options_dict_values[key_problem1] = [alpha, beta, solution_loc]
                            solution_file.write(key_problem1 + " | " + str(options_dict_values[key_problem1][-1]))
                            # problem 2 parameters
                            q_mat, alpha2, beta2, gamma = gnci(n_in, m, M, b)
                            options_dict_values[key_problem2] = [q_mat, alpha2, beta2, gamma, generate_solution_nonconvex(q_mat, alpha2, beta2, gamma)]
                            solution_file.write(key_problem2 + " | " + str(options_dict_values[key_problem2][-1] )+"\n")
    solution_file.close()

    return options_dict_values


def main():
    # build the parser for implementation
    # Default parser values:
    parser = build_parser()

    options = parser.parse_args()
    random.seed(options.seed)
    np.random.seed(options.seed)
    print("generating dictionary")
    options.dic = generate_dic(options)
    print("generating done")

    # Add imported algorithm modules to this list to have them be used.
    for alg in [GSA, PSO, RAN, GA]:
        print("running: {}".format(alg.to_string()))
        start_time = time.time()
        setup_alg(options, alg)
        end_time = time.time()
        print(str(alg) + "finished in %s secoonds\n\n" %(end_time - start_time))

if __name__ == '__main__':
    main()
