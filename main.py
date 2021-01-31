# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:52:40 2020

@author: Cory Kromer-Edwards

python main.py --threaded -ai --output-csv -r 0
"""
import shutil
from argparse import ArgumentParser

import random
import numpy as np

# For parallel computing
import multiprocessing
import signal

# Imports for the algorithms
import GA as GA
import GSA as GSA
import RAN as RAN
import PSO as PSO
# For graphing
import matplotlib.pyplot as plt

# For outputing csv
import csv


def print_progress_bar(iteration, total, decimals=3, fill='#', prefix='Progress:', suffix='Complete',
                       print_end='\r'):
    """
    Variables and progress bar code from:
    https://stackoverflow.com/a/34325723

    Updated to write line max to be width of console - 8% (EX: width = 235 then printline will be 206
    """
    terminal_width = shutil.get_terminal_size().columns
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    length = shutil.get_terminal_size().columns - (len(prefix) + len(suffix) + len(percent) + (terminal_width // 8))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def _thread_caller(params, alg_class):
    alg = alg_class(problem=1, debug=-1, num_particles=50, **params)
    _, fitness, _ = alg.run()
    return fitness


def grid_search(is_threaded, k, size, alg_import):
    """
  Performs grid search over all hyperparameters for a particular problem. The
  best values for each hyperparam will be printet when finished to be set as default.

  Parameters
  ----------
  is_threaded : Boolean
    Should the grid search be distributed.
  k : Integer
    Condition number to use.
  size : Int
    Size of square A matrix.
  alg_import : Module
    Imported algorithm module to be used.

  Raises
  ------
  exc
    Exception that may occur in a sub process.

  Returns
  -------
  None.

  """

    options = alg_import.get_params_gs()

    # The product object that is returned is a lazy iterator, not a list.
    total = len(list(alg_import.get_params_gs()))
    iteration = 0

    best_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_fitness = 90000000

    if is_threaded:
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        executor = multiprocessing.Pool(8)
        signal.signal(signal.SIGINT, original_sigint_handler)
        submitted = 1
        futures = dict()

    try:
        if is_threaded:
            for x in options:
                # for GA
                # params = {'size': size, 'k':k, 'mu':x[0], 'sigma':x[1], 'alpha':x[2], 'indpb':x[3], 'tournsize':x[4], 'cxpb':x[5], 'mutpb':x[6]}
                # for GSA
                params = {'size': size, 'k': k, 'gc': x[0], 'gd': x[1], 'number_generations': 10, 'problems': 0,
                          'cl': 10}
                futures[executor.apply_async(_thread_caller, (params, alg_import.Algorithm))] = x
                print(f'\rSubmitted {submitted} / {total} jobs', end='\r')
                submitted += 1
        else:
            for params in options:
                alg = alg_import.Algorithm(problem=1, mu=params[0], sigma=params[1], alpha=params[2], indpb=params[3],
                                           tournsize=params[4], cxpb=params[5], mutpb=params[6], debug=-1, size=size,
                                           k=k, num_particles=50)
                _, fitness, _ = alg.run()

                if fitness <= best_fitness:
                    best_fitness = fitness
                    best_values = params

                print_progress_bar(iteration, total, suffix=("Complete--Best fitness: {0:.3f}").format(best_fitness))
                iteration += 1

        if is_threaded:
            for future in futures:
                parameters = futures[future]
                fitness = future.get()
                if fitness <= best_fitness:
                    best_fitness = fitness
                    best_values = parameters

                print_progress_bar(iteration, total,
                                   suffix=("Complete--Best fitness: {0:.3f}").format(best_fitness))
                iteration += 1
    except KeyboardInterrupt:
        if is_threaded:
            executor.terminate()
    else:
        if is_threaded:
            executor.close()

    if is_threaded:
        executor.join()

    # print("Best values from grid search evaluation is:\n\tMu:%.3f\n\tSigma:%.3f\n\tAlpha:%.3f\n\tIndpb:%.3f\n\tTournsize:%i\n\tCxpb:%.3f\n\tMutpb:%.3f"% best_values)
    print("Best parameters had fitness: %.3f" % (best_fitness))


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
    parser.add_argument('-rsn', '--random-search-number', dest='rsn', type=int, default=10,
                        help='The number of random points to sample ')

    # =====================================================================================
    # Arguments for the PSO
    # =====================================================================================
    parser.add_argument('-vw', '--vel_weight', dest='vw', type=float, default=.5,
                        help='The previous velocity weight',
                        metavar='VW')
    parser.add_argument('-sw', '--social_weight', dest='sw', type=float, default=.5,
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
    parser.add_argument('-k', '--condition-number', dest='k', type=int, default=3,
                        help='The condition number that we want to approximate for A matrix',
                        metavar='K')
    # =====================================================================================
    # Arguments for Nonconvex Optimization problem
    # =====================================================================================
    parser.add_argument('-ncm', '--ncm', dest='ncm', type=int, default=3,
                        help='Number of local minimizers (3, 10)',
                        metavar='ncm')
    parser.add_argument('-ncb', '--ncb', dest='ncb', type=int, default=1,
                        help='Upper limit for random numbers generates for beta',
                        metavar='ncb')
    parser.add_argument('-ncM', '--ncM', dest='ncM', type=int, default=1,
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
        _, _, _, loss_values = alg.run()
    elif problem == 0:
        alg = alg_class(problem=0, **options)
        _, _, _, loss_values = alg.run()

    return loss_values


# Data output methods
# ==============================================================================================================
def plot_multi_data(num_particles, data_dict, alg_import):
    for key, values in data_dict.items():
        plt.plot(num_particles, values, 'b-', label=key)

    legend = plt.legend(ncol=3, title='Key:', bbox_to_anchor=(1.04, 1))
    plt.xlabel('Particle Number')
    plt.ylabel("Error")
    plt.savefig(alg_import.to_string() + '.svg', bbox_extra_artists=(legend,), bbox_inches='tight')


# def plot_single_data(iterations, data, key, alg_import):
#  plt.plot(iterations, data, 'b-', label=key)
#  legend = plt.legend(title='Key:', bbox_to_anchor=(1.04, 1))
#  plt.xlabel('Iteration')
#  plt.ylabel("Error")
#  plt.savefig(alg_import.to_string() + '__single_' + str(len(iterations)) + '.svg', bbox_extra_artists=(legend,), bbox_inches='tight')

def save_csv_multi(num_particles, output_dict, alg_import, seed):
    with open(r'csvs/' + alg_import.to_string() + "_seed_" + seed + "_"
              + "_all.csv", 'w+', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Key"] + [f"loss p{i + 1}" for i in range(num_particles)])
        for key, loss_values in output_dict.items():
            writer.writerow([key] + loss_values)


def save_csv_single(loss_values, options, alg_import):
    if alg_import.to_string() == "GSA":
        filename = "GSA_prob_{}_pop_{}_k_{}_n_{}_gc_{}_gd_{}_iter_{}.csv".format(options.problems,
                                                                                 options.num_particles, options.k,
                                                                                 options.size, options.gc, options.gd,
                                                                                 options.number_generations)
    if alg_import.to_string() == "PSO":
        filename = "PSO_prob_{}_pop_{}_k_{}_n_{}_sw_{}_cw_{}_vw_{}_iter_{}.csv".format(options.problems,
                                                                                       options.num_particles, options.k,
                                                                                       options.size, options.sw,
                                                                                       options.cw, options.vw,
                                                                                       options.number_generations)

    if alg_import.to_string() == "GA":
        filename = "GA_prob_{}_pop_{}_k_{}_n_{}.csv".format(options.problems, options.num_particles, options.k,
                                                            options.size)
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
            print(f"Running problem: {problem}")
            print(f"Threading is: {'Enabled' if options.is_threaded else 'Disabled'}")
            if options.use_pred_inputs:
                run_num = 1
                # for steps in [100, 1000, 10000, 100000]:
                for steps in [10000]:
                    options.number_generations = steps
                    log_dict = dict()
                    if options.is_threaded:
                        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
                        executor = multiprocessing.Pool(8)
                        signal.signal(signal.SIGINT, original_sigint_handler)
                        submitted = 1
                        futures = dict()

                    try:
                        # for k in [3, 10, 30, 100, 300, 1000]:
                        #  for n in [2, 5, 10, 20, 50, 100]:
                        k_values = [5, 50, 100, 500, 1000]
                        n_values = [5, 10, 50, 100]
                        for k in k_values:
                            for n in n_values:
                                options.k = k
                                options.size = n

                                if options.is_threaded:
                                    futures[executor.apply_async(run_alg, (problem, vars(options), alg_import.Algorithm))] = "k=" + str(k) + ", n=" + str(n)
                                    print(f'\rSubmitted {submitted} / {len(k_values) * len(n_values)} jobs', end='\r')
                                    submitted += 1
                                else:
                                    loss_values = run_alg(problem, vars(options), alg_import.Algorithm)
                                    key = "k=" + str(k) + ", n=" + str(n)
                                    log_dict[key] = loss_values
                                    print_progress_bar(run_num, len(k_values) * len(n_values))
                                    run_num += 1

                        if options.is_threaded:
                            for future in futures:
                                key = futures[future]
                                try:
                                    loss_values = future.get()
                                except Exception as exc:
                                    print('%r generated an exception: %s' % (key, exc))
                                    raise exc
                                else:
                                    log_dict[key] = loss_values
                                    print_progress_bar(run_num, len(k_values) * len(n_values))
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
                        save_csv_multi(options.num_particles, log_dict, alg_import, str(options.seed))
            else:
                loss_values = run_alg(problem, vars(options), alg_import.Algorithm)

                # There is no point to plot a single point. It may be used to plot all loss outputs from a single
                # algorithm though.
                # if options.is_plot_exported:
                # plot_single_data(min(loss_values), "k=" + str(options.k) + ", n=" + str(options.size))

                if options.is_csv_exported:
                    save_csv_single(loss_values, options, alg_import)


def main():
    # build the parser for implementation
    # Default parser values:
    parser = build_parser()

    options = parser.parse_args()
    #
    random.seed(options.seed)
    np.random.seed(options.seed)

    # Add imported algorithm modules to this list to have them be used.
    for alg in [GA]:
        # for alg in [GA, PSO, RAN, GSA]:
        print("running: {}".format(alg.to_string()))
        setup_alg(options, alg)


if __name__ == '__main__':
    main()
