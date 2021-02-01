# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 6:42:20 2021

@author: Cory Kromer-Edwards

Runs GridSearch for a given algorithm.
"""

# For parallel computing
import multiprocessing
import signal

from progress_bar import print_progress_bar


def _thread_caller(params, alg_class):
    alg = alg_class(problem=0, debug=-1, num_particles=50, **params)
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