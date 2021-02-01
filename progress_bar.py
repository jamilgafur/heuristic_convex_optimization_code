# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 6:45:30 2021

@author: Cory Kromer-Edwards

Prints progress bar to console.
Starter code from from StackOverflow answer:
    https://stackoverflow.com/a/34325723

Updated to write line max to be width of console - 8% (EX: width = 235 then printline will be 206
"""
import shutil


def print_progress_bar(iteration, total, decimals=3, fill='#', prefix='Progress:', suffix='Complete',
                       print_end='\r'):
    """
    Print a progress bar to the console. It will overwrite itself on each update.
    :param int iteration: The iteration that the task is on
    :param int total: Total number of iterations for task
    :param int decimals: How many decimals to show percent
    :param character fill: What character to 'fill' the progress bar to show progress
    :param string prefix: What to show at beginning of progress bar
    :param string suffix: What to show at end of progress bar
    :param character print_end: What should print function put at end of string
    :return:
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
