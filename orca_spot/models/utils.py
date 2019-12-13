"""
Module: utils.py
Authors: Christian Bergler, Hendrik Schroeter
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 12.12.2019
"""

"""Return `same` padding for a given kernel size."""
def get_padding(kernel_size):
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return tuple(s // 2 for s in kernel_size)
