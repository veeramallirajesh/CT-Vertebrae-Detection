import json
import re
from math import log
from typing import Union, Iterable, Tuple

import numpy as np
import pandas
import tabulate
from scipy.ndimage import zoom


def plot_with_conf(x, y_mean, y_conf, alpha=0.5, **kwargs):
    import matplotlib.pyplot as plt
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y_mean, **kwargs)
    y_mean = np.array(y_mean)
    y_conf = np.array(y_conf)
    lb = y_mean - y_conf
    ub = y_mean + y_conf

    ax.fill_between(x, lb, ub, facecolor=base_line.get_color(), alpha=alpha)


def print_attributes(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    for attr in dir(obj):
        if attr in ignore:
            continue
        if attr.startswith('_'):
            continue
        if not include_methods and callable(obj.__getattr__(attr)):
            continue
        print(attr, ':', obj.__getattr__(attr).__class__.__name__, ':', obj.__getattr__(attr))


# Print iterations progress
def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:" + str(4 + decimals) + "." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if getattr(print_progress_bar, 'last_printed_value', None) == (prefix, bar, percent, suffix):
        return
    print_progress_bar.last_printed_value = (prefix, bar, percent, suffix)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # # Print New Line on Complete
    # if iteration == total - 1:
    #     print()


def attr_dir(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    return {attr: obj.__getattr__(attr)
            for attr in dir(obj)
            if not attr.startswith('_') and (
                    include_methods or not callable(obj.__getattr__(attr))) and attr not in ignore}


def zoom_to_shape(a: np.ndarray, shape: Tuple, mode: str = 'smooth', verbose=1):
    from keras import backend
    a = np.array(a, dtype=backend.floatx())  # also does a copy
    shape_dim = len(a.shape)
    if len(a.shape) != len(shape):
        raise ValueError('The shapes must have the same dimension but were len({0}) = {1} (original) '
                         'and len({2}) = {3} desired.'.format(a.shape, len(a.shape), shape, len(shape)))
    if len(shape) == 0:
        return a
    zoom_factors = tuple(shape[idx] / a.shape[idx] for idx in range(shape_dim))

    def _current_index_in_old_array():
        return tuple(slice(0, length) if axis != current_axis else slice(current_pixel_index, current_pixel_index + 1)
                     for axis, length in enumerate(a.shape))

    def _current_pixel_shape():
        return tuple(length if axis != current_axis else 1
                     for axis, length in enumerate(a.shape))

    def _current_result_index():
        return tuple(
            slice(0, length) if axis != current_axis else slice(pixel_index_in_result, pixel_index_in_result + 1)
            for axis, length in enumerate(a.shape))

    def _current_result_shape():
        return tuple(orig_length if axis != current_axis else shape[axis]
                     for axis, orig_length in enumerate(a.shape))

    if mode == 'constant':
        result = zoom(a, zoom_factors)
        assert result.shape == shape
        return result
    elif mode == 'smooth':
        result = a
        for current_axis, zoom_factor in sorted(enumerate(zoom_factors), key=lambda x: x[1]):
            result = np.zeros(_current_result_shape(), dtype=backend.floatx())
            # current_length = a.shape[current_axis]
            desired_length = shape[current_axis]
            current_pixel_index = 0
            current_pixel_part = 0  # how much of the current pixel is already read
            for pixel_index_in_result in range(desired_length):
                pixels_remaining = 1 / zoom_factor
                pixel_sum = np.zeros(_current_pixel_shape())
                while pixels_remaining + current_pixel_part > 1:
                    pixel_sum += (1 - current_pixel_part) * a[_current_index_in_old_array()]
                    current_pixel_index += 1
                    pixels_remaining -= (1 - current_pixel_part)
                    current_pixel_part = 0

                # the remaining pixel_part
                try:
                    pixel_sum += pixels_remaining * a[_current_index_in_old_array()]
                except (IndexError, ValueError):
                    if verbose:
                        print('WARNING: Skipping {0} pixels because of numerical imprecision.'.format(pixels_remaining))
                else:
                    current_pixel_part += pixels_remaining

                # insert to result
                pixel_sum *= zoom_factor

                result[_current_result_index()] = pixel_sum
            a = result

        assert result.shape == shape
        return result
    else:
        return NotImplementedError('Mode not available.')


def throws(_e: Union[Exception, Iterable[Exception]]):
    return lambda f: f


def dummy_computation(*_args, **_kwargs):
    pass


def my_tabulate(data, **params):
    if data == [] and 'headers' in params:
        data = [(None for _ in params['headers'])]
    tabulate.MIN_PADDING = 0
    return tabulate.tabulate(data, **params)


def ce_loss(y_true, y_predicted):
    return -(y_true * log(y_predicted) + (1 - y_true) * log(1 - y_predicted))


# def multinomial(n, bins):
#     if bins == 0:
#         if n > 0:
#             raise ValueError('Cannot distribute to 0 bins.')
#         return []
#     remaining = n
#     results = []
#     for i in range(bins - 1):
#         x = binomial(remaining, 1 / (bins - i))
#         results.append(x)
#         remaining -= x
#
#     results.append(remaining)
#     return results


class UnknownTypeError(Exception):
    pass


# def shape_analysis(xs):
#     composed_dtypes = [list, tuple, np.ndarray, dict, set]
#     base_dtypes = [str, int, float, type, object]  # TODO add class and superclass of xs first element
#     all_dtypes = composed_dtypes + base_dtypes
#     if isinstance(xs, np.ndarray):
#         outer_brackets = ('[', ']')
#         shape = xs.shape
#         dtype = xs.dtype
#     elif isinstance(xs, tuple):
#         outer_brackets = ('(', ')')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, list):
#         outer_brackets = ('[', ']')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, dict) or isinstance(xs, set):
#         outer_brackets = ('{', '}')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif any(isinstance(xs, t) for t in base_dtypes):
#         for t in base_dtypes:
#             if isinstance(xs, t):
#                 return str(t.__name__)
#         raise AssertionError('This should be unreachable.')
#     else:
#         raise UnknownTypeError('Unknown type:' + type(xs).__name__)
#
#     if shape and shape != '?':
#         return outer_brackets[0] + str(xs.shape) + ' * ' + str(dtype) + outer_brackets[1]
#     else:
#         return outer_brackets[0] + outer_brackets[1]


def shorten_name(name):
    name = re.sub(r'\s+', r' ', str(name))
    name = name.replace(', ', ',')
    name = name.replace(', ', ',')
    name = name.replace(' ', '_')
    return re.sub(r'([A-Za-z])[a-z]*_?', r'\1', str(name))


def split_df_list(df, target_column):
    """
    df = data frame to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a data frame with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.

    SOURCE: https://gist.github.com/jlln/338b4b0b55bd6984f883
    """

    def split_list_to_rows(row, row_accumulator):
        split_row = json.loads(row[target_column])
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows,))
    new_df = pandas.DataFrame(new_rows)
    return new_df


try:
    import winsound


    def beep(*args, **kwargs):
        winsound.Beep(*args, **kwargs)
except ImportError:
    def beep(*_args, **_kwargs):
        pass
