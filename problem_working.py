

from __future__ import print_function, division

import pickle
from collections import deque, OrderedDict
from operator import itemgetter
from random import shuffle
from pprint import pprint
from time import time

from numpy import union1d, intersect1d


REPEAT = 10
print('Using {} repetitions for timing'.format(REPEAT))

### Utilities
def confirm_solution(a, b):
    """Confirm that a and b are lists of similar sets"""
    if a == 'IGNORE' or b == 'IGNORE':
        return True
    a = [set(element) for element in a]
    b = [set(element) for element in b]
    for seta in a:
        for index_b in list(range(len(b))):
            setb = b[index_b]
            if seta == setb:
                # Match found, pop from b and break inner loop
                b.pop(index_b)
                break
        else:  # No match found in b, no good
            return False

    return len(b) == 0


def timeme(repetitions):
    def timemedec(function):
        """Time me decorator"""
        def inner_function(*args, **kwargs):
            # The original algorithm changes the input inplace, which
            # makes it artificially faster for consequtive runs
            arg0s = [list(args[0]) for _ in range(repetitions)]
            starttime = time()
            for arg0 in arg0s:
                return_value = function(arg0, *args[1:], **kwargs)
            execution_time = time() - starttime
            return return_value, execution_time / repetitions
        return inner_function
    return timemedec


### Implementations
@timeme(REPEAT)
def original(l):
    """Original implementation"""
    len_l = len(l)
    i=0
    while i < (len_l - 1):
        for j in range(i + 1, len_l):
            # i,j iterate over all pairs of l's elements including new 
            # elements from merged pairs. We use len_l because len(l)
            # may change as we iterate
            i_set = set(l[i])
            j_set = set(l[j])

            if len(i_set.intersection(j_set)) > 0:
                # Remove these two from list
                l.pop(j)
                l.pop(i)

                # Merge them and append to the orig. list
                ij_union = list(i_set.union(j_set))
                l.append(ij_union)

                # len(l) has changed
                len_l -= 1

                # adjust 'i' because elements shifted
                i -= 1

                # abort inner loop, continue with next l[i]
                break
        i += 1
    return l


@timeme(REPEAT)
def less_conversion(l):
    """Original implementation with less type conversion"""
    l = [set(array) for array in l]
    len_l = len(l)
    i=0
    while i < (len_l - 1):
        for j in range(i + 1, len_l):
            # i,j iterate over all pairs of l's elements including new 
            # elements from merged pairs. We use len_l because len(l)
            # may change as we iterate
            i_set = l[i]
            j_set = l[j]

            if len(i_set.intersection(j_set)) > 0:
                # Remove these two from list
                l.pop(j)
                l.pop(i)

                # Merge them and append to the orig. list
                ij_union = i_set.union(j_set)
                l.append(ij_union)

                # len(l) has changed
                len_l -= 1

                # adjust 'i' because elements shifted
                i -= 1

                # abort inner loop, continue with next l[i]
                break
        i += 1
    return l


@timeme(REPEAT)
def no_pop(list_of_arrays):
    """Pure python group by common member

    This implementation works by merging into the existing sets
    """
    # Turn all sets in arrays into proper sets
    list_of_sets = [set(array) for array in list_of_arrays]
    number_of_sets = len(list_of_sets)

    # Loop over sets to merge into
    for outer_index in range(number_of_sets):
        outer_set = list_of_sets[outer_index]

        # If the set is None, it means that it has already been merged
        if outer_set is None:
            continue

        # Loop over sets to possible merge
        for inner_index in range(number_of_sets):
            if inner_index == outer_index:
                continue
            inner_set = list_of_sets[inner_index]

            # If the set at inner_index has already been used
            if inner_set is None:
                continue

            # If there is a common member, merge and set to None
            if len(outer_set.intersection(inner_set)) > 0:
                outer_set.update(inner_set)
                list_of_sets[inner_index] = None

    list_of_sets = [set_ for set_ in list_of_sets if set_ is not None]
    return list_of_sets


@timeme(REPEAT)
def numpy_simple(list_of_arrays):
    """No pop implementation using numpy functions"""
    # Turn all sets in arrays into proper sets
    number_of_arrays = len(list_of_arrays)

    # Loop over sets to merge into
    for outer_index in range(number_of_arrays):
        outer_array = list_of_arrays[outer_index]

        # If the set is None, it means that it has already been merged
        if outer_array is None:
            continue

        # Loop over sets to possible merge
        for inner_index in range(number_of_arrays):
            if inner_index == outer_index:
                continue
            inner_array = list_of_arrays[inner_index]

            # If the set at inner_index has already been used
            if inner_array is None:
                continue

            # If there is a common member, merge and set to None
            if intersect1d(outer_array, inner_array, assume_unique=True).size > 0:
                outer_array = union1d(outer_array, inner_array)
                list_of_arrays[inner_index] = None

        list_of_arrays[outer_index] = outer_array

    list_of_arrays = [array for array in list_of_arrays if array is not None]
    return list_of_arrays


@timeme(REPEAT)
def numba_one_array(list_of_arrays):
    """Try hand at numba implementation"""
    
    return 'IGNORE'


### Timings
def print_results(results, relative_to):
    """Print the results out nicely
               method1  method2  method3
    structure1
    structure2
    """
    # New line at beginning
    print()
    # Make print templates, forst methods
    anyresult = results.itervalues().next()
    method_width = max(20, max(len(key) for key in anyresult.keys()) + 3)
    def method(time_, speed_up):
        string = '{:.1e} ({:.1f}x su)'.format(time_, speed_up)
        string += ' ' * (method_width - len(string))
        return string

    # then structures
    structure_width = len('structure1 ')
    structure_template = '{{: <{}}}'.format(structure_width)

    # Make and print header
    header = ''.join(['{{: <{}}}'.format(method_width).format(key)
                       for key in anyresult.keys()])
    print(' ' * structure_width + header)

    # Iterate over results and print out
    for structure_name, structure_result in sorted(results.items(),
                                                   key=itemgetter(0)):
        print(structure_template.format(structure_name), end='')
        relative_to_time = structure_result[relative_to]
        # OrderedDict som comign out in order
        for value in structure_result.values():
            print(method(value, relative_to_time / value), end='')
        print()
    
def time_one_structure(structure_name):
    """Time a single structure"""
    print("Timing", structure_name)
    times = OrderedDict()
    # Calculate the correct solution from the original algorithm
    structure = pickle.load(open(structure_name))
    correct_result, times['original'] = original(structure)

    # Iterate over alternative solutions
    for function_name in ('less_conversion', 'no_pop', 'numpy_simple',
                          'numba_one_array'):
        # Load structure, calculate and check result
        structure = pickle.load(open(structure_name))
        function = globals()[function_name]
        result, times[function_name] = function(structure)
        if not confirm_solution(correct_result, result):
            raise RuntimeError('Bad result for ' + function_name)
    return times


def main():
    """Main function"""
    results = {}
    for suffix in [''] + [str(n) for n in range(1, 4)]:
        name = 'structure{}'.format(suffix)
        results[name] = time_one_structure(name + '.pckl')

    print_results(results, 'original')
    #pprint(results)


main()
