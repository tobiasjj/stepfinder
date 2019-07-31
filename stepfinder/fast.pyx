#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Stepfinder, find steps in data with low SNR
# Copyright 2016,2017,2018,2019 Tobias Jachowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.double_t DTYPE_d
ctypedef np.int_t DTYPE_i

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _iterative_variance(np.ndarray[DTYPE_d, ndim=1] data not None,
                        np.ndarray[DTYPE_d, ndim=1] sf not None,
                        np.ndarray[DTYPE_d, ndim=1] sb not None,
                        np.ndarray[DTYPE_d, ndim=1] xf not None,
                        np.ndarray[DTYPE_d, ndim=1] xb not None,
                        start, stop, window):
    """
    Calculate iterative variances
    """
    # check whether boundaries of arrays are beeing respected
    # stop > start, otherwise range does not trigger any loop iteration
    # -> lower boundary checks for start are sufficient
    # -> upper boundary checks for stop are sufficient
    minlen = min([len(data), len(sf), len(sb), len(xf), len(xb)])
    if start < 1 or stop < 0 or window < 0 \
            or start - window < 0 \
            or stop - 1 > minlen \
            or stop - 1 + window - 1 > minlen:
        raise IndexError('Indexing error! Only positive indices within the '
                         'bounds of the arrays are allowed.')

    cdef unsigned int w = window
    cdef unsigned int k
    # calculate elements by "shifting" the sum
    for k in range(start, stop):
        sf[k] = sf[k - 1] \
            + (data[k] - xf[k])**2 \
            - (data[k - w] - xf[k - w])**2
        sb[k] = sb[k - 1] \
            + (data[k + w - 1] - xb[k + w - 1])**2 \
            - (data[k - 1] - xb[k - 1])**2


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _delete_close_center(step_bounds not None,
                         np.ndarray direction not None,
                         unsigned int max_step_width,
                         unsigned int min_step_spacing,
                         switch_accept=True,
                         fuse=True,
                         copy=True):
    """
    Iteratively check distance of the center of one plateau (start, stop) to
    the following one, and either fuse them or delete the next one, if the
    distance is too small.
    Split steps based whose step_bounds > max_step_width.
    This will probably lead to too many FPs, but these will be automatically
    deleted later in the function `delete_small_steps()`. A Correct threshold
    would be 2 * min_step_spacing (see Smith1998), but this would lead to too
    many undetected steps (due to noise)
    """
    if copy:
        _start = step_bounds[:, 0].copy()
        _stop = step_bounds[:, 1].copy()
    else:
        _start = step_bounds[:, 0]
        _stop = step_bounds[:, 1]

    cdef np.ndarray[DTYPE_i, ndim=1] start = _start
    cdef np.ndarray[DTYPE_i, ndim=1] stop = _stop
    cdef np.ndarray[DTYPE_d, ndim=1] center = (start + (stop - start) / 2)
    cdef np.ndarray[DTYPE_i, ndim=1] keep = np.ones_like(direction, dtype=int)
    cdef bint _switch_accept = switch_accept
    cdef bint _fuse = fuse
    cdef unsigned int i = 0

    while i < len(step_bounds) - 1:
        # Fuse/delete if:
        # a) max_step_width not reached
        # b) distance of centers too small
        # c) same direction (if switch_accept)
        if stop[i] - start[i] <= max_step_width \
                and center[i + 1] - center[i] < min_step_spacing \
                and (direction[i + 1] == direction[i]
                     or not switch_accept):
                # TODO: Alternatively split plateaus actively and do this
                # corresponding to the center of step_mass?!
                # Otherwise, if the splitting is not done at the right
                # position, one of the each other closely following steps
                # will not have a sufficiently great step size and,
                # therefore, will be deleted.
            if fuse:
                # Correct the start of the following step_bound to be the
                # start of this one
                start[i + 1] = start[i]
                # Correct the center of the now bigger following step_bound
                center[i + 1] = start[i + 1] \
                    + (stop[i + 1] - start[i + 1]) / 2
            else:
                # Transfer the values from this step_bound to the following
                start[i + 1] = start[i]
                stop[i + 1] = stop[i]
                center[i + 1] = center[i]
                direction[i + 1] = direction[i]
            # Delete the old current step_bound, after the values of the
            # next one had been corrected / transferred
            keep[i] = 0
        i += 1

    return np.c_[start[keep == 1], stop[keep == 1]], direction[keep == 1]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _calculate_plateau_heights(np.ndarray[DTYPE_d, ndim=1] data not None,
                               np.ndarray[DTYPE_i, ndim=2] plateaus not None,
                               np.ndarray[DTYPE_d, ndim=1] plateau_heights not None):
    cdef unsigned int i
    cdef unsigned int start
    cdef unsigned int stop
    for i in range(0, len(plateaus)):
        # take data between the current and the next step
        start = plateaus[i, 0]
        stop = plateaus[i, 1]
        plateau_heights[i] = np.sum(data[start:stop]) / (stop - start)
