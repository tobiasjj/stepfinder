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

import matplotlib.pyplot as plt
import numpy as np
import pyximport; pyximport.install()
import warnings
from collections import namedtuple

from .fast import _iterative_variance
from .fast import _delete_close_center
from .fast import _calculate_plateau_heights


StepsSimulated = namedtuple('StepsSimulated', 'data resolution noise dwells '
                            'indices number')
FBNLFilterResult = namedtuple('FBNLFilterResult', 'data resolution window '
                              'window_var p data_filtered sf sb f b xf xb '
                              'step_mass step_size noise noise_mean aSNR mSNR '
                              'STD outls')
FBNLFilterBankResult = namedtuple('FBNLFilterBankResult',
                                  'windows windows_var data_filtered_mean '
                                  'step_mass_mean step_size_mean noises_mean '
                                  'data resolution window window_var p '
                                  'data_filtered sf sb f b xf xb step_mass '
                                  'step_size noise noise_mean aSNR mSNR STD '
                                  'outls')
StepFinderResult = namedtuple('StepFinderResult', 'fbnl_filter '
                              'expected_min_step_size y_c '
                              'expected_min_dwell_t min_dwell_time '
                              'switch_accept step_size_threshold '
                              'steps_pre steps min_sizes_pre min_sizes '
                              'quality_pre quality')
Steps = namedtuple('Steps', 'indices direction bounds number plateaus '
                   'p_centers step_sizes plateau_heights dwell_points')
StepQuality = namedtuple('StepQuality', 'step_sd step_noise '
                         'step_noise_over_sd')


def moving_mean(data, window):
    """
    Calculate a filtered signal by using a moving mean. The first datapoint is
    the mean of the first `window` datapoints and the last datapoint is the
    mean of the last `window` datapoints of the original data. This function
    does not handle the lost edges of the data, i.e. the filtered data is
    shortened by `window` datapoints.

    Parameters
    ----------
    data : 1D numpy.ndarray of type float
        Data to calculate the rolling mean from.
    window : int
        Length of the window to calculate the rolling mean with.

    Returns
    -------
    1D numpy.ndarray of type float
        The data filtered with a rolling mean.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def get_contiguous_segments(plateaus, min_distance_center=1, min_length_high=1,
                            min_length_low=1, check_center_distance_first=True,
                            fuse=True, validate_truncated=True):
    # original source derived from
    # http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-bool_array-in-a-numpy-array
    # modified and extended by Tobias Jachowski
    """
    Find contiguous True values (plateau) of the boolean array `plateaus`.
    Return a 2D array where the first column is the start index of the region
    and the second column is the stop index (segments).

    Parameters
    ----------
    plateaus : np.ndarray of type bool
    min_distance_center : int
        Minimum distance, the center of two successive plateaus need to be
        apart from each other. If the distance is shorter, the stop of the
        leading and the start of the following plateau are neglected,
        effectively fusing the two plateaus.
    min_length_high : int
        Minimum number of contiguous True values a plateau has to have to get
        detected as one. If the plateau ist shorter, it is assumed to be no
        plateau and is therefore neglected. The minimum length of the plateaus
        usually is checked, after the distance of the center was checked.
        See parameter `check_center_distance_first`.
    min_length_low : int
        Minimum number of contiguous False values a valley has to have to get
        detected as one. If the valley is shorter, it is assumed to be part of
        a greater plateau, consisting of the preceding plateau, the short
        valley itself, and the succeeding plateau and is therefore neglected.
        The length of valleys is checked after the minimum length of the
        plateaus was checked.
    check_center_distance_first : bool
        Check `min_distance_center` before `min_length_high`.
    fuse : bool
        Fuse plateaus with a too short center distance, or delete the ones
        following the other ones too close.
    validate_truncated : bool, optional
        Validate the length of plateaus starting or stopping at the very first
        [0] or last [-1] index of the plateaus array, i.e. are possibly
        truncated plateaus. If set to True, even these plateaus need to have at
        least `min_length_high` datapoints. If set to False, even truncated -
        and therefore possibly too short - plateaus are detected.

    Returns
    -------
    2D numpy.ndarray of type int
        Segments of plateaus with start indices in the first and stop indices
        in the second column.
    """

    if plateaus.size == 0:
        return np.empty((0, 2), dtype=int)

    # Find the indices of changes in "plateaus"
    d = np.diff(plateaus)
    idx, = d.nonzero()

    # We need to start things after the change in "plateaus". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    lock_first = False
    if plateaus[0]:
        # If the start of plateaus is True prepend a 0
        idx = np.r_[0, idx]
        lock_first = not validate_truncated

    lock_last = False
    if plateaus[-1]:
        # If the end of plateaus is True, append the length of the array
        idx = np.r_[idx, plateaus.size]
        lock_last = not validate_truncated

    # No contiguous segments detected
    if idx.size < 2:
        return np.empty((0, 2), dtype=int)

    idx_start = idx[:-1:2]
    idx_stop = idx[1::2]

    def check_center_distance(idx_start, idx_stop):
        if min_distance_center == 1:
            return idx_start, idx_stop

        # Create a linked list from the two arrays describing the start and
        # stop inidices of the plateaus. This implementation avoids the (for
        # long lists) expensive pop() function of lists.
        class Plateau(object):
            def __init__(self, start, stop, next=None):
                self.next = next
                self._start = start
                self._stop = stop
                self.center = start + (stop - start) / 2
            @property
            def start(self):
                return self._start
            @start.setter
            def start(self, start):
                self._start = start
                self.center = self.start + (self.stop - self.start) / 2
            @property
            def stop(self):
                return self._stop
            @stop.setter
            def stop(self, stop):
                self._stop = stop
                self.center = self.start + (self.stop - self.start) / 2
        current = Plateau(idx_start[-1], idx_stop[-1])
        for start, stop in zip(idx_start[-2::-1], idx_stop[-2::-1]):
            current = Plateau(start, stop, current)
        first = current

        # Iteratively check distance of the center of one plateau (start, stop)
        # to the following one, and either fuse them or delete the next one, if
        # the distance is too small
        num_plateaus = len(idx_start)
        while current.next is not None:
            if current.next.center - current.center < min_distance_center:
                if fuse:
                    # Correct the stop of the leading plateau to be the one
                    # of the following and implicitly correct the center
                    # position of the now bigger leading plateau
                    current.stop = current.next.stop
                # Delete the following plateau and its corresponding center
                current.next = current.next.next
                # Correct number for deleted plateau
                num_plateaus -= 1
            else:
                # select the next plateau
                current = current.next

        # Create numpy arrays from linked list
        start = np.empty(num_plateaus)
        stop = np.empty(num_plateaus)
        current = first
        i = 0
        while current is not None:
            start[i] = current.start
            stop[i] = current.stop
            current = current.next
            i += 1

        return start, stop

    def check_length_high(idx_start, idx_stop):
        if min_length_high == 1:
            return idx_start, idx_stop
        # Check length of plateaus first
        # (stop_high - start_high) or (uneven_index - even_index) >= min_length
        plateau = idx_stop - idx_start >= min_length_high
        # If stops are locked, make sure to reset them to True
        plateau[0] = plateau[0] or lock_first
        plateau[-1] = plateau[-1] or lock_last
        # Select only accepted starts/stops (plateaus)
        idx_start = idx_start[plateau]
        idx_stop = idx_stop[plateau]
        return idx_start, idx_stop

    checks = [check_length_high, check_center_distance]
    if check_center_distance_first:
        checks.reverse()
    for check in checks:
        idx_start, idx_stop = check(idx_start, idx_stop)

    # Only one plateau detected, ignore all valleys
    if idx_start.size <= 1:
        idx = np.sort(np.r_[idx_start, idx_stop])
        idx.shape = (-1, 2)
        return idx

    # Check length of valleys
    # (stop_low - start_low) or (even_index - uneven_index) >= min_length
    valley = np.where(idx_start[1:] - idx_stop[:-1] >= min_length_low)[0]
    # Select only accepted stops/starts (valleys) and keep first start and last
    # stop of plateaus
    start = idx_start[np.r_[0, valley + 1]]
    stop = idx_stop[np.r_[valley, idx_stop.size - 1]]

    # Concatenate and sort passed start indices
    idx = np.sort(np.r_[start, stop])
    # Reshape the result into two columns
    idx.shape = (-1, 2)

    return idx


def idx_to_idx_segments(idx, start=None, stop=None):
    """
    Create stop indices
    Concatenate and sort start (and stop) indices
    """
    if idx.size >= 2:
        idx_segs = np.sort(np.r_[idx, idx[1:-1]])
    else:  # size is 1 or empty
        idx_segs = np.array([], dtype=int)

    if idx.size >= 2:
        if start is not None and start < idx[0]:
            idx_segs = np.r_[start, idx[0], idx_segs]
        if stop is not None and stop > idx[-1]:
            idx_segs = np.r_[idx_segs, idx[-1], stop]
    if idx.size == 1:
        if start is not None and start < idx[0]:
            idx_segs = np.r_[start, idx[0], idx_segs]
        if stop is not None and stop > idx[-1]:
            idx_segs = np.r_[idx_segs, idx[-1], stop]

    # Reshape the result into two columns
    idx_segs.shape = (-1, 2)
    return idx_segs


def iqr_outlier_threshold(signal, iqr_factor=1.5):
    """
    Calculate the interquartile range (IQR) threshold of data points of a
    signal to be classified as outliers.

    A datapoints with values that are further than `iqr_factor` times the IQR
    away from 1st or 3rd quartile (i.e. the threshold) can be classified as
    outliers. An `iqr_factor` of 1.5 is gives a threshold fo weak outliers, an
    `iqr_factor` of 3.0 gives a threshold for strong outliers.

    Parameters
    ----------
    signal : 1D numpy.ndarray of type float
        Data to calculate the interquartile range threshold from.
    iqr_factor : int
        Factor the IQR is multiplied with to calculate the threshold.

    Returns
    -------
    int
        The calculated threshold.
    """
    qrt1 = np.nanpercentile(signal, 25, interpolation='midpoint')
    qrt3 = np.nanpercentile(signal, 75, interpolation='midpoint')
    iqr = qrt3 - qrt1
    upper = qrt3 + iqr * iqr_factor
    lower = qrt1 - iqr * iqr_factor
    return (upper - lower) / 2


def log_spaced_ints(start, stop, number):
    """
    Create an array with `number` log spaced integer values, the first value
    starting at `start` and the last value beeing `stop`.

    Parameters
    ----------
    start : int
        Value of the first number. Value should be >= 1.
    stop : int
        Value of the last number. Value should be >= `start`.
    number : int
        Number of values to create.

    Returns
    -------
    1D numpy.ndarray of type int
        The array with the log spaced int values.
    """
    start = np.log10(start)
    stop = np.log10(stop)
    windows = np.round(np.logspace(start, stop, number))
    return np.unique(windows.astype(int))


def log_spaced_time_windows(tmin, tmax=None, resolution=None, number=None):
    """
    Create an array with `number` log spaced time window (int) values, the
    first value starting at `tmin` * `resolution` and the last value beeing
    `stop` * `resolution`.

    Parameters
    ----------
    tmin : float
        Length of the first time window.
    tmax : float, optional
        Length of the last window. Defaults to `tmin`.
    resolution : float, optional
        Resolution of the time window length. Defaults to 1.0.
    number : int, optional
        Number of values to create. Defaults to 1.

    Returns
    -------
    1D numpy.ndarray of type int
        The array with the log spaced time window values.
    """
    tmax = tmax or tmin
    resolution = resolution or 1.0
    number = number or 1
    start = tmin * resolution
    stop = tmax * resolution
    windows = log_spaced_ints(start, stop, number=number)
    return windows


def cap_data(data, cap_length, inspect_length):
    """
    Cap data at the ends with values distributed normally.

    The mean and the standard deviation of the data points for the caps is
    calculated from the first and last `inspect_length` datapoints of the
    `data`. The data is than capped with `cap_length` datapoints at both ends.

    Parameters
    ----------
    data : 1D numpy.ndarray of type float
        The data which should be capped with normally distributed datapoints
        at both ends.
    cap_length : int
        The length of the caps the data should be extended with.
    inspect_length : int
        The length of the datapoints that should be used to calculate the mean
        and the standard deviation of the caps.

    Returns
    -------
    1D numpy.ndarray of type float
        The capped data.
    """
    median_start, std_start = \
        np.median(data[:int(inspect_length) + 1]), \
        np.std(data[:int(inspect_length) + 1], ddof=1)
    median_stop, std_stop = \
        np.median(data[-int(inspect_length):]), \
        np.std(data[-int(inspect_length):], ddof=1)
    noise_start = \
        np.random.normal(loc=median_start, scale=std_start, size=cap_length)
    noise_stop = \
        np.random.normal(loc=median_stop, scale=std_stop, size=cap_length)
    data = np.r_[noise_start, data, noise_stop]
    return data


def simulate_steps(duration=10.0, resolution=1000.0, dwell_time=1.0,
                   step_size=8.0, SNR=0.5, movement='diffusive',
                   constant_dwell=False):
    """
    Simulate data of steps with noise.

    Parameters
    ----------
    duration : float
        Duration of whole trace in seconds.
    resolution : float
        Resolution of the trace in data points / second.
    dwell_time : float
        Time between steps in seconds.
    step_size : float
        The size of a step.
    SNR : float
        Signal to noise ratio, i.e. the size of a step divided by the standard
        deviation of the noise.
    movement : str
        'monoton' or 'diffusive' movement.
    constant_dwell : bool
        Constant or exponential distributed dwell times.

    Returns
    -------
    StepsSimulated : namedtuple
    """
    # Number of data points between steps
    dwell_points = np.round(dwell_time * resolution).astype(int)

    # Number of data points in whole trajectory
    length = np.round(duration * resolution).astype(int)

    # Random steps with constant or exponential distributed dwell times
    if movement == 'monoton' and constant_dwell:
        data = step_size * np.floor(np.arange(0, length) / dwell_points)
        dwells = np.full(length + 1, dwell_points, dtype=int)
        indices = np.arange(dwell_points, length, dwell_points)
    else:
        data = np.empty(length)
        height = 0
        i = 0
        step = 0
        points = dwell_points
        dwells = np.empty(0, dtype=int)
        indices = np.empty(0, dtype=int)
        while i < length:
            if not constant_dwell:
                random = np.random.exponential(scale=dwell_points)
                points = np.ceil(random).astype(int)
            points = min(length - i, points)
            if movement == 'monoton':
                y = np.full(points, step * step_size)
                step += 1
            else:  # 'diffusive':
                sign = np.random.choice([-1, 1])
                height = height + sign * step_size
                y = np.full(points, height)
            data[i:i + points] = y
            i += points
            dwells = np.r_[dwells, points]
            if i < length:
                indices = np.r_[indices, i]

    # Standard deviation of noise corresponding to SNR set by user
    noise_STD = step_size / SNR
    noise = np.random.normal(scale=noise_STD, size=len(data))

    return StepsSimulated(data, resolution, noise, dwells, indices,
                          len(indices))


def filter_fbnl(data, resolution, window, window_var=None, p=None,
                cap_data=True):
    """
    Forward Backward Nonlinear Filter

    Implementation of the Filter proposed by Chung and Kennedy (1991) [1].
    Additionally, the function returns the step_size, noise and step_mass, used
    by the step finder algorithm as described by Smith (1998) [2].

    Parameters
    ----------
    data : np.ndarray of type float
        data to be filtered
    resolution : float
        Resolution of the data in Hz.
    window : int
        Window-length. Steps shorter than 2 * window will be smoothed out.
    window_var : int
        Averaging length for variances sf and sb (should be <= window to make
        sense). Defaults to window.
    p : float
        Nonlinearity for calculating weights. 3 to 6 is ok, 10 is much, 1 is
        quite low -> weight = s.d.^(-2*p) (-> 0.5=std.-dev., 1=variance, ...)
        Defaults to 1.
    cap_data : bool
        Cap the data to protect the ends from be "eaten up" by the filtering
        process. See also function `cap_data()`.

    Returns
    -------
    FBNLFilterResult : namedtuple

    References
    ----------
    [1] Chung, S.H. & Kennedy, R.A. 1991 "Forward-backward nonlinear filtering
    technique for extracting small biological signals from noise." J. Neurosci.
    Meth. 40, 71-86

    [2] Smith, D.A. 1998 "A Quantitative Method for the Detection of Edges in
    Noisy Time-Series." Phil. Trans. R. Soc. Lond. B 353, 1969-1981
    """
    if cap_data:
        f = _filter_fbnl_capped
    else:
        f = _filter_fbnl

    (data, resolution, window, window_var, p, data_filtered, sf, sb, f, b, xf,
     xb) = f(data, resolution, window, window_var=window_var, p=p)

    # Calculate the step_size, noise, and step_masses
    # noise = weighted moving variances
    # "stepamplitude": step_size over noise
    step_size = xb - xf
    with np.errstate(invalid='ignore'):
        noise = np.sqrt(f * sf + b * sb)
    step_mass = step_size / noise
    noise_mean = np.nanmean(noise)

    # Calculate the SNR and the STD of step_masses
    # sm = step_mass.copy()
    sm = step_mass[~np.isnan(step_mass)]  # = 0.0

    # Select all datapoints above 3 times the interquartile range
    # (i.e. outliers) and assume them to be the outls we want to detect
    iqr_threshold = iqr_outlier_threshold(sm, iqr_factor=3.0)
    outls = abs(sm) > iqr_threshold

    # Get the values of outls and noise
    sm_outls = sm[outls]
    sm_noise = sm[~outls]

    step_mass_STD = sm_noise.std(ddof=1)  # STD noise

    if np.sum(outls) > 0:
        step_mass_SNR = sm_outls / step_mass_STD
        step_mass_SNR_mean = np.mean(np.abs(step_mass_SNR))
        step_mass_SNR_median = np.median(np.abs(step_mass_SNR))
    else:
        step_mass_SNR_mean = np.nan
        step_mass_SNR_median = np.nan

    sm = step_mass.copy()
    sm[np.isnan(sm)] = 0.0
    outls = abs(sm) > iqr_threshold

    return FBNLFilterResult(data, resolution, window, window_var, p,
                            data_filtered, sf, sb, f, b, xf, xb, step_mass,
                            step_size, noise, noise_mean, step_mass_SNR_mean,
                            step_mass_SNR_median, step_mass_STD, outls)


def _filter_fbnl_capped(data, resolution, window, window_var=None, p=None):
    window_var = window_var or window
    loss = window + window_var - 1
    inspect = int(np.ceil(window / 2))
    _data = cap_data(data, loss, inspect)

    (data, resolution, window, window_var, p, data_filtered, sf, sb, f, b, xf,
     xb) = _filter_fbnl(_data, resolution, window, window_var, p)

    data = data[loss:-loss]
    data_filtered = data_filtered[loss:-loss]
    sf = sf[loss:-loss]
    sb = sb[loss:-loss]
    f = f[loss:-loss]
    b = b[loss:-loss]
    xf = xf[loss:-loss]
    xb = xb[loss:-loss]

    return (data, resolution, window, window_var, p, data_filtered, sf, sb, f,
            b, xf, xb)


def _filter_fbnl(data, resolution, window, window_var=None, p=None):
    N = len(data)
    window_var = window_var or window
    p = p if p is not None else 1

    # init results to nan
    # Filtered data
    data_filtered = np.full(N, np.nan)

    # Variances measured over window_var Points
    sf = np.full(N, np.nan)
    sb = np.full(N, np.nan)

    xb = np.full(N, np.nan)
    xf = np.full(N, np.nan)

    x = moving_mean(data, window)

    # save shifted x to xb and xf
    xb[window:N - window] = x[window + 1:]
    xf[window:N - window] = x[:N - 2 * window]

    # calculate first elements of sf and sb
    loss = window + window_var - 1
    sf[loss] = np.sum((data[window:loss + 1] - xf[window:loss + 1])**2)
    sb[loss] = np.sum((data[loss:loss + window_var]
                       - xb[loss:loss + window_var])**2)

    # calculate other elements
    # use cython to speed up the calculation
    start = loss + 1
    stop = N - loss
    _iterative_variance(data, sf, sb, xf, xb, start, stop, window_var)

    # Make Variances real Variances by dividing by window_var
    sf = sf / window_var
    sb = sb / window_var

    # calculate weights from Variances
    f = sf**(-p)
    b = sb**(-p)

    # norm f(i,k) and b(i,k)
    total = f + b
    f = f / total
    b = b / total

    ratio_fb = np.sum(f) / np.sum(b)
    if ratio_fb > 1.15 or ratio_fb < 1 / 1.15:
        print('WARNING: ratio f/b is critical: f/b = {:.3f}'.format(ratio_fb))

    # Filtered result is the weighted sum over forward and backward estimations
    data_filtered = f * xf + b * xb

    return (data, resolution, window, window_var, p, data_filtered, sf, sb, f,
            b, xf, xb)


def find_steps(step_mass, y_c, max_step_width=None, min_step_spacing=None,
               H=None, L=None, switch_accept=True):
    """
    Find steps by comparing variances of forward and backward estimation of
    datapoints.

    Implementation of D.A. Smiths Algorithm [1]. Search for maximum is
    performed differently than discribed in this paper:
    Beginning from the left side, all datapoints with y above threshold `y_c`
    are 'collected' bins (segments). Each segment has a start and a stop
    position. A segment is filled, and therefore a new one opened, if either
    the sign of y changed or the length (stop - start) of the segment is more
    than `max_step_width`.

    Parameters
    ----------
    step_mass = step_size / noise
            step_size = xb - xf (see [1])
            noise = np.sqrt(f * sf + b * sb) (see [1])
    y_c : float
        Threshold of step detection. Set to about 1.5 to be sure to find only
        steps, not noise. step_size abs(xb - xf) is at least y_c times the
        noise in the data. In practice, choose: y_c = 2/3 * D/STD = 2/3 *
        min_step_size / noise.
            /\
           /  \           /\
        __/____\_________/__\___________
         /      \ |y_c  /    \
        /        \|____/      \/\/\_/\__
    max_step_width : int, optional
        Maximum width a step is expected to have. If a detected step has a
        greater width, it is accepted as one. Defaults to `min_step_spacing`.
            /\
           /  \           /\
          /____\         /  \
         /  M   \       /    \ (M = max_step_width)
        /        \     /      \
    min_step_spacing : int, optional
        Minimum distance the center of a step needs to be separated from the
        previous one, to be accepted as a separate one. If distance is too
        small, steps are fused.
            /\
           /  \           /\
          /  __\____E____/_ \  (E = min_step_spacing)
         /      \       /    \
        /        \     /      \
    H : int, optional
        Minimum width, a step needs to have, to be accepted as one. If width is
        too small, step is neglected. H is checked after min_step_spacing.
            /\
           /  \           /\
          /____\         /  \
         /  H   \       /    \
        /        \     /      \
    L : int, optional
        Minimum distance the edge of a step needs to be separated from the
        previous one, at the height were the threshold is set (y_c), to be
        accepted as a separate one. If distance is too small, steps are fused.
        L is checked after H.
            /\
           /  \           /\
          /    \____L____/  \
         /      \       /    \
        /        \     /      \
    switch_accept : bool, optional
        Accept each other following steps, even if the distance `E` would not
        be sufficient, if their direction is opposite from each other.

    Returns
    -------
    indices : 1D numpy.ndarray of type int
        Positions of the steps.
    direction : 1D numpy.ndarray of type bool
        Direction of the steps. True is a positive step, False is a negative
        one.
    step_bounds : 2D numpy.ndarray of type int
        The detected start and stop indices of the steps.
    number : int
        The number of the detected steps.
    plateaus : 2D numpy.ndarray of type int
        The start and stop indices of the plateaus between the steps.
    p_centers : 1D numpy.ndarray of type int
        The indices of the centers of the plateaus.

    Notes
    -----
    For distributions of y which are not steadily growing from left to right
    at some point, total bin widths of at least max_step_width are ensured.
    However be aware that big steps can swallow small steps which are up to
    min_step_size left of it. This can happen only if the y of the big step is
    large compared to the small step, which means, that the center of mass is
    very close to the position of the big step.
    Searching from right to left would preserve a small step on the left, but
    would swallow a small step in distance of about min_step_size on the right.
    If you want to ensure a maximum bin spacing at all costs, search for steps
    in both directions and combine steps you found twice.

    References
    ----------
    [1] Smith, D.A. 1998 "A Quantitative Method for the Detection of Edges in
    Noisy Time-Series." Phil. Trans. R. Soc. Lond. B 353, 1969-1981
    """
    step_mass = step_mass.copy()
    step_mass[np.isnan(step_mass)] = 0.0

    min_step_spacing = min_step_spacing or 1
    max_step_width = max_step_width or min_step_spacing
    H = H or 1
    L = L or 1

    pos_step_bounds = get_contiguous_segments(step_mass > y_c,
                                              min_length_high=H,
                                              min_distance_center=1,
                                              min_length_low=L)
    neg_step_bounds = get_contiguous_segments(step_mass < -y_c,
                                              min_length_high=H,
                                              min_distance_center=1,
                                              min_length_low=L)

    # sort step_bounds
    step_bounds = np.r_[pos_step_bounds, neg_step_bounds]
    sort_idx = step_bounds[:, 0].argsort()
    step_bounds = step_bounds[sort_idx]
    # create array with directions (up/down, pos/neg) of steps
    pos = np.ones(len(pos_step_bounds), dtype=bool)
    neg = np.zeros(len(neg_step_bounds), dtype=bool)
    direction = np.r_[pos, neg]
    direction = direction[sort_idx]

    # check center distance and direction of the plateaus
    # use cython to speed up the calculation
    step_bounds, direction = _delete_close_center(step_bounds, direction,
                                                  max_step_width,
                                                  min_step_spacing,
                                                  switch_accept=switch_accept,
                                                  copy=False)

    # determine indices of center of step_bounds and maximal step_masses
    indices = []
    for start, stop in step_bounds:
        # calculate the center of mass
        idx = np.arange(start, stop)
        weights = step_mass[start:stop]
        center_of_mass = int(np.round(np.average(idx, weights=weights)))
        # due to noise the center of mass can be outside the segment indices.
        # This usualy happens, if the SNR is too low. Make sure, the indices
        # stay within the plateua limits:
        center_of_mass = max(start, center_of_mass)
        center_of_mass = min(stop - 1, center_of_mass)
        indices.append(center_of_mass)
    indices = np.array(indices)

    # For convenience calculate start/stop indices of plateaus between the
    # steps
    datapoints = len(step_mass)
    plateau_indices = np.r_[0, indices, datapoints]
    plateaus = idx_to_idx_segments(plateau_indices)

    p_centers = (plateaus[:, 0] + (plateaus[:, 1] - plateaus[:, 0]) / 2)
    p_centers = np.round(p_centers).astype(int)
    p_centers[p_centers > datapoints - 1] = datapoints - 1

    # For convenience calculate number of steps
    number = len(indices)

    return indices, direction, step_bounds, number, plateaus, p_centers


def analyse_steps(indices, plateaus, data):
    """
    Calculate dwell times, plateau heights and step sizes. A plateau is the
    range between two steps. Plateau heights are mean values of 'data' of a
    plateau. Step sizes are the difference of two each other following
    plateaus.

    Indexing looks as follows:

        dwell_points[i-1]     __________________________|
        plateau_heights[i]    |
    __________________________| step_sizes[i]

                     indices[i]             indices[i + 1]

    Mind the following:
    -------------------
    - There is one plateau height more than steps
    - dwell times of outer steps cannot be calculated as boundarys are
      artificially created

    The following scheme illustrates this:
                                                              n.A.
                                                              5
                                                   3     ___________
                                                   4     |
                                       2     ____________| 4
                                       3     |
                           1     ____________| 3
                           2     |
               0     ____________| 2
               1     |
    n.A. ____________| 1
    0    |
    _____| 0
        0           1           2           3           4

    Parameters
    ----------
    indices : 1D numpy.ndarray of type int
    plateaus : 2D numpy.ndarray of type int
    data : 1D numpy.ndarray of type float

    Returns
    -------
    step_sizes : 1D numpy.ndarray of type float
    plateau_heights : 1D numpy.ndarray of type float
    dwell_points : 1D numpy.ndarray of type int
    """
    # dwell times: Do not take outer values, as they are arbitrary
    dwell_points = indices[1:] - indices[:-1]

    # Calculate absolute plateau heights, by averaging over data from one step
    # to the next one
    plateau_heights = np.zeros(len(plateaus))
    # calculate plateau heights of plateaus starting from plateau before the
    # first step till plateau after the last step

    # use cython to speed up the calculation
    with np.errstate(invalid='ignore'):
        _calculate_plateau_heights(data, plateaus, plateau_heights)

    # Relative step_sizes: differences of absolute step_sizes
    step_sizes = plateau_heights[1:] - plateau_heights[:-1]
    return step_sizes, plateau_heights, dwell_points


def get_min_step_sizes(indices, y_c, fbnl_filter, step_size_threshold=None):
    """
    Get minimum step_sizes

    Parameters
    ----------
    steps : Steps
    y_c : float
    fbnl_filter : FBNLFilterBankResult
    step_size_threshold : str or float
        The deletion procedure searches for steps, that are not as high as a
        threshold value. The threshold value can be set manually or
        automatically, in the latter case either to a constant value or
        dynamically, adapted to the noise around a step. The following values
        are then chosen:
            float: threshold-value = float
                NOT RECOMMENDED for serious analysis
            'static': threshold = y_c / <noise>_data
                Choose static if you want a constant threshold for step sizes
                for all steps.
            'adapt': threshold(step) = y_c / <noise>_data(step +-
                min_step_spacing)
                BEST CHOICE: Every found step has the same reliability, given
                by y_c
        Defaults to 'adapt'.

    Returns
    -------
    min_step_sizes : 1D numpy.ndarray of type float
    step_size_threshold : str or float
    """
    step_size_threshold = step_size_threshold or 'adapt'
    number = len(indices)
    if step_size_threshold == 'adapt':
        window = fbnl_filter.window
        window_var = fbnl_filter.window_var
        noise = fbnl_filter.noise
        start_min = window + window_var - 1
        stop_max = len(noise) - start_min

        # adapt to mean noise at step
        noise_mean = np.empty(number)
        for i, step in enumerate(indices):
            start = max(step - (window + window_var), start_min)
            stop = min(step + (window + window_var), stop_max)
            noise_mean[i] = noise[start:stop].mean()

        min_step_sizes = y_c * noise_mean
    elif step_size_threshold == 'static':
        # set to noise_mean * step_size_threshold / noise
        min_step_sizes = np.ones(number) * y_c \
            * fbnl_filter.noise_mean
    else:
        min_step_sizes = np.ones(number) * step_size_threshold
    return min_step_sizes, step_size_threshold


def delete_small_steps(steps, min_step_sizes):
    """
    Delete all steps with sizes smaller than the threshold min_step_sizes.
    Sizes, means and dwell times of steps around are corrected accordingly,
    after a deletion.

    IMPORTANT REMARK:
    The search procedure is performed from the left to the right. There is no
    reason not to do it from the right to the left. Both procedures do
    however yield different results, as the deletion of a step changes the
    step_sizes of the neighbouring steps. Differences in left to right -
    right to left deletion do usually occur in areas of constant slope.
    Fortunately, the quality of the steps in these regimes should be quite
    low anyway.

    Parameters
    ----------
    steps : Steps
    min_step_sizes : 1D numpy.ndarray of type float

    Returns
    -------
    Steps : namedtuple
    """
    # Create a linked list from the plateaus and their corresponding steps.
    # This implementation avoids the (for long lists) expensive pop() function
    # of lists.
    class Plateau(object):
        def __init__(self, start, stop, height,
                     step_bounds=None, step_minsize=None,
                     prev=None, next=None):
            self.prev = prev
            self.next = next
            if prev is not None:
                prev.next = self
            if next is not None:
                next.prev = self
            self.start = start
            self.stop = stop
            self.height = height
            self.step_bounds = step_bounds
            self.step_minsize = step_minsize
    first = Plateau(steps.plateaus[0][0], steps.plateaus[0][1],
                    steps.plateau_heights[0],
                    steps.bounds[0], min_step_sizes[0])
    current = first
    for v in zip(steps.plateaus[1:], steps.plateau_heights[1:],
                 steps.bounds[1:], min_step_sizes[1:]):
        current = Plateau(v[0][0], v[0][1], v[1], v[2], v[3], prev=current)
    current = Plateau(steps.plateaus[-1][0], steps.plateaus[-1][1],
                      steps.plateau_heights[-1], prev=current)

    # Iteratively check minimum step size and fuse/delete plateaus and steps
    # and (re)check previous and following step(s) for mininum
    # step_size, which could have been changed due to the deletion.
    current = first
    num_plateaus = steps.number + 1
    while current.next is not None:
        if np.abs(current.next.height - current.height) < current.step_minsize:
            # Calculate new plateau height
            lc = current.stop - current.start
            ln = current.next.stop - current.next.start
            height_new = (current.height * lc + current.next.height * ln) \
                         / (lc + ln)

            # Correct plateaus start index and height
            current.next.start = current.start
            current.next.height = height_new

            # Remove current plateau and its step and replace
            # references to current with the previous and next plateau
            num_plateaus -= 1
            current.next.prev = current.prev
            if current.prev is not None:
                # Current plateau is not the first
                # Replace reference from previous to current with next one
                # Step size of previous plateau was changed. Recheck the
                # step size -> set current plateau to be the previous one
                current.prev.next = current.next
                current = current.prev
            else:
                # Current plateau is the first
                # Set the next to be the first one and check the
                # next plateau
                first = current.next
                current = current.next
        else:
            # Nothing deleted, check next plateau
            current = current.next

    # Create numpy arrays
    plateaus = np.empty((num_plateaus, 2), dtype=int)
    p_heights = np.empty(num_plateaus)
    step_bounds = np.empty((num_plateaus - 1, 2), dtype=int)

    current = first
    i = 0
    while current is not None:
        plateaus[i][0] = current.start
        plateaus[i][1] = current.stop
        p_heights[i] = current.height
        if current.next is not None:
            step_bounds[i] = current.step_bounds
        current = current.next
        i += 1

    indices = plateaus.flatten()[1:-1:2]
    step_sizes = p_heights[1:] - p_heights[:-1]
    direction = step_sizes > 0
    number = num_plateaus - 1
    p_centers = np.round(
            plateaus[:,0] + (plateaus[:,1] - plateaus[:,0]) / 2
        ).astype(int)
    dwell_points = indices[1:] - indices[:-1]

    return Steps(indices, direction, step_bounds, number, plateaus,
                 p_centers, step_sizes, p_heights, dwell_points)


def get_step_qualities(steps, fbnl_filter):
    """
    Calculate standard deviation and mean noise of a step.

    Mean noise is the weighted noise noise**2 = sf * f + sb * b. It should be a
    function of the tweezer-force, not peaking at steps.
    Standard deviation is the real standard deviation, which includes steps.
    Averaged over a step-plateau, both values should be equal for perfect
    steps. If a step-plateau contains another step, the s.d. is higher than
    the noise. The ratio noise/s.d. is a qualifier for a step, as good
    step-plateas do not include steps.

    Parameters
    ----------
    steps : Steps
    fbnl_filter : FBNLFilterResult or FBNLFilterBankResult

    Returns
    -------
    StepQuality : namedtuple
    """
    indices = steps.indices  # indices of steps
    plateaus = steps.plateaus  # start and stop indices of plateaus
    p_heights = steps.plateau_heights  # mean of height between to steps

    data = fbnl_filter.data
    noise = fbnl_filter.noise
    window = fbnl_filter.window
    window_var = fbnl_filter.window_var

    start_min = window + window_var - 1
    stop_max = len(noise) - start_min

    step_sd = np.zeros_like(indices, dtype=float)
    step_noise = np.zeros_like(indices, dtype=float)

    # go through all (left and right the step adjoining) plateaus
    for i, (p_l, p_r, ph_l, ph_r) in enumerate(zip(
            plateaus[:-1], plateaus[1:], p_heights[:-1], p_heights[1:])):

        # get the center of the plateaus and make sure the indices are at least
        # a lenght of size window apart from the steps themself and window +
        # window_var from the ends of the data (filter -> nan -> see data
        # capping)
        start_l = int(np.round(p_l[0] + (p_l[1] - p_l[0]) / 2))
        start_l = max(start_l, p_l[0] + window)
        start_l = max(start_l, start_min)
        stop_l = p_l[1] - window
        stop_l = max(stop_l, start_l)

        start_r = p_r[0] + window
        stop_r = int(np.round(p_r[1] - (p_r[1] - p_r[0]) / 2))
        stop_r = min(stop_r, p_r[1] - window)
        stop_r = min(stop_r, stop_max)
        start_r = min(start_r, stop_r)

        data_l = data[start_l:stop_l]
        data_r = data[start_r:stop_r]

        # Total number of points from both of the halfs of the plateaus minus
        # the window size
        total_number = stop_r - start_r + stop_l - start_l

        with np.errstate(divide='ignore'):
            # s.d. from step-plateaus to data
            p_h = np.r_[data_l - ph_l, data_r - ph_r]
            step_sd[i] = np.sqrt(np.sum(p_h**2) / (total_number - 1))

        # s.d. from mean to data, averaged over step
        total_noise = np.r_[noise[start_l:stop_l], noise[start_r:stop_r]]
        if len(total_noise) > 0:
            step_noise[i] = np.mean(total_noise)
        else:
            step_noise[i] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        step_noise_over_sd = step_noise / step_sd

    return StepQuality(step_sd, step_noise, step_noise_over_sd)


def find_and_analyse_steps(fbnl_filter, expected_min_step_size=None,
                           expected_min_dwell_t=None, switch_accept=True,
                           step_size_threshold=None, use_mean=False,
                           verbose=True):
    """
    Find steps. See function `filter_find_analyse_steps()` for an explanation.

    Parameters
    ----------
    fbnl_filter : FBNLFilterResult or FBNLFilterBankResult
    expected_min_step_size : float, optional
        Defaults to 1.
    expected_min_dwell_t : float, optional
        Minimum step spacing (dwell time) that can be found.
    step_size_threshold : str or float, optional
        Minimum step_size allowed. Can be set to a value to 'static' or to
        'adapt'.
        float: threshold-value=number
            NOT RECOMMENDED for serious analysis
        'static': threshold = y_c / <noise>_data
            Choose static if you want a constant threshold for step sizes for
            all steps.
        'adapt': threshold(step) = y_c / <noise>_data(step +- min_step_spacing)
            BEST CHOICE: Every found step has the same reliability, given by
            y_c
    use_mean : bool, optional
        If fbnl_filter is a `FBNLFilterBankResult`, one can set the use of the
        `step_mass_mean` to detect peaks, instead of the usually used
        `step_mass`.
    verbose : bool, optional
        Be verbose.

    Returns
    -------
    StepFinderResult : namedtuple
    """
    # Determine the optimal threshold y_c
    # In the paper of Smith: 1 / sqrt(window) << y_c < |D| / STD =
    # step_size[k] / noise[k], where k is the position of an edge.
    # In practice, choose: y_c = 2/3 * D/STD = 2/3 * min_step_size/noise
    # Probably another method could be based on the interquartile range with a
    # factor of 3.0:
    # y_c = iqr_outlier_threshold(step_mass[~np.isnan(step_mass)],
    #                             iqr_factor=3.0)
    if expected_min_step_size is None:
        # Per default assume an SNR of 1.0
        y_c = 2/3
    else:
        y_c = 2/3 * expected_min_step_size / fbnl_filter.noise_mean

    if y_c <= 2 * 1 / np.sqrt(fbnl_filter.window) and verbose:
        print('WARNING: The threshold y_c needs to satisfy'
              ' 1 / sqrt(window) << y_c < abs(D) / noise!\n'
              'Either increase the window size for filtering, or increase '
              'the expected_min_step_size, which effectively will result in '
              'only greater steps beeing detected.')
    tmp = ('Absolute noise of the signal (steps excluded): {:.3f}\n'
           'Threshold y_c relative to the noise, for step detection: {:.3f}\n'
           '  y_c = 2/3 * `expected_min_step_size` / noise')
    if verbose:
        print(tmp.format(fbnl_filter.noise_mean, y_c))

    # Minimum step spacing (dwell time) / min distance of two successive steps
    # to be apart from each other. Positive steps following positive and
    # negative following negative steps need to be at least min_step_spacing
    # points apart. Greater values reduce TP and FP rate and let each other
    # following steps fuse. Small values increase TP and FP rate.
    # From Smith1998: E should be < window * 2. A good assumption as a value is
    # window.
    min_step_spacing = fbnl_filter.window
    if expected_min_dwell_t is not None:
        expected_min_step_spacing = int(np.round(expected_min_dwell_t
                                                 * fbnl_filter.resolution))
        min_step_spacing = max(expected_min_step_spacing, fbnl_filter.window)
        if expected_min_step_spacing < fbnl_filter.window and verbose:
            tmp = ('\nWARNING: The expected minimal dwell time ({:.4f} s, # '
                   '{}) is smaller than the filter window time ({:.4f} s, # '
                   '{})!'
                   '\nTherefore, the minimal dwell time is set to the window '
                   'time.'
                   '\nTo be able to detect steps following each other and '
                   'beeing separated only by a smaller time period as the '
                   'window time, you have to decrease the filter window time '
                   'with the parameter \'filter_time\'. But, bear in mind '
                   'that this will effectively reduce the SNR ratio which '
                   'will reduce the distinguishability of steps from noise.'
                   '\nThe cleanest solution to this problem would be to '
                   'collect the data with a higher resolution.\n')
            pars = (expected_min_dwell_t, expected_min_step_spacing,
                    fbnl_filter.window / fbnl_filter.resolution,
                    fbnl_filter.window)
            print(tmp.format(*pars))
    elif verbose:
        print('Autoselection of `min_dwell_time` was enabled.')
    min_dwell_time = min_step_spacing / fbnl_filter.resolution
    tmp = ('Selected a `min_dwell_time` (`min_step_spacing`) of {:.4f} s (l: '
           '{}) for step detection.')
    if verbose:
        print(tmp.format(min_dwell_time, min_step_spacing))

    if use_mean:
        step_mass = fbnl_filter.step_mass_mean
    else:
        step_mass = fbnl_filter.step_mass

    # Find steps
    indices, direction, step_bounds, number, plateaus, p_centers \
        = find_steps(step_mass, y_c, min_step_spacing=min_step_spacing,
                     switch_accept=switch_accept)
    if verbose:
        print('Total number of steps found: {}'.format(number))

    if number > 0:
        # Analyse steps and save step-information, before small steps will be
        # deleted
        step_sizes, plateau_heights, dwell_points \
            = analyse_steps(indices, plateaus, fbnl_filter.data)
        steps_pre = Steps(indices, direction, step_bounds, number, plateaus,
                          p_centers, step_sizes, plateau_heights, dwell_points)
        # Calculate step_size treshold (depending on step_size_threshold =
        # 'adaptive', 'static' or a float)
        min_step_sizes_pre, step_size_threshold \
            = get_min_step_sizes(indices, y_c, fbnl_filter,
                                 step_size_threshold)
        step_distribution_pre = get_step_qualities(steps_pre, fbnl_filter)

        # Delete small steps and calculate new qualities
        steps = delete_small_steps(steps_pre, min_step_sizes_pre)
        num_deleted = steps_pre.number - steps.number
        if num_deleted > 0 and verbose:
            print('Number of steps deleted: {}'.format(num_deleted))
            print('New total number of steps: {}'.format(steps.number))
        min_step_sizes, step_size_threshold \
            = get_min_step_sizes(steps.indices, y_c, fbnl_filter,
                                 step_size_threshold)
        step_distribution = get_step_qualities(steps, fbnl_filter)

        # Inform if step sizes of all steps are smaller than minimum allowed
        # step spacings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dwell_points_pre_mean = steps_pre.dwell_points.mean()
        if dwell_points_pre_mean < min_step_spacing and verbose:
            print('INFORMATION: Mean dwell time of all steps (including '
                  'deleted) is lower than minimum step spacing.')

        # Warn if step sizes of steps - deleted steps excluded - are
        # smaller than minimum allowed step spacing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dwell_points_mean = steps.dwell_points.mean()
        if dwell_points_mean < min_step_spacing and verbose:
            print('WARNING: Mean dwell time is lower than minimum step',
                  'spacing.',
                  'Try the following:',
                  '\n    a) reduce minimum step spacing, if you expect about',
                  'the same number of forward and backward steps',
                  '\n    b) increase threshold y_c, if you expect most steps',
                  'in one direction.')

        result = StepFinderResult(fbnl_filter, expected_min_step_size, y_c,
                                  expected_min_dwell_t, min_dwell_time,
                                  switch_accept, step_size_threshold,
                                  steps_pre, steps, min_step_sizes_pre,
                                  min_step_sizes, step_distribution_pre,
                                  step_distribution)
    else:
        # no steps, create one big plateau
        plateaus = np.array([0, len(fbnl_filter.data)], dtype=int)
        plateau_heights = np.array(np.sum(fbnl_filter.data))
        steps = Steps(np.empty(0, dtype=int), np.empty(0, dtype=bool),
                      np.empty(shape=(0, 2)), 0, plateaus, p_centers,
                      np.empty(0), plateau_heights, np.empty(0))
        min_step_sizes = np.array([])
        step_distribution = StepQuality(np.empty(0), np.empty(0), np.empty(0))
        result = StepFinderResult(fbnl_filter, expected_min_step_size, y_c,
                                  expected_min_dwell_t, min_dwell_time,
                                  switch_accept, step_size_threshold,
                                  steps, steps, min_step_sizes, min_step_sizes,
                                  step_distribution, step_distribution)
    return result


def filter_find_analyse_steps(data, resolution, filter_time=None,
                              filter_min_t=None, filter_max_t=None,
                              filter_number=None, edginess=None,
                              expected_min_step_size=None,
                              expected_min_dwell_t=None,
                              step_size_threshold=None, cap_data=True,
                              verbose=True, plot=True):
    """
    Fiter data, find steps and analyse the steps. See notes for further
    explanation.

    Parameters
    ----------
    data : numpy.ndarray of type float
        data to be filtered.
    resolution : float
        The resolution of the data in Hz.
    filter_time : float, optional
        The filter time used to filter the data for the edge detector. Set it
        to overwrite the automatically detected optimal filter time. Otherwise,
        the optimal `filter_time` is automatically detected  based on the STD
        of the step_mass from different tested times (see parameters
        `filter_min_t`, `filter_max_t` and `filter_number`)
        You only should set the time manually, if you know that it performs
        well with the given data. A good rule of thumb is to set the
        filter_time to 1/2 the time of the expected minimum dwell time. Higher
        values lead to better averaging, but can miss steps. REMARK: Shorter
        spaced steps can still be found if step direction switches. See also
        the parameter `expected_min_dwell_t`.
    filter_min_t : float, optional
        The shortest time of the window in s with which the datapoints are
        filtered.
    filter_max_t : float, optional
        The longest time of the window in s with which the datapoints are
        filtered. You need to also set `filter_number` to make this parameter
        effective. Defaults to `filter_min_t`.
    filter_number : int, optional
        Number of filters used to determine the optimal filter window length.
        Defaults to 1.
    edginess : float, optional
        The edginess (nonlinearity factor p in Chung1991) of the filtered data.
        The edginess only affects the filtered data, but not the detection of
        the steps itself. An edginess of 0 will smooth out edges, like a simple
        moving mean filter. A higher edginess will preserve edges. A too high
        edginess will cause ripples due to amplification of edges of noise.
        From the paper:
        "Thus, for a faithful preservation of fast signal transient details,
        such as the abrupt changes observed during a single channel
        conductance, a large value of the weighting factor p [...] is
        effective, whereas slow signal variations, e.g., during exponential
        decay back to the baseline, can best be extracted with a smaller p
        value." Chung1991 used values between 1 and 100.
    expected_min_step_size : float, optional
        Low values yield wrong steps. High values yield no steps. Expected
        minimum step size is converted into the threshold value y_c = 2/3 *
        `expected_min_step_size` / `noise` (see [2]). Defaults to 1.
    expected_min_dwell_t : float, optional
        The minimal dwell time (`step_spacing`), one expects to be existent in
        the data. Used to check the filtered signal for steps. Higher values
        lead to deletion of smaller steps, and each other closly following
        ones, lower values lead to many FPs. Recommended: Set it to None and
        let the algorithm automatically set the min_dwell_t, based on the
        `filter_time`.
    step_size_threshold : str or float, optional
        Minimum step size threshold for the deletion procedure. Following
        values are allowed:
        float: Take the value as a threshold for all steps. Use 0 to turn step
            deletion off: threshold-value = number
            NOT RECOMMENDED for serious analysis
        'static': threshold = y_c / <noise>_data
            Take the same threshold for all steps. Choose static if you want a
            constant threshold for step sizes for all steps.
        'adapt':  Calculate threshold for every step corresponding to y_c and
            noise. threshold(step) = y_c / <noise>_data(step +-
            min_step_spacing). BEST CHOICE: Every found step has the same
            reliability, given by y_c.
    cap_data : bool, optional
        Cap the data to protect the ends from be "eaten up" by the filtering
        process. Defaults to True.
    verbose : bool, optional
        Be verbose. Defaults to True.
    plot : bool, optional
        Plot an overview of the process to find the optimum `filter_time`. And
        additionally return the figure. Defaults to True.

    Returns
    -------
    StepFinderResult : namedtuple
    fig : matplotlib.pyplot.figure
        Only returned, if parameter `plot` is set to True.

    Notes
    -----
    Step finder's work-steps
    ========================
    The step_finder has four main work-steps:

    0) Find the optimum `filter_time`, by repeating work-steps 1) - 3) with
    different filter_times and analyse the results.

    1) Filter the data with the forward-backward non-linear filter proposed by
    Chung & Kennedy (1991, J. Neuroscience) [1].

    2) After that it searches for steps by comparing the difference of the
    forward and the backward filter to the noise at the current point. The
    expected_min_step_size (-> threshold y_c) and the expected_min_dwell_t
    (-> min_step_spacing, E in paper) decide wether there is a step or not. See
    Smith (1998, Phil. Trans. R. Soc. Lond. B) [2].

    3) In the last step, steps are being deleted when they are smaller than a
    minimum allowed step size (see parameter `step_size_threshold` and function
    `get_min_step_sizes()`). This is important, because statistical
    fluctuations can locally look like large steps, even if the plateau behind
    such a step is about the same height as the plateau before.

    Besides these main working steps, information about dwell time and step
    size is being calculated and the significance of a step is determined, e.g.
    if there is a constant slope in the data, the algorithm will find large
    steps.

    Problems:
    =========
    Classification of steps by the parameter m:
    Be aware, that there is always a systematic error coming from the feedback
    of the tweezers or other experimental errors, which the algorithm does not
    consider. The Classification Parameter m, especially the threshold y_c is
    therefore not definitely comparable for all data you analyse.

    Application:
    ============
    See [3].

    Remarks:
    ========
    Regarding the parameters `filter_tmin_t`, `filter_tmax_t` and
    `filter_time`:
    High values smoothen the noise, but probably also diminish the signal. Low
    values leave the signals intact, but probably do not sufficiently reduce
    the noise. Medium values smoothen the noise and leave the signals intact
    and thereby increase the SNR of the data.

    From [1]:
    "The number of predictors [here filter_number] and their lengths can be
    varied. Three short pairs of the predictors used to extract exponentially
    decaying signals imbedded in the noise [...] preserved the original
    features of the signal, but the background noise was not effectively
    suppressed [...]. As the predictors of longer lengths were added [...], a
    further reduction in the noise was achieved at the expense of distorting
    the original signal. Although there are a large number of possible
    predictor combinations that can be used for processing a segment of data,
    the choice of the bank of predictors in practice is straightforward.
    Various choices of lengths of predictors should be based on the expected
    durations of signal features. If, for example, we anticipate signals of
    width 10 samples to be present in the data, then there should be at least
    one predictor whose length is less than 10 points [here you would set tmin
    to <= 10]. Naturally, if longer signal features are present, then
    correspondingly longer window predictors should be included in the bank.

    Regarding the parameter `expected_min_step_size`:
    For perfectly aligned Tweezers m/sqrt(min_step_spacing / 2) is the ttest
    for having NO step. As there are always systematic errors, assume t = m, so
    setting y_c to 2 makes algorithm find steps with at least 95 % reliability.


    References
    ----------
    [1] Chung, S.H. & Kennedy, R.A. 1991 "Forward-backward nonlinear filtering
    technique for extracting small biological signals from noise." J. Neurosci.
    Meth. 40, 71-86
    [2] Smith, D.A. 1998 "A Quantitative Method for the Detection of Edges in
    Noisy Time-Series." Phil. Trans. R. Soc. Lond. B 353, 1969-1981
    [3] N.J. Carter & R.A. "Cross 2005 Mechanics of the kinesin step." Nature
    435, 308-312
    """
    filter_max_t = filter_max_t or filter_time
    try:
        filter_min_t = filter_min_t or min(filter_time, filter_max_t)
    except:
        print('You have to give at least one `filter_time`')
        return
    windows = log_spaced_time_windows(filter_min_t, filter_max_t, resolution,
                                      filter_number)

    if filter_time is not None:
        window_edge = max(int(np.round(filter_time * resolution)), 1)
        windows = np.unique(np.r_[window_edge, windows])
        i_window = np.where(windows == window_edge)[0][0]

    # Set window size for variance to the same size as the windows
    windows_var = windows

    steps_number_pre = []
    steps_number = []
    aSNRs = []
    mSNRs = []
    STDs = []

    # mean of data_filtered of several banks of predictors (windows)
    data_filtered_mean = np.zeros_like(data)
    step_size_mean = np.zeros_like(data)
    noise_mean = np.zeros_like(data)

    if verbose:
        print('\nPre-processing and analysing filter windows:',
             '\n----------------------------------------------------------------',
             '\nnumber  time(s) #length  aSNR    mSNR   STD  #outls  #pre #steps',
             '\n----------------------------------------------------------------')

    # Filter the data, calculate step_masses of all differently filtered data
    # and find the steps
    for i, (window, window_var) in enumerate(zip(windows, windows_var)):
        # window-length for calculating variances to window-length of
        # averaging
        fbnl_filter = filter_fbnl(data, resolution, window=window,
                                  window_var=window_var, p=edginess,
                                  cap_data=cap_data)
        step_finder_result \
            = find_and_analyse_steps(fbnl_filter, expected_min_step_size,
                                     expected_min_dwell_t,
                                     step_size_threshold=step_size_threshold,
                                     verbose=False)

        steps_number_pre.append(step_finder_result.steps_pre.number)
        steps_number.append(step_finder_result.steps.number)

        aSNR = fbnl_filter.aSNR
        mSNR = fbnl_filter.mSNR
        STD = fbnl_filter.STD
        outls = np.sum(fbnl_filter.outls)

        aSNRs.append(aSNR)
        mSNRs.append(mSNR)
        STDs.append(STD)

        # mean of data_filtered of several banks of predictors
        data_filtered_mean += fbnl_filter.data_filtered / len(windows)
        step_size_mean += fbnl_filter.step_size / len(windows)
        noise_mean += fbnl_filter.noise / len(windows)

        if verbose:
            tmp = '{:5d} {:8.4f} {:6d} {:7.3f} {:7.3f} {:6.3f} {:6d} {:6d} {:5d}'
            pars = (i, window / resolution, window, aSNR, mSNR, STD,
                    outls, step_finder_result.steps_pre.number,
                    step_finder_result.steps.number)
            print(tmp.format(*pars))

    if verbose:
        print('-----------------------------------------------------------------')

    step_mass_mean = step_size_mean / noise_mean

    # Automatically select a conservatively chosen window size
    # which corresponds to 2/3 times the window size at the STD minimum.
    STDs = np.array(STDs)
    if filter_time is None:
        idx = np.argwhere(STDs.min() == STDs)
        window = int(np.round(np.mean(windows[idx]) * 4/5))
    else:
        window = windows[i_window]
    # Set the window for the variance weight to the window size
    window_var = window

    if filter_time is None and verbose:
        print('Autoselection of `filter_time` was enabled.'
              '\n  Please, verify your results and, if necessary, adjust this '
              'parameter accordingly.')
    if verbose:
        tmp = ('Selected a `filter_time` (window) of {:.1f} ms (l: {}) '
               'for the calculation of the step_mass (= step_size/noise).')
        print(tmp.format(window / resolution * 1000, window))

    fbnl_filter = filter_fbnl(data, resolution, window=window,
                              window_var=window_var, p=edginess,
                              cap_data=cap_data)

    fbnl_filter = FBNLFilterBankResult(windows, windows_var,
                                       data_filtered_mean, step_mass_mean,
                                       step_size_mean, noise_mean,
                                       *fbnl_filter)

    step_finder_result \
        = find_and_analyse_steps(fbnl_filter, expected_min_step_size,
                                 expected_min_dwell_t,
                                 step_size_threshold=step_size_threshold,
                                 verbose=verbose)
    if plot:
        stepfinder_characteristics = {
            'windows': windows,
            'steps_number_pre': np.array(steps_number_pre),
            'steps_number': np.array(steps_number),
            'aSNRs': np.array(aSNRs),
            'mSNRs': np.array(mSNRs),
            'STDs': STDs,
            'resolution': resolution,
            'window_time': window / resolution
        }
        fig, axes = plot_stepfinder_characteristics(stepfinder_characteristics)
        return step_finder_result, fig
    return step_finder_result


def plot_stepfinder_characteristics(stepfinder_characteristics, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
    else:
        fig = axes[0].get_figure()

    windows = stepfinder_characteristics['windows']
    steps_number_pre = stepfinder_characteristics['steps_number_pre']
    steps_number = stepfinder_characteristics['steps_number']
    aSNRs = stepfinder_characteristics['aSNRs']
    mSNRs = stepfinder_characteristics['mSNRs']
    STDs = stepfinder_characteristics['STDs']
    resolution = stepfinder_characteristics['resolution']
    window_time = stepfinder_characteristics['window_time']

    #fig.suptitle('Result of the filter window time optimization process')

    ax = axes[0]
    ax.plot(windows / resolution * 1000, steps_number_pre, 'c.', label='pre')
    ax.plot(windows / resolution * 1000, steps_number, 'm.', label='deleted')
    ax.axvline(x=window_time * 1000, alpha=0.5)
    ax.legend(loc='best')
    ax.set_xlabel('Filter window time (ms)')
    ax.set_ylabel('Step count')

    ax = axes[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ax.plot(windows / resolution * 1000, steps_number / steps_number_pre,
                '.')
    ax.axvline(x=window_time * 1000, alpha=0.5)
    ax.set_xlabel('Filter window time (ms)')
    ax.set_ylabel('Ratio of valid steps')

    ax = axes[2]
    ax.plot(windows / resolution * 1000, STDs, '.', label='STD')
    ax.axvline(x=window_time * 1000, alpha=0.5)
    ax.set_xlabel('Filter window time (ms)')
    ax.set_ylabel('STD of step_mass w/o outls')

    ax = axes[3]
    ax.plot(windows / resolution * 1000, aSNRs, '.', label='average SNR')
    ax.plot(windows / resolution * 1000, mSNRs, '.', label='median SNR')
    ax.axvline(x=window_time * 1000, alpha=0.5)
    ax.legend(loc='best')
    ax.set_xlabel('Filter window time (ms)')
    ax.set_ylabel('SNR of outls over STD of step_mass')

    return fig, axes


def plot_result(step_finder_result, simulated_steps=None, decimate=None,
                xlim=None, ylims=None, unfiltered=True, print_pars=True,
                step_size_bins=None, dwell_time_bins=None, axes_steps=None,
                axes_distribution=None):
    """
    Plot the result of a step_finder.

    step_finder_result : StepFinderResult
    simulated_steps : StepsSimulated, optional
        Plot not only the result from the step_finder itself, but also the
        original simulated steps.
    decimate : int, optional
        Decimate the data to be plotted, to reduce the number of points.
        Defaults to 1.
    xlim : tuple of 2 int
        Xlimit of the 'position', 'amplitude', 'step size' and
        'quality of steps' plot.
    ylims : list of tuple of 2 int
        The ylims of the 'position', 'amplitude', 'step size' and
        'quality of steps' plot. Therefore, the lenght of `ylims` needs to be
        4.
    unfiltered : bool, optional
        Plot the unfiltered data, too. Defaults to True.
    step_size_bins : int, optional
        Bins to be used to plot the histogram of step sizes. Defaults to
        'auto'.
    dwell_time_bins : int, optional
        Bins to be used to plot the histogram of dwell times. Defaults to
        'auto'.

    Returns
    -------
    fig_steps : matplotlib.pyplot.figure
    fig_distribution : matplotlib.pyplot.figure
    """
    fbnl_filter = step_finder_result.fbnl_filter
    resolution = fbnl_filter.resolution
    filter_time = fbnl_filter.window / resolution
    p = fbnl_filter.p
    step_mass = fbnl_filter.step_mass

    y_c = step_finder_result.y_c
    step_size_threshold = step_finder_result.step_size_threshold
    min_dwell_time = step_finder_result.min_dwell_time

    steps_pre = step_finder_result.steps_pre
    steps = step_finder_result.steps

    step_distribution = step_finder_result.quality

    datapoints = len(fbnl_filter.data)
    tmin = 0
    tmax = datapoints / resolution
    time = np.linspace(tmin, tmax, datapoints)
    decimate = decimate or 1
    xlim = xlim or (tmin, tmax)
    if ylims is None:
        ylims = [None, None, None, None]

    step_size_bins = step_size_bins or 'auto'
    dwell_time_bins = dwell_time_bins or 'auto'

    if axes_steps is None:
        fig_steps, axes_steps = plt.subplots(4, sharex=True)
        axes_steps = axes_steps.flatten()
    else:
        fig_steps = axes_steps[0].get_figure()

    # Plot data, filtered data and steps
    ax = axes_steps[0]
    if unfiltered:
        ax.plot(time[::decimate], fbnl_filter.data[::decimate], alpha=0.5,
                color='grey')
    ax.plot(time[::decimate], fbnl_filter.data_filtered[::decimate],
            alpha=0.5, color='black')
    if simulated_steps is not None:
        ax.plot(time[::decimate], simulated_steps.data[::decimate], alpha=0.8,
                color='yellow')
    ax.step(time[steps.plateaus[:, 0]], steps.plateau_heights, 'm',
            where='post', lw=0.5)
    ax.plot(time[steps_pre.indices],
            fbnl_filter.data_filtered[steps_pre.indices], 'c.')
    ax.plot(time[steps.indices], fbnl_filter.data_filtered[steps.indices],
            'm.')
    ax.set_xlim(xlim)
    ax.set_ylim(ylims[0])
    ax.set_ylabel('Data (m)')

    # Text of the used parameters of the step finder algorithm
    if print_pars:
        tmp = (r'steps: {}, filter_time$ = {:.1f}\,ms$, p$ = {:.1f}$, '
               'y_c$ = {:.3f}$,\nmin_dwell_time$ = {:.1f}\,ms$, '
               'step_size_threshold$ = {}$')
        # min_step_spacing, y_c, step_size_threshold, filter_time, p
        pars = (steps.number, filter_time * 1000, p, y_c,
                min_dwell_time * 1000, step_size_threshold)
        text_pars = tmp.format(*pars)
        ax.text(0.5, 0.98, text_pars, fontsize=7, color='red',
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes)

    # Plot step_mass with y_c and steps
    ax = axes_steps[1]
    ax.plot(time[::decimate], step_mass[::decimate], alpha=0.5, color='grey')
    ax.axhline(y_c, alpha=0.5)
    ax.axhline(-y_c, alpha=0.5)
    ax.plot(time[steps_pre.indices], step_mass[steps_pre.indices], 'c.')
    if steps is not None:
        ax.plot(time[steps.indices], step_mass[steps.indices], 'mo')
    ax.set_xlim(xlim)
    ax.set_ylim(ylims[1])
    ax.set_ylabel('Amplitude\n(stepsize/noise)')

    # Step sizes and threshold
    ax = axes_steps[2]
    ax.stem(time[steps_pre.indices], steps_pre.step_sizes, 'c', 'c.', 'k-')
    ax.stem(time[steps.indices], steps.step_sizes, 'm', 'mo', 'k-')
    ax.step(time[steps_pre.indices], step_finder_result.min_sizes_pre, 'c',
            where='mid')
    ax.step(time[steps_pre.indices], - step_finder_result.min_sizes_pre, 'c',
            where='mid')
    ax.step(time[steps.indices], step_finder_result.min_sizes, 'm',
            where='mid')
    ax.step(time[steps.indices], - step_finder_result.min_sizes, 'm',
            where='mid')
    ax.set_xlim(xlim)
    ax.set_ylim(ylims[2])
    ax.set_ylabel('Step size')

    # Plot noise and sd
    ax = axes_steps[3]
    # take center of plateaus as start/stop values for plotting function of
    # ratio of sd to noise (there is one more plateau as steps)
    p_center = (steps.plateaus[:, 0]
                + (steps.plateaus[:, 1] - steps.plateaus[:, 0]) / 2)
    p_center = np.round(p_center).astype(int)
    p_center[p_center > datapoints - 1] = datapoints - 1
    ax.step(time[p_center][:-1], step_distribution.step_noise_over_sd, 'm',
            where='post')
    ax.plot(time[steps.indices], step_distribution.step_noise_over_sd, 'mo')
    ax.set_xlim(xlim)
    ax.set_ylim(ylims[3])
    ax.set_ylabel('Quality of steps\nnoise (w/o steps) /\ns.d. around each step')
    ax.set_xlabel('Time (s)')

    if axes_distribution is None:
        fig_distribution, axes_distribution = plt.subplots(2)
        axes_distribution = axes_distribution.flatten()
    else:
        fig_distribution = axes_distribution[0].get_figure()
    #fig_distribution.suptitle('Distribution of step sizes and dwell times')

    # Step size histogram
    ax = axes_distribution[0]
    ax.hist(steps_pre.step_sizes, step_size_bins, color='c',
            alpha=0.35, linewidth=0, label='all steps')
    ax.hist(steps.step_sizes, step_size_bins, color='m',
            alpha=0.65, linewidth=0.5, label='steps after deletion')
    ax.set_ylabel('Count')
    ax.set_xlabel('Step size')

    # Dwell time histogram
    ax = axes_distribution[1]
    ax.hist(steps_pre.dwell_points / resolution, dwell_time_bins, color='c',
            alpha=0.35, linewidth=0, label='all steps')
    ax.hist(steps.dwell_points / resolution, dwell_time_bins, color='m',
            alpha=0.65, linewidth=0.5, label='steps after deletion')
    ax.legend(loc='best')
    ax.set_ylabel('Count')
    ax.set_xlabel('Dwell time (s)')

    return fig_steps, fig_distribution
