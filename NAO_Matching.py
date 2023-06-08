# testing NAO-matching methodology

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import scipy.stats as stats
import xarray as xr

# import dictionaries
from dictionaries import *
from functions import *

# create a function for step 1
# for each start date, i
# compute the signal-adjusted NAO index of the ensemble mean
# as inputs, the function takes the ensemble members array, an associated //
#  model time array, the year of interest, the observed NAO index //
#  and the observed time array
# the function returns the signal-adjusted NAO index of the ensemble mean //
#  and the associated time array
def sign_adjust_NAO_index(year, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time):
    """
    Compute the signal-adjusted NAO index of the ensemble mean
    
    Parameters
    ----------
    year : int
        Year of interest
    ensemble_members_array : numpy.ndarray
        Array of ensemble members
        Must be directly associated with ensemble_members_time
    ensemble_members_time : numpy.ndarray   
        Array of ensemble members time
        Must be directly associated with ensemble_members_array
    obs_nao_index : numpy.ndarray
        Array of observed NAO index
    obs_time : numpy.ndarray
        Array of observed NAO index time

    Returns
    -------
    signal_adjusted_NAO_index : numpy.ndarray
        Array of signal-adjusted NAO index
    """

    # make sure that ensemble_members_array is a numpy array
    # if not, convert it to a numpy array
    if type(ensemble_members_array) != np.ndarray:
        ensemble_members_array = np.array(ensemble_members_array)
    
    # make sure that ensemble_members_time is a numpy array
    # if not, convert it to a numpy array
    if type(ensemble_members_time) != np.ndarray:
        ensemble_members_time = np.array(ensemble_members_time)

    # select only the ensemble members of the year of interest
    # and compute the ensemble mean
    # first find the indices of the year of interest
    year_indices = np.where(ensemble_members_time == year)[0]
    # then select the ensemble members of the year of interest
    year_ensemble_members = ensemble_members_array[year_indices]
    # compute the ensemble mean
    year_ensemble_mean_NAO = np.mean(year_ensemble_members, axis = 0)

    # now we need to adjust the ensemble mean NAO index
    # # to account for the underestimation of predictable signals
    # to do this we calculate the ACC, RPC and RPS
    # but we use a cross validation approach
    # where the required hindcast (specified by year) is left out
    # and the years either side of it are also ommitted
    # so for 1970, we omit 1969, 1970 and 1971, but use the rest of the years
    # first we check that the year of interest is not the first or last year
    # if it is, then we can only omit one year
    if year == ensemble_members_time[0]:
        # if it is the first year, then we omit the first two years
        # and use the rest of the years
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year) | (ensemble_members_time == year + 1))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year) & (ensemble_members_time != year + 1))[0]
    elif year == ensemble_members_time[-1]:
        # if it is the last year, then we omit the last two years
        # and use the rest of the years
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year) | (ensemble_members_time == year - 1))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year) & (ensemble_members_time != year - 1))[0]
    else:
        # otherwise, we omit the year of interest and the two years either side
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year) | (ensemble_members_time == year - 1) | (ensemble_members_time == year + 1))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year) & (ensemble_members_time != year - 1) & (ensemble_members_time != year + 1))[0]

    # set up the arrays for the grand ensemble mean and the ensemble members array with the use years
    grand_ensemble_mean_cross_val = np.mean(ensemble_members_array, axis=0)[use_indices]
    ensemble_members_array_cross_val = ensemble_members_array[:, use_indices]
    ensemble_members_time_cross_val = ensemble_members_time[use_indices]

    # do the same for the observed NAO index
    # and obs time
    obs_nao_index_cross_val = obs_nao_index[use_indices]
    obs_time_cross_val = obs_time[use_indices]

    # for the start date and end date
    # we want to use the full period
    start_date = ensemble_members_time[0]
    end_date = ensemble_members_time[-1]

    # check that the start date and end date are datetimes
    # if not, convert them to datetimes
    if type(start_date) != dt.datetime:
        start_date = dt.datetime(start_date, 1, 1)
    if type(end_date) != dt.datetime:
        end_date = dt.datetime(end_date, 1, 1)

    # check that the start date and end date are specific values
    # print them
    print('Start date: ', start_date)
    print('End date: ', end_date)

    # Compute the ACC using the pearsonr_score function
    # for now we will calculate ACC, RPC and RPS for the long period only
    # we will add the short period later
    acc_score_long, _ = pearsonr_score(obs_nao_index_cross_val, grand_ensemble_mean_cross_val, ensemble_members_time_cross_val, obs_time_cross_val, start_date, end_date)

    # Now compute the RPC using the ACC value
    rpc_score_long = calculate_rpc_time(acc_score_long, ensemble_members_array_cross_val, ensemble_members_time_cross_val, start_date, end_date)

    # Now compute the RPS using the RPC value
    rps_score_long = calculate_rps_time(rpc_score_long, obs_nao_index_cross_val, ensemble_members_array_cross_val, ensemble_members_time_cross_val, start_date, end_date)

    # Now multiply the ensemble mean NAO index by the RPS
    # to get the signal-adjusted NAO index
    signal_adjusted_NAO_index = year_ensemble_mean_NAO * rps_score_long

    # return the signal-adjusted NAO index
    return signal_adjusted_NAO_index

