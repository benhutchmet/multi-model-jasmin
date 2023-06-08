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
def signal_adjust_NAO_index(year, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time):
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

# Now we write the function which performs the NAO matching
# This function takes as arguments:
# the year
# the ensemble members array
# the signal-adjusted NAO index (for the given year)
# number of members to select (N)
# then selects the N=? members of the array
# with the smallest absolute differences
# between the signal-adjusted ensemble mean NAO index
# and the NAO index for the individual ensemble members
# and then returns the mean of these N members
def select_nao_matching_members(year, ensemble_members_array, signal_adjusted_nao_index, n_members_to_select):
    """
    Selects N members of the array with the smallest absolute differences
    between the signal-adjusted ensemble mean NAO index
    and the NAO index for the individual ensemble members
    and then returns the mean of these N members.

    Parameters
    ----------
    year : int
        Year of interest
    ensemble_members_array : numpy.ndarray
        Array of ensemble members for the year of interest
    signal_adjusted_nao_index : float
        Signal-adjusted NAO index for the given year
    n_members_to_select : int
        Number of members to select

    Returns
    -------
    mean_of_selected_members : float
        Mean of the selected N members
    """

    # Calculate absolute differences between the signal-adjusted NAO index and the NAO indices of individual ensemble members
    absolute_differences = np.abs(ensemble_members_array - signal_adjusted_nao_index)

    # Find the indices of the N members with the smallest absolute differences
    smallest_diff_indices = np.argsort(absolute_differences)[:n_members_to_select]

    # Select the N members with the smallest absolute differences
    selected_members = ensemble_members_array[smallest_diff_indices]

    # Calculate the mean of the selected members
    mean_of_selected_members = np.mean(selected_members)

    # Return the mean of the selected members
    return mean_of_selected_members

# Example usage:
# The 'year', 'ensemble_members_array', and 'signal_adjusted_nao_index' can be obtained from previous steps
# 'n_members_to_select' is a user-defined parameter specifying how many members to select
# mean_of_selected_members = select_nao_matching_members(year, ensemble_members_array, signal_adjusted_nao_index, n_members_to_select)


# Now we write the function which performs the NAO matching
def nao_matching(years, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time, n_members_to_select):
    """
    Perform the NAO matching methodology for a range of years
    
    Parameters
    ----------
    years : list
        List of years to perform NAO matching
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
    n_members_to_select : int
        Number of members to select
    
    Returns
    -------
    results : numpy.ndarray
        Array containing the year and the mean of selected members for each year
    """

    # Create an empty list to store results
    results = []

    # Loop through each year
    for year in years:
        
        # Compute the signal-adjusted NAO index of the ensemble mean for the current year
        signal_adjusted_nao_index = signal_adjust_NAO_index(year, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time)
        
        # Select the ensemble members for the current year
        current_year_indices = np.where(ensemble_members_time == year)[0]
        current_year_ensemble_members = ensemble_members_array[current_year_indices]

        # Select N members of the array with the smallest absolute differences and compute their mean
        mean_of_selected_members = select_nao_matching_members(year, current_year_ensemble_members, signal_adjusted_nao_index, n_members_to_select)
        
        # Append the result to the list
        results.append([year, mean_of_selected_members])
    
    # Convert the results list to a numpy array
    results_array = np.array(results)
    
    # Return the results array
    return results_array


# Example usage:
# Define the range of years for which you want to perform NAO matching
# years = range(1970, 1981)  # example range
# n_members_to_select = 5  # example number of members to select

# Call the nao_matching function
# results = nao_matching(years, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time, n_members_to_select)

# The 'results' array will contain the year and the mean of selected members for each year in the range
# print(results)

# Now write a function to plot the results
# takes models
# model times by model
# model NAO anoms by model
# obs NAO anoms
# obs times
# as input
# and plots the results
def plot_NAO_matched(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, years, n_members_to_select):
    """
    Plot the results of NAO matching methodology.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    years : list
        List of years for NAO matching.
    n_members_to_select : int
        Number of members to select in NAO matching.

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize a list to store results
    all_ensemble_members = []

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        for member in model_nao_anom:
            all_ensemble_members.append(member)

    # Call the NAO matching function
    results = nao_matching(years, all_ensemble_members, model_time, obs_nao_anom, obs_time, n_members_to_select)

    # Extract the years and the mean of selected members from the results array
    years = results[:, 0]
    nao_matched_nao_anom = results[:, 1]

    # Plot the NAO index of the selected members
    ax.plot(years, nao_matched_nao_anom, color='red', label='NAO-matched DCPP-A')

    # Plot the observed NAO index
    ax.plot(obs_time, obs_nao_anom, color='black', label='Observations')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    ax.legend(loc="lower right")
    plt.show()