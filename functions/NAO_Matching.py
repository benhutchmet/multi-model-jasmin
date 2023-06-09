# testing NAO-matching methodology

# import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import scipy.stats as stats
import xarray as xr
from scipy.stats import t

# import dictionaries
# and functions
# Also load the dictionaries from dictionaries.py
sys.path.append("/home/users/benhutch/multi-model/multi-model-jasmin/dictionaries")
from dictionaries import *

# modules
sys.path.append("/home/users/benhutch/multi-model/multi-model-jasmin/functions")
from functions import *


# Define a function to calculate the confidence intervals
def calculate_confidence_intervals(ensemble, alpha=0.05):
    """
    Calculate the confidence intervals for a given ensemble.
    
    Parameters
    ----------
    ensemble : numpy.ndarray
        The ensemble array (members x time).
    alpha : float
        The significance level (default is 0.05 for 95% confidence).
    
    Returns
    -------
    ci_lower : numpy.ndarray
        The lower bound of the confidence interval.
    ci_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """
    
    # Calculate the ensemble mean and standard deviation
    ensemble_mean = np.mean(ensemble, axis=1)
    ensemble_std = np.std(ensemble, axis=1, ddof=1)

    # Calculate the degrees of freedom
    n = ensemble.shape[1]
    dof = n - 1
    
    # Calculate the t-scores for lower and upper confidence levels
    t_score_lower = t.ppf(alpha / 2, dof)
    t_score_upper = t.ppf(1 - alpha / 2, dof)

    # Calculate the confidence intervals
    ci_lower = ensemble_mean + t_score_lower * (ensemble_std / np.sqrt(n))
    ci_upper = ensemble_mean + t_score_upper * (ensemble_std / np.sqrt(n))

    return ci_lower, ci_upper


# Define a new function to lag the ensemble
def lag_ensemble(ensemble_members_array, ensemble_members_time, lag=4):
    """
    Lag the ensemble members array by combining each year with the previous lag-1 years.

    Parameters
    ----------
    ensemble_members_array : numpy.ndarray
        A 2D array of shape (n_ensemble_members, n_years) containing the ensemble members.
    ensemble_members_time : numpy.ndarray
        A 1D array of length n_years containing the time values for each year in the ensemble.
    lag : int, optional
        The number of years to lag the ensemble by. Default is 4.

    Returns
    -------
    lagged_ensemble_members_array : numpy.ndarray
        A 2D array of shape (n_ensemble_members * (n_years - lag + 1), lag) containing the lagged ensemble members.
    lagged_ensemble_members_time : numpy.ndarray
        A 1D array of length n_years - lag + 1 containing the time values for each year in the lagged ensemble.
    """

    # check that the ensemble members array and ensemble members time are the same length
    # if not, raise an error and exit the function
    if ensemble_members_array.shape[1] != ensemble_members_time.shape[0]:
        raise ValueError('ensemble_members_array and ensemble_members_time must be the same length')
    
    
    # make sure that ensemble_members_array is a numpy array
    # if not, convert it to a numpy array
    if type(ensemble_members_array) != np.ndarray:
        ensemble_members_array = np.array(ensemble_members_array)
    
    # make sure that ensemble_members_time is a numpy array
    # if not, convert it to a numpy array
    if type(ensemble_members_time) != np.ndarray:
        ensemble_members_time = np.array(ensemble_members_time)
    
    # get the number of ensemble members
    n_ensemble_members = ensemble_members_array.shape[0]
    
    # get the number of years
    n_years = ensemble_members_array.shape[1]
    
    # create an empty array to store the lagged ensemble members
    # this will have shape 
    # [lag*n_ensemble_members,n_years]
    lagged_ensemble_members_array = np.empty((lag*n_ensemble_members,n_years))

    # create an empty array to store the lagged ensemble members time
    # this will have shape [(n_years - lag) + 1]
    lagged_ensemble_members_time = np.empty((n_years - lag) + 1, dtype='datetime64[ns]')

    # Fill the values for the ensemble members time array
    # this will include all of the values of ensemble members time
    # except for the first lag-1 years
    # this is because we are lagging the ensemble by lag-1 years
    lagged_ensemble_members_time = ensemble_members_time[lag-1:]

    # loop over each ensemble member
    for i in range(n_ensemble_members):
        # loop over each year
        for j in range(n_years):
            # if the year is less than lag-1
            if j < lag-1:
                # set the value of the lagged ensemble members array to NaN
                lagged_ensemble_members_array[i,j] = np.nan
            # if the year is greater than or equal to lag-1
            else:
                # loop over each lag
                for k in range(lag):
                    # set the value of the lagged ensemble members array
                    # to the ensemble members array value
                    # for the ensemble member i
                    # and the year j-k
                    lagged_ensemble_members_array[i*lag+k,j] = ensemble_members_array[i,j-k]

    # exclude the Nans from the returned lagged array
    lagged_ensemble_members_array = lagged_ensemble_members_array[:,3:]                
    # check the lagged ensemble members array
    #print("lagged ensemble members_array", lagged_ensemble_members_array)
    
    # return the lagged ensemble members array and the lagged ensemble members time
    return lagged_ensemble_members_array, lagged_ensemble_members_time



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

    # Convert year to a Pandas Timestamp object
    year_timestamp = pd.to_datetime(str(year))

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
    year_indices = np.where(ensemble_members_time == year_timestamp)[0]
    # then select the ensemble members of the year of interest
    year_ensemble_members = ensemble_members_array[:, year_indices]
    # compute the ensemble mean
    year_ensemble_mean_NAO = np.mean(year_ensemble_members, axis=0)

    # now we need to adjust the ensemble mean NAO index
    # to account for the underestimation of predictable signals
    # to do this we calculate the ACC, RPC and RPS
    # but we use a cross validation approach
    # where the required hindcast (specified by year) is left out
    # and the years either side of it are also omitted
    # so for 1970, we omit 1969, 1970 and 1971, but use the rest of the years
    # first we check that the year of interest is not the first or last year
    # if it is, then we can only omit one year
    if year_timestamp == ensemble_members_time[0]:
        # if it is the first year, then we omit the first two years
        # and use the rest of the years
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year_timestamp) | (ensemble_members_time == year_timestamp + pd.DateOffset(years=1)))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year_timestamp) & (ensemble_members_time != year_timestamp + pd.DateOffset(years=1)))[0]
    elif year_timestamp == ensemble_members_time[-1]:
        # if it is the last year, then we omit the last two years
        # and use the rest of the years
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year_timestamp) | (ensemble_members_time == year_timestamp - pd.DateOffset(years=1)))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year_timestamp) & (ensemble_members_time != year_timestamp - pd.DateOffset(years=1)))[0]
    else:
        # otherwise, we omit the year of interest and the two years either side
        # find the indices of the years to omit
        omit_indices = np.where((ensemble_members_time == year_timestamp) | (ensemble_members_time == year_timestamp - pd.DateOffset(years=1)) | (ensemble_members_time == year_timestamp + pd.DateOffset(years=1)))[0]
        # select the years to use
        use_indices = np.where((ensemble_members_time != year_timestamp) & (ensemble_members_time != year_timestamp - pd.DateOffset(years=1)) & (ensemble_members_time != year_timestamp + pd.DateOffset(years=1)))[0]

    # look at the obs time and the ensemble members time
    # print("obs time pre cross val", obs_time)
    # print("model time pre cross val", ensemble_members_time)

    # change the shape of obs time depending on lagged or not
    if ensemble_members_time.shape == (54,):
        # Set up the indices for the obs_time_adjusted array
        obs_time_adjusted = obs_time[2:]
    elif ensemble_members_time.shape == (51,):
        # Set up the indices for the obs_time_adjusted array
        obs_time_adjusted = obs_time[5:]
    else:
        # Handle the case where the shape is neither (54,) nor (51,)
        raise ValueError("Unexpected shape for ensemble_members_time: {}".format(ensemble_members_time.shape))

    # change the obs_time accordingly
    obs_time = obs_time_adjusted
        
    # look at the indicies
    print("indicies", use_indices)
    
    # set up the arrays for the grand ensemble mean and the ensemble members array with the use years
    grand_ensemble_mean_cross_val = np.mean(ensemble_members_array, axis=0)[use_indices]
    ensemble_members_array_cross_val = ensemble_members_array[:, use_indices]
    ensemble_members_time_cross_val = ensemble_members_time[use_indices]

    # do the same for the observed NAO index and obs time
    obs_nao_index_cross_val = obs_nao_index[use_indices]
    obs_time_cross_val = obs_time[use_indices]

    # for the start date and end date, we want to use the full period
    start_date = ensemble_members_time[0].astype(dt.datetime)
    end_date = ensemble_members_time[-1].astype(dt.datetime)

    # print('start_date', start_date)
    # print('end_date', end_date)

    # print('start_date types', type(start_date))
    # print('end_date types', type(end_date))

    # check that the start date and end date are datetimes
    # if not, convert them to datetimes
    if not isinstance(start_date, dt.datetime):
        start_date = start_date.replace(month=1, day=1)
    if not isinstance(end_date, dt.datetime):
        end_date = end_date.replace(month=1, day=1)

    # check that the start date and end date are specific values
    # print them
    print('Start date: ', start_date)
    print('End date: ', end_date)

    # Compute the ACC using the pearsonr_score function
    # for now, we will calculate ACC, RPC, and RPS for the long period only
    # we will add the short period later
    # check the time for the obs
    # print("obs time cross val", obs_time_cross_val)
    # print(np.shape(obs_time_cross_val))
    # print("model time cross val", ensemble_members_time_cross_val)
    # print(np.shape(ensemble_members_time_cross_val))

    # print("model cross val grand ensemble shape", np.shape(grand_ensemble_mean_cross_val))
    # print("model cross val member array shape", np.shape(ensemble_members_array_cross_val))
    # print("obs cross val shape", np.shape(obs_nao_index_cross_val))

    # test the pearsonr scipy method for getting the ACC values
    acc_score_long, p_value = pearsonr(grand_ensemble_mean_cross_val, obs_nao_index_cross_val)
    
    # Now compute the RPC using the ACC value
    rpc_score_long = calculate_rpc_time(acc_score_long, ensemble_members_array_cross_val, ensemble_members_time_cross_val, start_date, end_date)

    # Now compute the RPS using the RPC value
    rps_score_long = calculate_rps_time(rpc_score_long, obs_nao_index_cross_val, ensemble_members_array_cross_val, ensemble_members_time_cross_val, start_date, end_date)

    # print for debugging
    print("year_ensemble_mean_nao", year_ensemble_mean_NAO)
    print("rps score long", rps_score_long)

    # Now multiply the ensemble mean NAO index by the RPS to get the signal-adjusted NAO index
    signal_adjusted_NAO_index = year_ensemble_mean_NAO * rps_score_long

    # print("from function signal adjusted nao index", signal_adjusted_NAO_index)

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
    mean_of_selected_members : float or None
        Mean of the selected N members, or None if there are not enough members available
    """
    
    # Repeat the value of signal_adjusted_nao_index to match the shape of ensemble_members_array
    signal_adjusted_nao_index = np.repeat(signal_adjusted_nao_index, len(ensemble_members_array))

    # Calculate absolute differences between the signal-adjusted NAO index and the NAO indices of individual ensemble members
    absolute_differences = np.abs(ensemble_members_array[:,0] - signal_adjusted_nao_index)

    # Find the indices of the N members with the smallest absolute differences
    smallest_diff_indices = np.argsort(absolute_differences)[:n_members_to_select]

    print("smallest diff indices", smallest_diff_indices)
    print("smallest diff indices shape", np.shape(smallest_diff_indices))

    # Check if there are enough members available for selection
    if len(smallest_diff_indices) < n_members_to_select:
        print("error: length of smallest diff indices is less than the number of members to select")
        return None

    # Select the N members with the smallest absolute differences
    selected_members = ensemble_members_array[smallest_diff_indices]

    # Calculate the mean of the selected members
    mean_of_selected_members = np.mean(selected_members)

    # Return the mean of the selected members
    return selected_members, mean_of_selected_members


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
    results_members : numpy.ndarray
        Array containing the year and the selected members for each year
    """

    # Convert ensemble_members_array and ensemble_members_time to NumPy arrays if they are not already
    ensemble_members_array = np.array(ensemble_members_array)
    ensemble_members_time = np.array(ensemble_members_time)

    # Create an empty list to store results
    results = []
    results_members = []

    # Loop through each year
    for year in years:
        
        # Compute the signal-adjusted NAO index of the ensemble mean for the current year
        signal_adjusted_nao_index = signal_adjust_NAO_index(year, ensemble_members_array, ensemble_members_time, obs_nao_index, obs_time)

        # Select the ensemble members for the current year
        current_year_indices = np.where(ensemble_members_time == year)[0]
        current_year_ensemble_members = ensemble_members_array[:, current_year_indices]

        # Select N members of the array with the smallest absolute differences and compute their mean
        selected_members, mean_of_selected_members = select_nao_matching_members(year, current_year_ensemble_members, signal_adjusted_nao_index, n_members_to_select)
        
        # Append the year and the mean of selected members to the results list
        results.append([year, mean_of_selected_members])

        # Also append the selected members to the results members list
        results_members.append([year, selected_members])
    
    # Convert the results list to a numpy array
    results_array = np.array(results)
    
    # Return the results array
    return results_array, results_members


# Now write a function to plot the results
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
    results, _ = nao_matching(years, all_ensemble_members, list(model_times_by_model.values())[0], obs_nao_anom, obs_time, n_members_to_select)

    # Extract the years and the mean of selected members from the results array
    years = results[:, 0]
    nao_matched_nao_anom = results[:, 1]

    # # Calculate the ACC for the short and long periods
    # # Using the function pearsonr_score
    # acc_score_short, p_value_short = pearsonr_score(obs_nao_anom,nao_matched_nao_anom, years, obs_time, "1969-01-01", "2010-12-31")
    # acc_score_long, p_value_long = pearsonr_score(obs_nao_anom,nao_matched_nao_anom, years, obs_time, "1969-01-01", "2019-12-31")

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

# Define a new function which plots the results of NAO matching
# using the lagged array with 712 members
def plot_NAO_matched_lagged(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, years, n_members_to_select, lag):
    """ Plot the results of NAO matching methodology for lagged ensemble.

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
    obs_time : 

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

    # Make sure tha all_ensemble_members is an array
    all_ensemble_members = np.array(all_ensemble_members)

    # Call the lag ensemble function
    lagged_ensemble_members_array, lagged_ensemble_members_time = lag_ensemble(all_ensemble_members, list(model_times_by_model.values())[0], lag=lag)

    # Calculate the NAO index for the lagged ensemble
    lagged_ensemble_mean = np.mean(lagged_ensemble_members_array, axis=0)

    # Extract the number of members
    # From the lagged ensemble
    n_members = lagged_ensemble_members_array.shape[0]

    # Calculate the ACC for the short and long periods
    # For the lagged ensemble
    acc_score_short_lagged, p_value_short_lagged = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, years, obs_time, "1969-01-01", "2010-12-31")
    acc_score_long_lagged, p_value_long_lagged = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, years, obs_time, "1969-01-01", "2019-12-31")

    # Now use these ACC scores to calculate the RPC score
    # For the lagged ensemble
    # Using the function calculate_rpc_time
    rpc_short_lagged = calculate_rpc_time(acc_score_short_lagged, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01", "2010-12-31")
    rpc_long_lagged = calculate_rpc_time(acc_score_long_lagged, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01", "2019-12-31")

    # Call the NAO matching function
    results, results_members = nao_matching(years, lagged_ensemble_members_array, lagged_ensemble_members_time, obs_nao_anom, obs_time, n_members_to_select)

    # Convert results members to an array
    results_members_array = np.array(results_members)
    
    # Look at the dimensions of the results members
    print("The dimensions of the results members are: ", np.shape(results_members))
    #print("Results members are: ", results_members)

    # Extract just the arrays of 20 members and stack them into a new array
    reformatted_array = np.array([item[1].flatten() for item in results_members])

    # have a look at this
    print("the reformatted array has shape:", np.shape(reformatted_array))
    #print("the reformatted array", reformatted_array)

    # set results members to be the reformatted array
    results_members = reformatted_array
    
    # Extract the years and the mean of selected members from the results array
    years, nao_matched_nao_anom_lagged = results[:, 0], results[:, 1]

    # Calculate the ACC for the short and long periods
    # Using the function pearsonr_score
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom,nao_matched_nao_anom_lagged, years, obs_time, "1969-01-01", "2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom,nao_matched_nao_anom_lagged, years, obs_time, "1969-01-01", "2019-12-31")

    # Plot the NAO index of the selected members
    ax.plot(years, nao_matched_nao_anom_lagged, color='red', label='NAO-matched DCPP-A lagged')

    # Call the function to calculate the confidence intervals
    ci_5, ci_95 = calculate_confidence_intervals(results_members, alpha=0.05)


    # print the shape of the confidence intervals
    print(np.shape(ci_5))
    print(np.shape(ci_95))
    print(np.shape(years))

    # Add the confidence intervals to the plot
    ax.fill_between(years[:-9], ci_5[:-9], ci_95[:-9], color='red', alpha=0.3)
    ax.fill_between(years[-10:], ci_5[-10:], ci_95[-10:], color='red', alpha=0.2)

    # Plot the observed NAO index 
    ax.plot(obs_time, obs_nao_anom, color='black', label='Observations')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # Check if the p_values are less than 0.01 and set the text accordingly
    if p_value_short < 0.01 and p_value_long < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = '< 0.01'
    elif p_value_short < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = f'= {p_value_long:.2f}'
    elif p_value_long < 0.01:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = '< 0.01'
    else:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = f'= {p_value_long:.2f}'
    
    # Set the title with the ACC and RPC scores
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short_lagged:.2f} (+{rpc_long_lagged:.2f}), N = {n_members}")

    ax.legend(loc="lower right")
    plt.show()
   