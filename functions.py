# Functions which are used to process the data on JASMIN further
# Data on JASMIN has been manipulated through a series of steps using \\
# CDO and bash
# Python is used for further manipulation

# Import libraries which may be used by the functions
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mstats
from sklearn.utils import resample
# import the datetime library
from datetime import datetime

# Also load the dictionaries from dictionaries.py
from dictionaries import *

# Function which loads each of the individual ensemble members \\
# from their directory in JASMIN into a dictionary of datasets \\
# grouped by model
def load_ensemble_members(data_dir, model_dirs):
    """
    Load all individual ensemble members into a dictionary of datasets grouped by model.

    Args:
    data_dir (str): Directory containing the model directories.
    model_dirs (list): List of model directories.

    Returns:
    dict: A dictionary of datasets grouped by model.
    """

    def load_datasets(model_dir):
        """
        Load datasets in a model directory.

        Args:
        model_dir (str): Model directory.

        Returns:
        xarray.Dataset: Concatenated datasets in the model directory.
        """
        model_path = os.path.join(data_dir, model_dir)
        files = [f for f in os.listdir(model_path) if f.endswith(".nc")]
        datasets = [xr.open_dataset(os.path.join(model_path, file), chunks={'time': 10}) for file in files]

        return xr.concat(datasets, dim='ensemble_member')

    return {model_dir: load_datasets(model_dir) for model_dir in model_dirs}


# Function to extract the psl and time variabiles from the NetCDF files
def process_ensemble_members(datasets_by_model):
    """
    Processes the ensemble members contained in the datasets_by_model dictionary
    by extracting the desired data, converting units, and setting the time variable's data type.

    Parameters:
    datasets_by_model (dict): Dictionary of datasets grouped by model

    Returns:
    dict: The model_times_by_model dictionary
    dict: The model_nao_anoms_by_model dictionary
    """
    
    def process_model_dataset(dataset):
        """
        Processes the dataset by extracting the desired data, converting units,
        and setting the time variable's data type.

        Parameters:
        dataset (xarray.Dataset): The input dataset

        Returns:
        numpy.ndarray: The model_time array
        numpy.ndarray: The model_nao_anom array
        """

        # Extract the data for the model
        # Extract the data based on the dataset type
        if "psl" in dataset:
            data_var = "psl"  # For the model dataset
        elif "var151" in dataset:
            data_var = "var151"  # For the observations dataset
        else:
            raise ValueError("Unknown dataset type. Cannot determine data variable.")

        model_data = dataset[data_var]
        model_time = model_data["time"].values

        # Set the type for the time variable
        model_time = model_time.astype("datetime64[Y]")

        # Process the model data from Pa to hPa
        if len(model_data.dims) == 4:
            model_nao_anom = model_data[:, :, 0, 0] / 100
        elif len(model_data.dims) == 3:
            model_nao_anom = model_data[:, 0, 0] / 100
        else:
            raise ValueError("Unexpected number of dimensions in the dataset.")

        return model_time, model_nao_anom

    model_times_by_model, model_nao_anoms_by_model = {}, {}
    for model, ds in datasets_by_model.items():
        model_times_by_model[model], model_nao_anoms_by_model[model] = process_model_dataset(ds)
    
    return model_times_by_model, model_nao_anoms_by_model

# Function for processing the observations
def process_observations(obs):
    """
    Process the observations data by extracting the time, converting units, and calculating NAO anomalies.

    Parameters
    ----------
    obs : xarray.Dataset
        The xarray dataset containing the observations data.

    Returns
    -------
    obs_nao_anom : numpy.ndarray
        The processed observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    """
    # Extract the data for the observations
    obs_psl = obs["var151"]
    obs_time = obs_psl["time"].values

    print(np.shape(obs_psl))
    print(np.shape(obs_time))

    # Set the type for the time variable
    obs_time = obs_time.astype("datetime64[Y]")

    # Process the obs data from Pa to hPa
    obs_nao_anom = obs_psl[:] / 100

    return obs_nao_anom, obs_time

# Function to calculate ensemble mean for each model
def ensemble_mean(data_array):
    return np.mean(data_array, axis=0)

# Function to calculate the ACC and significance
def pearsonr_score(obs, model, model_times, obs_times, start_date, end_date):
    """
    Calculate the Pearson correlation coefficient and p-value between two time series,
    considering the dimensions of the model and observation time arrays.

    Parameters:
    obs (array-like): First time series (e.g., observations)
    model (array-like): Second time series (e.g., model mean)
    model_times (array-like): Datetime array corresponding to the model time series
    obs_times (array-like): Datetime array corresponding to the observation time series
    start_date (str): Start date (inclusive) in the format 'YYYY-MM-DD'
    end_date (str): End date (inclusive) in the format 'YYYY-MM-DD'

    Returns:
    tuple: Pearson correlation coefficient and p-value
    """

    # Ensure the time series are numpy arrays or pandas Series
    time_series1 = np.array(obs)
    time_series2 = np.array(model)

    # Ensure the start_date and end_date are pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Convert obs_times to an array of Timestamp objects
    obs_times = np.vectorize(pd.Timestamp)(obs_times)

    # Analyze dimensions of model_times and obs_times
    model_start_index = np.where(model_times == start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]
    obs_start_index = np.where(obs_times >= start_date)[0][0]
    obs_end_index = np.where(obs_times <= end_date)[0][-1]

    # Filter the time series based on the analyzed dimensions
    filtered_time_series1 = time_series1[obs_start_index:obs_end_index+1]
    filtered_time_series2 = time_series2[model_start_index:model_end_index+1]

    # Calculate the Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(filtered_time_series1, filtered_time_series2)

    return correlation_coefficient, p_value
    
# example usage included for debugging
# obs_nao_anom = np.random.rand(100)  # Replace with your actual time series
# grand_ensemble_mean = np.random.rand(100)  # Replace with your actual time series
# model_times = pd.date_range(start='1960-01-01', periods=len(grand_ensemble_mean), freq='M')
# obs_times = pd.date_range(start='1960-01-01', periods=len(obs_nao_anom), freq='M')

# acc_score, p_value = pearsonr_score_v2(obs_nao_anom, grand_ensemble_mean, model_times, obs_times, '1966-01-01', '2010-12-31')

# print(acc_score, p_value)

# Define a function to calculate the RPC score
def calculate_rpc(correlation_coefficient, forecast_members):
    """
    Calculate the Ratio of Predictable Components (RPC) given the correlation
    coefficient (ACC) and individual forecast members.

    Parameters:
    correlation_coefficient (float): Correlation coefficient (ACC)
    forecast_members (array-like): Individual forecast members

    Returns:
    float: Ratio of Predictable Components (RPC)
    """

    # Convert the input arrays to numpy arrays
    forecast_members = np.array(forecast_members)

    # Calculate the standard deviation of the predictable signal for forecasts (σfsig)
    sigma_fsig = np.std(np.mean(forecast_members, axis=0))

    # Calculate the total standard deviation for forecasts (σftot)
    sigma_ftot = np.std(forecast_members)

    # Calculate the RPC
    rpc = correlation_coefficient / (sigma_fsig / sigma_ftot)

    return rpc

def calculate_rpc_time(correlation_coefficient, forecast_members, model_times, start_date, end_date):
    """
    Calculate the Ratio of Predictable Components (RPC) given the correlation
    coefficient (ACC) and individual forecast members for a specific time period.

    Parameters:
    correlation_coefficient (float): Correlation coefficient (ACC)
    forecast_members (array-like): Individual forecast members
    model_times (array-like): Datetime array corresponding to the model time series
    start_date (str): Start date (inclusive) in the format 'YYYY-MM-DD'
    end_date (str): End date (inclusive) in the format 'YYYY-MM-DD'

    Returns:
    float: Ratio of Predictable Components (RPC)
    """

    # Convert the input arrays to numpy arrays
    forecast_members = np.array(forecast_members)

    # Convert the start_date and end_date to pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Find the start and end indices of the time period for the model
    model_start_index = np.where(model_times >= start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]

    # Filter the forecast members based on the start and end indices
    forecast_members = forecast_members[:, model_start_index:model_end_index+1]

    # Calculate the standard deviation of the predictable signal for forecasts (σfsig)
    sigma_fsig = np.std(np.mean(forecast_members, axis=0))

    # Calculate the total standard deviation for forecasts (σftot)
    sigma_ftot = np.std(forecast_members)

    # Calculate the RPC
    rpc = correlation_coefficient / (sigma_fsig / sigma_ftot)

    return rpc


# Two ways of calculating the uncertainty
# First calculates the ensemble standard deviation
# for each start date and then uses this to get the 
# confidence intervals
def calculate_confidence_intervals_sd(ensemble_members_array, lower_bound=5, upper_bound=95):
    """
    Calculate the confidence intervals of an ensemble members array using the ensemble
    standard deviation for each start date. This method calculates the ensemble standard
    deviation and uses it to obtain the confidence intervals.

    Parameters
    ----------
    ensemble_members_array : numpy.ndarray
        The array of ensemble members data.
    lower_bound : int, optional, default: 5
        The lower percentile bound for the confidence interval.
    upper_bound : int, optional, default: 95
        The upper percentile bound for the confidence interval.

    Returns
    -------
    conf_interval_lower : numpy.ndarray
        The lower bound of the confidence interval.
    conf_interval_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """
    # Calculate the ensemble standard deviation for each start date
    ensemble_sd = np.std(ensemble_members_array, axis=0)
    
    # Calculate the grand ensemble mean
    grand_ensemble_mean = np.mean(ensemble_members_array, axis=0)

    # Calculate the z-scores corresponding to the lower and upper bounds
    z_score_lower = np.percentile(ensemble_sd, lower_bound)
    z_score_upper = np.percentile(ensemble_sd, upper_bound)

    # Calculate the 5% and 95% confidence intervals using the standard deviation and the grand ensemble mean
    conf_interval_lower = grand_ensemble_mean - z_score_upper * ensemble_sd
    conf_interval_upper = grand_ensemble_mean + z_score_upper * ensemble_sd

    return conf_interval_lower, conf_interval_upper

# Second way - more simple
# Function to calculate the confidence intervals
def calculate_confidence_intervals(ensemble_members_array, lower_bound=5, upper_bound=95):
    """
    Calculate the confidence intervals of an ensemble members array using a simpler
    method that directly computes the percentiles of the ensemble members data.

    Parameters
    ----------
    ensemble_members_array : numpy.ndarray
        The array of ensemble members data.
    lower_bound : int, optional, default: 5
        The lower percentile bound for the confidence interval.
    upper_bound : int, optional, default: 95
        The upper percentile bound for the confidence interval.

    Returns
    -------
    conf_interval_lower : numpy.ndarray
        The lower bound of the confidence interval.
    conf_interval_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """
    conf_interval_lower = np.percentile(ensemble_members_array, lower_bound, axis=0)
    conf_interval_upper = np.percentile(ensemble_members_array, upper_bound, axis=0)
    return conf_interval_lower, conf_interval_upper

# Function to lag the NAO data and create a new time
# array
def process_lagged_ensemble_mean(data, lag=4, start_year=1969, end_year=2019):
    """
    Lag the input data by a specified number of years and create a new time array
    corresponding to the lagged data. This function is useful for processing ensemble
    mean data, such as the North Atlantic Oscillation (NAO) time series.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input data to be lagged, typically an ensemble mean time series.
    lag : int, optional, default: 4
        The number of years to lag the data by.
    start_year : int, optional, default: 1969
        The start year of the time array.
    end_year : int, optional, default: 2019
        The end year of the time array.
    
    Returns
    -------
    lagged_data_mean : numpy.ndarray
        The lagged ensemble mean data.
    model_time_lagged : numpy.ndarray
        The new time array corresponding to the lagged data.
    """
    # create a dummy time array to compare against
    # for years from 1969 to 2019
    time_array = np.arange(start_year, end_year + 1)
    print("time_array: ", time_array)
    print("length of time_array: ", len(time_array))

    def lagged_ensemble_mean(data, lag):
        # Initialize an empty array for the lagged ensemble mean
        #lagged_mean = np.empty(len(data) - lag + 1)
        lagged_mean = np.empty((len(data) - lag) + 1)
        # check if the length of the lagged_mean array is correct
        print(len(lagged_mean))

        # Calculate the lagged ensemble mean for each year
        for i in range((len(data) - lag) + 1):
            lagged_mean[i] = np.mean(data[i:i + lag])

        return lagged_mean

    # Calculate the lagged ensemble mean for the data
    lagged_data_mean = lagged_ensemble_mean(data, lag)

    # Calculate the corresponding model_time for the lagged data
    model_time_lagged = time_array

    return lagged_data_mean, model_time_lagged

# Example usage
# lagged_adjusted_var_mean, model_time_lagged = process_lagged_ensemble_mean(adjusted_var_model_nao_anom_raw, model_time_raw, lag=4)

# Function to adjust the variance of the ensemble
# Used once the no. of ensemble members has been 4x
# through the lagging process
def adjust_variance(model_time_series, rpc_short, rpc_long):
    """
    Adjust the variance of an ensemble mean time series by multiplying by the RPC score. This accounts for the signal to noise issue in the ensemble mean.

    Parameters
    ----------
    model_time_series : numpy.ndarray
        The input ensemble mean time series.
    rpc_short : float
        The RPC score for the short period.
    rpc_long : float
        The RPC score for the long period.

    Returns
    -------
    model_time_series_var_adjust_short : numpy.ndarray
        The variance adjusted ensemble mean time series for the short period RPC (1960-2010).
    model_time_series_var_adjust_long : numpy.ndarray
        The variance adjusted ensemble mean time series for the long period RPC (1960-2019).
    """

    # Adjust the variance of the ensemble mean time series
    model_time_series_var_adjust_short = rpc_short * model_time_series
    model_time_series_var_adjust_long = rpc_long * model_time_series

    return model_time_series_var_adjust_short, model_time_series_var_adjust_long


# Example usage - included for debugging
# obs_data = np.array([1, 2, 3, 4, 5])
# model_data = np.array([2, 4, 6, 8, 10])

# obs_time_series_var_adjust, model_time_series_var_adjust = adjust_variance(obs_data, model_data)
# print(obs_time_series_var_adjust)
# print(model_time_series_var_adjust)

# function for calculating the RMSE and 5-95% uncertainty intervals for the variance adjusted output
def compute_rmse_confidence_intervals(obs_nao_anoms, adjusted_lagged_model_nao_anoms, obs_time, model_time_lagged, lower_bound=5, upper_bound=95):
    """
    Compute the root-mean-square error (RMSE) between the variance-adjusted ensemble
    mean and the observations. Calculate the 5%-95% confidence intervals for the
    variance-adjusted model output.

    Parameters
    ----------
    obs_nao_anoms : numpy.ndarray
        The observed NAO anomalies time series.
    adjusted_lagged_model_nao_anoms : numpy.ndarray
        The adjusted and lagged model NAO anomalies time series.
    obs_time : numpy.ndarray
        The time array for the observed NAO anomalies.
    model_time_lagged : numpy.ndarray
        The time array for the adjusted and lagged model NAO anomalies.
    lower_bound : int, optional, default: 5
        The lower percentile bound for the confidence interval.
    upper_bound : int, optional, default: 95
        The upper percentile bound for the confidence interval.

    Returns
    -------
    conf_interval_lower : numpy.ndarray
        The lower bound of the confidence interval.
    conf_interval_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """
    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # Find the indices of the common years in both arrays
    obs_indices = np.where(np.isin(obs_time, common_years))[0]
    model_indices = np.where(np.isin(model_time_lagged, common_years))[0]

    # Create new arrays with the corresponding values for the common years
    obs_nao_anoms_matched = obs_nao_anoms[obs_indices].values
    adjusted_lagged_model_nao_anoms_matched = adjusted_lagged_model_nao_anoms[model_indices]
    
    # Compute the root-mean-square error (RMSE) between the ensemble mean and the observations
    rmse = np.sqrt(np.mean((obs_nao_anoms_matched - adjusted_lagged_model_nao_anoms_matched)**2, axis=0))

    # Calculate the upper z-score for the RMSE
    z_score_upper = np.percentile(rmse, upper_bound)
    
    # Calculate the 5% and 95% confidence intervals using the RMSE
    conf_interval_lower = adjusted_lagged_model_nao_anoms_matched - (z_score_upper * rmse)
    conf_interval_upper = adjusted_lagged_model_nao_anoms_matched + (z_score_upper * rmse)

    return conf_interval_lower, conf_interval_upper

# optimize function
def optimize_ensemble_members(all_ensemble_members_array, no_ensemble_members, obs_nao_anom, obs_time, model_times):
    """
    Greedily select no_ensemble_members ensemble members that maximizes the ACC scores.
    """
    ensemble_members = []
    for _ in range(no_ensemble_members):
        max_acc_score_short, max_acc_score_long, best_member = -np.inf, -np.inf, None
        for member in all_ensemble_members_array:
            if any(np.array_equal(member, ensemble) for ensemble in ensemble_members):
                continue
            ensemble = ensemble_members + [member.tolist()]
            grand_ensemble_mean = np.mean(ensemble, axis=0)
            acc_score_short, _ = pearsonr_score(obs_nao_anom, grand_ensemble_mean, model_times, obs_time, "1966-01-01","2010-12-31")
            acc_score_long, _ = pearsonr_score(obs_nao_anom, grand_ensemble_mean, model_times, obs_time, "1966-01-01","2019-12-31")
            if acc_score_short > max_acc_score_short and acc_score_long > max_acc_score_long:
                max_acc_score_short, max_acc_score_long, best_member = acc_score_short, acc_score_long, member.tolist()
        if best_member is not None:
            ensemble_members.append(best_member)
    return [np.array(member) for member in ensemble_members]


# Now write a plotting function
def plot_ensemble_members_and_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time):
    """
    Plot the ensemble mean of all members from all models and each of the ensemble members.

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

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Plot the ensemble members and calculate the ensemble mean for each model
    ensemble_means = []

    # Initialize a dictionary to store the count of ensemble members for each model
    ensemble_member_counts = {}

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # If the model_name is not in the dictionary, initialize its count to 0
        if model_name not in ensemble_member_counts:
            ensemble_member_counts[model_name] = 0

        # Plot ensemble members
        for member in model_nao_anom:
            ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

            # Increment the count of ensemble members for the current model
            ensemble_member_counts[model_name] += 1

        # Calculate and store ensemble mean
        ensemble_means.append(ensemble_mean(model_nao_anom))

    # Convert the ensemble_member_counts dictionary to a list of tuples
    ensemble_member_counts_list = [(model, count) for model, count in ensemble_member_counts.items()]

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # save the number of ensemble members
    no_ensemble_members = len(all_ensemble_members_array[:,0])

    # Calculate the grand ensemble mean using the new method
    grand_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # Calculate the ACC score for the //
    # short period using the function pearsonr_score
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")

    # Calculate the ACC score for the //
    # long period using the function pearsonr_score
    # long period 1966 - 2019
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2019-12-31")

    # Calculate the RPC score for the short period
    # using the function calculate_rpc_time
    # short period 1966 - 2010
    rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, list(model_times_by_model.values())[0], "1966-01-01","2010-12-31")

    # Calculate the RPC score for the long period
    # using the function calculate_rpc_time
    # long period 1966 - 2019
    rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, list(model_times_by_model.values())[0], "1966-01-01","2019-12-31")

    # Calculate the 5-95% confidence intervals using the two functions options
    # First calculate_confidence_intervals_sd
    # Then calculate_confidence_intervals
    conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(all_ensemble_members_array)

    # Plot the grand ensemble mean with the ACC score in the legend
    ax.plot(list(model_times_by_model.values())[0], grand_ensemble_mean, color="red", label=f"DCPP-A")

    # Plot the 5-95% confidence intervals
    # different shading for the two different time periods
    # short period 1966 - 2010
    ax.fill_between(list(model_times_by_model.values())[0][:-9], conf_interval_lower[:-9], conf_interval_upper[:-9], color="red", alpha=0.3)
    # for period 2010 - 2019
    ax.fill_between(list(model_times_by_model.values())[0][-10:], conf_interval_lower[-10:], conf_interval_upper[-10:], color="red", alpha=0.2)

    # Plot ERA5 data
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # check if the p-value is les than 0.01
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
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f}), N = {no_ensemble_members}")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    #fig.savefig(os.path.join(plots_dir, "nao_ensemble_mean_and_individual_members.png"), dpi=300)
    # include the number of ensemble members in the filename
    # and the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()

# Define a function that will plot a randomly selected group of ensemble members and plot the average
def plot_random_ensemble_members_and_stats(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, no_ensemble_members=10):
    """
    Plot randomly selected ensemble members, the ensemble mean of these members, and observations, along with ACC and RPC scores.

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
    no_ensemble_members : int, optional
        The number of ensemble members to randomly select. The default is 10.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    all_ensemble_members = []

    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        for member in model_nao_anom:
            all_ensemble_members.append(member)

    all_ensemble_members_array = np.array(all_ensemble_members)

    # Randomly select no_ensemble_members ensemble members
    random_indices = np.random.choice(range(len(all_ensemble_members_array)), no_ensemble_members, replace=False)
    random_ensemble_members = all_ensemble_members_array[random_indices]

    for member in random_ensemble_members:
        ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

    # Calculate and plot the grand ensemble mean, ACC score, RPC score, and confidence intervals based on the random ensemble members
    grand_ensemble_mean = np.mean(random_ensemble_members, axis=0)

    # calculate ACC score and p-value for short period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")

    # calculate ACC score and p-value for long period
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2019-12-31")

    # calculate RPC score for short period
    rpc_short = calculate_rpc_time(acc_score_short, random_ensemble_members, list(model_times_by_model.values())[0], "1966-01-01","2010-12-31")

    # calculate RPC score for long period
    rpc_long = calculate_rpc_time(acc_score_long, random_ensemble_members, list(model_times_by_model.values())[0], "1966-01-01","2019-12-31")

    # Calculate the 5-95% confidence intervals using the two functions options
    conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(random_ensemble_members)

    # Plot the grand ensemble mean
    ax.plot(list(model_times_by_model.values())[0], grand_ensemble_mean, color="red", label=f"DCPP-A")

    # Plot the 5-95% confidence intervals
    # different shading for the two different time periods
    # short period 1966 - 2010
    ax.fill_between(list(model_times_by_model.values())[0][:-9], conf_interval_lower[:-9], conf_interval_upper[:-9], color="red", alpha=0.3)
    # for period 2010 - 2019
    ax.fill_between(list(model_times_by_model.values())[0][-10:], conf_interval_lower[-10:], conf_interval_upper[-10:], color="red", alpha=0.2)

    # Plot the observations
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")
    ax.legend(loc="lower right")

    # check if the p-value is les than 0.01
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
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f}), N = {no_ensemble_members}")

    # Save the plot
    #plots_dir = "plots"  # replace this with your actual plots directory
    # also include the no_ensemble_members in the filename
    # and the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    plt.show()


# Define a modified version of the function above
# which involves an optimization step
# to randomly select the best ensemble members
def plot_random_ensemble_members_and_stats_optimize(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, no_ensemble_members=10):
    """
    Plot optimally selected ensemble members, the ensemble mean of these members, and observations, along with ACC and RPC scores.

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
    no_ensemble_members : int, optional
        The number of ensemble members to select. The default is 10.

    Returns
    -------
    None
    """
    # set up
    fig, ax = plt.subplots(figsize=(10, 6))

    all_ensemble_members = []

    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        for member in model_nao_anom:
            all_ensemble_members.append(member)

    all_ensemble_members_array = np.array(all_ensemble_members)

    # Optimally select no_ensemble_members ensemble members
    optimal_ensemble_members = optimize_ensemble_members(all_ensemble_members_array, no_ensemble_members, obs_nao_anom, obs_time, list(model_times_by_model.values())[0])

    # plot each of the members
    for member in optimal_ensemble_members:
        ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

    # Calculate and plot the grand ensemble mean, ACC score, RPC score, and confidence intervals based on the random ensemble members
    grand_ensemble_mean = np.mean(optimal_ensemble_members, axis=0)

    # calculate ACC score and p-value for short period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")

    # calculate ACC score and p-value for long period
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2019-12-31")

    # calculate RPC score for short period
    rpc_short = calculate_rpc_time(acc_score_short, optimal_ensemble_members, list(model_times_by_model.values())[0], "1966-01-01","2010-12-31")

    # calculate RPC score for long period
    rpc_long = calculate_rpc_time(acc_score_long, optimal_ensemble_members, list(model_times_by_model.values())[0], "1966-01-01","2019-12-31")

    # Calculate the 5-95% confidence intervals using the two functions options
    conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(optimal_ensemble_members)

    # Plot the grand ensemble mean
    ax.plot(list(model_times_by_model.values())[0], grand_ensemble_mean, color="red", label=f"DCPP-A")

    # Plot the 5-95% confidence intervals
    # different shading for the two different time periods
    # short period 1966 - 2010
    ax.fill_between(list(model_times_by_model.values())[0][:-9], conf_interval_lower[:-9], conf_interval_upper[:-9], color="red", alpha=0.3)
    # for period 2010 - 2019
    ax.fill_between(list(model_times_by_model.values())[0][-10:], conf_interval_lower[-10:], conf_interval_upper[-10:], color="red", alpha=0.2)

    # Plot the observations
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")
    ax.legend(loc="lower right")

    # check if the p-value is les than 0.01
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
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f}), N = {no_ensemble_members}")

    # Save the plot
    #plots_dir = "plots"  # replace this with your actual plots directory
    # also include the no_ensemble_members in the filename
    # and the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_optimize_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    plt.show()   

# Define a function that will just plot the noise for demonstration purposes
def plot_ensemble_members_and_obs(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, no_ensemble_members=10, plot_obs=True):
    """
    Plot the ensemble mean of all members from all models and a random selection of the ensemble members.

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
    no_ensemble_members : int, optional
        The number of ensemble members to randomly select and plot. The default is 10.
    plot_obs : bool, optional
        Whether to plot the observations. The default is True.

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Plot the ensemble members and calculate the ensemble mean for each model
    ensemble_means = []

    # Initialize a dictionary to store the count of ensemble members for each model
    ensemble_member_counts = {}

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # If the model_name is not in the dictionary, initialize its count to 0
        if model_name not in ensemble_member_counts:
            ensemble_member_counts[model_name] = 0

        # Add each member to the list of all ensemble members
        for member in model_nao_anom:
            all_ensemble_members.append(member)

            # Increment the count of ensemble members for the current model
            ensemble_member_counts[model_name] += 1

        # Calculate and store ensemble mean
        ensemble_means.append(ensemble_mean(model_nao_anom))

    # Convert the ensemble_member_counts dictionary to a list of tuples
    ensemble_member_counts_list = [(model, count) for model, count in ensemble_member_counts.items()]

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # Randomly select "no_ensemble_members" ensemble members to plot
    random_ensemble_members_indices = np.random.choice(len(all_ensemble_members_array), no_ensemble_members, replace=False)
    random_ensemble_members = all_ensemble_members_array[random_ensemble_members_indices]

    # Plot the randomly selected ensemble members
    for member in random_ensemble_members:
        ax.plot(model_time, member, color="red", alpha=0.4, linewidth=0.8)

    # print the number of ensemble members
    # in the top right corner
    ax.text(0.98, 0.98, f"Number of ensemble members: {no_ensemble_members}", transform=ax.transAxes, ha="right", va="top")

    # optionally plot the observations
    if plot_obs:
        # Plot ERA5 data
        ax.plot(obs_time[3:], obs_nao_anom[3:], color="black", label="ERA5")
    else:
        pass

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # Set the title with the ACC and RPC scores
    #ax.set_title(f"NAO ensemble mean and individual members (ACC: {acc_score:.2f}, RPC: {rpc:.2f})")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    # if plot_obs: is true, then include with_obs in the filename
    # if plot_obs: is false, then include without_obs in the filename
    # make sure to include the no_ensemble_members in the filename
    # include the number of ensemble members in the filename
    # and the current date
    if plot_obs:
        fig.savefig(os.path.join(plots_dir, f"nao_ensemble_members_with_obs_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)
    else:
        fig.savefig(os.path.join(plots_dir, f"nao_ensemble_members_without_obs_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()


# Define a function for plotting the model subplots for the raw data
# TODO -- SORT OUT THIS FUNCTION AND ITS ASSOCIATED FUNCTIONS
    # -RPC SCORES TOO LOW
    # -EXTEND PERIOD FOR 2010-2019
    # IS IS SCALED TO THE RIGHT MAGNITUDE?
    # EMAIL DOUG POTENTIALLY
def plot_ensemble_members_and_lagged_adjusted_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, lag=4):
    """
    Plot the ensemble mean of all members from all models and each of the ensemble members, with lagged and adjusted variance applied to the grand ensemble mean.

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
    lag : int, optional, default: 4
        The number of years to lag the grand ensemble mean by.

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Plot the ensemble members and calculate the ensemble mean for each model
    ensemble_means = []

    # Initialize a dictionary to store the count of ensemble members for each model
    ensemble_member_counts = {}

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # If the model_name is not in the dictionary, initialize its count to 0
        if model_name not in ensemble_member_counts:
            ensemble_member_counts[model_name] = 0

        # Plot ensemble members
        for member in model_nao_anom:
            ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

            # Increment the count of ensemble members for the current model
            ensemble_member_counts[model_name] += 1

        # Calculate and store ensemble mean
        ensemble_means.append(ensemble_mean(model_nao_anom))

    # Convert the ensemble_member_counts dictionary to a list of tuples
    ensemble_member_counts_list = [(model, count) for model, count in ensemble_member_counts.items()]

    # count the number of ensemble members
    no_ensemble_members = len(all_ensemble_members)

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # Calculate the grand ensemble mean using the new method
    grand_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # print the shape of the grand ensemble mean
    print("shape of grand ens mean", np.shape(grand_ensemble_mean))
    # also print the model times and its shape
    print("model time", (list(model_times_by_model.values())[0]))

    # Apply lagging and variance adjustment to the grand ensemble mean
    lagged_grand_ensemble_mean, model_time_lagged = process_lagged_ensemble_mean(grand_ensemble_mean, list(model_times_by_model.values())[0], lag)

    # check the time output from this function
    print("shape of model_time_lagged", np.shape(model_time_lagged))
    print("model_time_lagged", model_time_lagged)
    print("shape of lagged_grand_ensemble_mean", np.shape(lagged_grand_ensemble_mean))
    print("lagged_grand_ensemble_mean", lagged_grand_ensemble_mean)

    # calculate the ACC (short and long) for the lagged grand ensemble mean
    acc_score_short_lagged, _ = pearsonr_score(obs_nao_anom, lagged_grand_ensemble_mean, model_time_lagged, obs_time, "1969-01-01","2010-12-31")
    acc_score_long_lagged, _ = pearsonr_score(obs_nao_anom, lagged_grand_ensemble_mean, model_time_lagged, obs_time, "1969-01-01","2015-12-31")

    # and for the grand ensemble mean
    acc_score_short, _ = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")
    acc_score_long, _ = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2019-12-31")

    # calculate the RPC (short and long) for the lagged grand ensemble mean
    rpc_short_lagged = calculate_rpc_time(acc_score_short_lagged, all_ensemble_members_array, list(model_times_by_model.values())[0], "1969-01-01","2010-12-31")
    rpc_long_lagged = calculate_rpc_time(acc_score_long_lagged, all_ensemble_members_array, list(model_times_by_model.values())[0], "1969-01-01","2015-12-31")

    # print these rpc scores
    # print(rpc_short_lagged)
    # print(rpc_long_lagged)

    # and for the grand ensemble mean
    rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, list(model_times_by_model.values())[0], "1966-01-01","2010-12-31")
    rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, list(model_times_by_model.values())[0], "1966-01-01","2019-12-31")

    # print(rpc_short)
    # print(rpc_long)

    #test
    #set rpc_short=10
    # replicate marcheggiani
    # THIS IS NOT RIGHT - EMAIL DOUG?
    #rpc_short_lagged=10.1
    
    # print the time series before adjustment
    print(lagged_grand_ensemble_mean)

    # apply the variance adjustment (via RPC scaling) to the lagged grand ensemble mean
    lagged_adjusted_grand_ensemble_mean_short, lagged_adjusted_grand_ensemble_mean_long = adjust_variance(lagged_grand_ensemble_mean,rpc_short_lagged,rpc_long_lagged)

    print(lagged_adjusted_grand_ensemble_mean_short)
    print(lagged_adjusted_grand_ensemble_mean_long)
    
    # Also just apply the variance adjustment to the grand ensemble mean
    adjusted_grand_ensemble_mean_short, adjusted_grand_ensemble_mean_long = adjust_variance(grand_ensemble_mean,rpc_short,rpc_long)

    # Calculate the ACC score using the function pearsonr_score with the lagged and adjusted grand ensemble mean

    # look into the model-time_lagged_data
    # print(obs_time)
    # print(np.shape(obs_time))
    # print(model_time_lagged)
    # print(np.shape(model_time_lagged))

    # ------------------NOTE------------------
    # why is model_time_lagged used for pearsonr_score
    # and list(model_times_by_model.values())[0] used for calculate_rpc_time

    # -----------------NOTE-------------------
    # we chose to use the SHORT version of RPC lagged adjusted grand ensmeble mean
    
    #  for the short period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_grand_ensemble_mean_short, model_time_lagged, obs_time, "1969-01-01","2010-12-31")

    # for the long period
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_grand_ensemble_mean_short, model_time_lagged, obs_time, "1969-01-01","2015-12-31")

    # calculate RPC score for short period
    rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, list(model_times_by_model.values())[0], "1969-01-01","2010-12-31")

    # calculate RPC score for long period
    rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, list(model_times_by_model.values())[0], "1969-01-01","2015-12-31")

    # print(obs_time[6:])
    # print(model_time_lagged)
    # print(np.shape(obs_nao_anom[6:]))
    # print(np.shape(lagged_adjusted_grand_ensemble_mean))
    # print(obs_nao_anom[6:])
    # print(lagged_adjusted_grand_ensemble_mean)
    
    # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
    conf_interval_lower, conf_interval_upper = compute_rmse_confidence_intervals(obs_nao_anom, lagged_adjusted_grand_ensemble_mean_short, obs_time, model_time_lagged)

    # Plot the grand ensemble mean with the ACC score in the legend
    ax.plot(model_time_lagged, lagged_adjusted_grand_ensemble_mean_short[:-4], color="red", label=f"DCPP-A")

    # print(np.shape(model_time_lagged))
    # print(np.shape(adjusted_grand_ensemble_mean[3:-5]))
    # print(model_time_lagged)
    # print(list(model_times_by_model.values())[0][3:-5])
    
    # Plot the grand ensemble mean variance adjusted only
    ax.plot(list(model_times_by_model.values())[0], adjusted_grand_ensemble_mean_short, color="red", alpha=0.8, linestyle="--")

    # Plot the 5-95% confidence intervals for the short period
    ax.fill_between(model_time_lagged[:-5], conf_interval_lower[:-5], conf_interval_upper[:-5], color="red", alpha=0.3)
    # for period 2010 - 2019
    ax.fill_between(model_time_lagged[-6:], conf_interval_lower[-6:], conf_interval_upper[-6:], color="red", alpha=0.2)

    # Plot ERA5 data
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO (hPa)")

    # check if the p-value is les than 0.01
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
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f}), N = {no_ensemble_members}")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    # with the lag in the filename
    # and the current date
    # and the number of ensemble members#
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_lag_{lag}_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()


def plot_subplots_ensemble_members_and_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time):
    """
    Plot a series of subplots for each model with ensemble mean of all members and each of the ensemble members.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models. Should be all 12 models.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.

    Returns
    -------
    None
    """

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate over the models
    for i, model_name in enumerate(models):
        ax = axes[i]

        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # Initialize an empty list to store all ensemble members
        all_ensemble_members = []

        # Plot ensemble members
        for member in model_nao_anom:
            ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

        # Convert the list of all ensemble members to a NumPy array
        all_ensemble_members_array = np.array(all_ensemble_members)

        # Calculate the ensemble mean
        ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

        # count the number of ensemble members
        no_ensemble_members = len(all_ensemble_members)

        # Calculate the ACC score using the function pearsonr_score
        # for the short period
        acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, ensemble_mean, model_time, obs_time, "1966-01-01","2010-12-31")
        # for the long period
        acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, ensemble_mean, model_time, obs_time, "1966-01-01","2019-12-31")

        # Calculate the RPC score for the short period
        rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, model_time, "1966-01-01","2010-12-31")
        # for the long period
        rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, model_time, "1966-01-01","2019-12-31")

        # Calculate the 5-95% confidence intervals using the function calculate_confidence_intervals
        conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(all_ensemble_members_array)

        # Plot the ensemble mean with the ACC score in the legend
        ax.plot(model_time, ensemble_mean, color="red", label="DCPP-A")

        # Plot the 5-95% confidence intervals
        ax.fill_between(model_time[:-9], conf_interval_lower[:-9], conf_interval_upper[:-9], color="red", alpha=0.3)
        # for period 2010 - 2019
        ax.fill_between(model_time[-10:], conf_interval_lower[-10:], conf_interval_upper[-10:], color="red", alpha=0.2)

        # Plot ERA5 data
        ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
        ax.set_ylim([-10, 10])
        #ax.set_xlabel("Year")
        ax.set_ylabel("NAO (hPa)")

        # set the x-axis label for the bottom two subplots
        if i in [10, 11]:
            ax.set_xlabel("year")

        # check if the p-value is les than 0.01
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
        # the title will be formatted like this:
        # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
        ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f})", fontsize = 10)

        # x axis label
        #ax.set_xlabel("Year")
        
        # format the model name in the top left of the figure
        # with the number of ensemble members (N = ??) beneath it
        ax.text(0.02, 0.98, f"{model_name}", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.02, 0.85, f"N = {no_ensemble_members}", transform=ax.transAxes, ha="left", va="top")

        # Add the legend in the bottom right corner
        ax.legend(loc="lower right")

    # Adjust the layout
    plt.tight_layout()

    # set up a superior title
    # plt.suptitle("Ensemble members and their mean for each model", y=1.02)

    # Save the figure
    # In the plots_dir directory
    # with the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_members_and_mean_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()

# Define a function to plot the model subplots for the lagged //
# and var adjust data
def plot_subplots_ensemble_members_and_lagged_adjusted_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, lag=4):
    """
    Plot a series of subplots for each model with ensemble mean of all members, each of the ensemble members,
    and the lagged and variance-adjusted grand ensemble mean.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models. Should be all 12 models.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    lag : int, optional, default: 4
        The number of years to lag the grand ensemble mean by.

    Returns
    -------
    None
    """

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Iterate over the models
    for i, model_name in enumerate(models):
        ax = axes[i]

        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # Initialize an empty list to store all ensemble members
        all_ensemble_members = []

        # Plot ensemble members
        for member in model_nao_anom:
            ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

        # Convert the list of all ensemble members to a NumPy array
        all_ensemble_members_array = np.array(all_ensemble_members)

        # Calculate the ensemble mean
        ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

        # count the number of ensemble members
        no_ensemble_members = len(all_ensemble_members)

        # Apply lagging and variance adjustment to the grand ensemble mean
        lagged_ensemble_mean, model_time_lagged = process_lagged_ensemble_mean(ensemble_mean, model_time, lag)
        lagged_adjusted_ensemble_mean = adjust_variance(lagged_ensemble_mean)

        # Calculate the ACC score using the function pearsonr_score
        # for the short period
        acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean, model_time_lagged, obs_time, "1969-01-01","2010-12-31")
        # for the long period
        acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean, model_time_lagged, obs_time, "1969-01-01","2015-12-31")

        # Calculate the RPC score
        # for the short period
        rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, model_time, "1969-01-01","2010-12-31")
        # for the long period
        rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, model_time, "1969-01-01","2019-12-31")

        # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
        conf_interval_lower, conf_interval_upper = compute_rmse_confidence_intervals(obs_nao_anom, lagged_adjusted_ensemble_mean, obs_time, model_time_lagged)

        # Plot the lagged and adjusted ensemble mean with the ACC score in the legend
        ax.plot(model_time_lagged, lagged_adjusted_ensemble_mean[:-4], color="red", label="DCPP-A")

        # Plot the 5-95% confidence intervals
        # Plot the 5-95% confidence intervals for the short period
        ax.fill_between(model_time_lagged[:-5], conf_interval_lower[:-5], conf_interval_upper[:-5], color="red", alpha=0.3)
        # for period 2010 - 2019
        ax.fill_between(model_time_lagged[-6:], conf_interval_lower[-6:], conf_interval_upper[-6:], color="red", alpha=0.2)

        # Plot ERA5 data
        ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
        ax.set_ylim([-10, 10])
        ax.set_ylabel("NAO (hPa)")

        # set the x-axis label for the bottom two subplots
        if i in [10, 11]:
            ax.set_xlabel("year")
            
        # check if the p-value is les than 0.01
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
        # the title will be formatted like this:
        # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
        ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f})", fontsize = 10)

        # set the x-axis label
        #ax.set_xlabel("Year")
        
        # format the model name in the top left of the figure
        # with the number of ensemble members (N = ??) beneath it
        ax.text(0.02, 0.98, f"{model_name}", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.02, 0.85, f"N = {no_ensemble_members}", transform=ax.transAxes, ha="left", va="top")

        # Add the legend in the bottom right corner
        ax.legend(loc="lower right")

    # Adjust the layout
    plt.tight_layout()

    # set up a superior title
    # plt.suptitle("Ensemble members, lagged & variance-adjusted mean for each model", y=1.02)

    # Save the figure
    # In the plots_dir directory
    # with the lag in the filename
    # and the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_members_and_lagged_adjusted_mean_{lag}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()

def calculate_acc_by_ensemble_size(models, model_nao_anoms_by_model, model_times_by_model, obs_nao_anom, obs_time, step_size=2, num_samples=400):
    """
    Calculate ACC scores as the ensemble size increases and plot them.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    step_size : int, optional
        The step size for increasing the ensemble size (default is 1).
    num_samples : int, optional
        The number of random samples to take for each ensemble size (default is 1000).

    Returns
    -------
    None
    """

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Iterate over the models
    for model_name in models:
        model_nao_anom = model_nao_anoms_by_model[model_name]
        # initialize a time array
        model_time = model_times_by_model[model_name]

        # Add each member to the list of all ensemble members
        all_ensemble_members.extend(model_nao_anom)

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # The total number of ensemble members
    total_ensemble_members = len(all_ensemble_members)

    # Initialize lists to store the ensemble sizes and their corresponding ACC scores
    ensemble_sizes = []
    acc_scores_short = []
    acc_scores_long = []
    conf_ints_lower_short = []
    conf_ints_upper_short = []
    conf_ints_lower_long = []
    conf_ints_upper_long = []

    # Iterate over the ensemble sizes from 1 to the total number of ensemble members
    for ensemble_size in range(1, total_ensemble_members + 1, step_size):
        # Initialize a list to store the ACC scores for the current ensemble size
        current_short_acc_scores = []
        current_long_acc_scores = []

        # Draw num_samples random samples of size ensemble_size and calculate ACC for each sample
        for _ in range(num_samples):
            # Draw a random sample of ensemble members
            sample = resample(all_ensemble_members_array, n_samples=ensemble_size)

            # Calculate the ensemble mean
            ensemble_mean = np.mean(sample, axis=0)

            # Calculate the short period ACC score
            # and append to the list
            acc_score_short, _ = pearsonr_score(obs_nao_anom, ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")
            current_short_acc_scores.append(acc_score_short)

            # Calculate the long period ACC score
            # and append to the list
            acc_score_long, _ = pearsonr_score(obs_nao_anom, ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2019-12-31")
            current_long_acc_scores.append(acc_score_long)

        # Calculate the mean ACC score for the current ensemble size
        # for the short period
        mean_acc_score_short = np.mean(current_short_acc_scores)
        # for the long period
        mean_acc_score_long = np.mean(current_long_acc_scores)

        # Calculate the 5-95% confidence interval for the current ACC scores
        # for the short period
        conf_interval_short = np.percentile(current_short_acc_scores, [5, 95])
        # for the long period
        conf_interval_long = np.percentile(current_long_acc_scores, [5, 95])

        # Append the ensemble size, its corresponding mean ACC score, and confidence interval to the lists
        ensemble_sizes.append(ensemble_size)
        acc_scores_short.append(mean_acc_score_short)
        acc_scores_long.append(mean_acc_score_long)
        conf_ints_lower_short.append(conf_interval_short[0])
        conf_ints_upper_short.append(conf_interval_short[1])
        conf_ints_lower_long.append(conf_interval_long[0])
        conf_ints_upper_long.append(conf_interval_long[1])

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the ACC scores against the ensemble sizes
    # for the short period
    ax.plot(ensemble_sizes, acc_scores_short, color="red", label="Short period")
    # for the long period
    ax.plot(ensemble_sizes, acc_scores_long, color="blue", label="Long period")

    # Plot the 5-95% confidence intervals
    # for the short period
    ax.fill_between(ensemble_sizes, conf_ints_lower_short, conf_ints_upper_short, color="red", alpha=0.2)
    # for the long period
    ax.fill_between(ensemble_sizes, conf_ints_lower_long, conf_ints_upper_long, color="blue", alpha=0.2)

    # Save the figure
    # In the plots_dir directory
    # with the current date
    # and the step size in the filename
    # and the number of samples in the filename
    fig.savefig(os.path.join(plots_dir, f"nao_acc_by_ensemble_size_{datetime.now().strftime('%Y-%m-%d')}_{step_size}_{num_samples}.png"), dpi=300)

    ax.set_xlabel("Number of ensemble members")
    ax.set_ylabel("ACC score")
    ax.set_title("ACC score by ensemble size")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Show the figure
    plt.show()


