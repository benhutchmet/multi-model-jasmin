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
from scipy.stats import pearsonr

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

    # Set the type for the time variable
    obs_time = obs_time.astype("datetime64[Y]")

    # Process the obs data from Pa to hPa
    obs_nao_anom = obs_psl[:, 0, 0] / 100

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

# Usage for debugging
# # Assuming you have already calculated the correlation coefficient (ACC)
# acc_score = ... # Replace this with the actual value

# # Assuming the individual forecast members are stored in a list of lists or a 2D numpy array
# forecast_members = ... # Replace this with the actual data

# # Calculate the RPC
# rpc = calculate_rpc(acc_score, forecast_members)

# # Print the RPC value
# print(rpc)

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
def process_lagged_ensemble_mean(data, time_array, lag=4):
    """
    Lag the input data by a specified number of years and create a new time array
    corresponding to the lagged data. This function is useful for processing ensemble
    mean data, such as the North Atlantic Oscillation (NAO) time series.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input data to be lagged, typically an ensemble mean time series.
    time_array : numpy.ndarray
        The time array corresponding to the input data.
    lag : int, optional, default: 4
        The number of years to lag the data by.
    
    Returns
    -------
    lagged_data_mean : numpy.ndarray
        The lagged ensemble mean data.
    model_time_lagged : numpy.ndarray
        The new time array corresponding to the lagged data.
    """
    def lagged_ensemble_mean(data, lag):
        # Initialize an empty array for the lagged ensemble mean
        lagged_mean = np.empty(len(data) - lag + 1)

        # Calculate the lagged ensemble mean for each year
        for i in range(len(data) - lag + 1):
            lagged_mean[i] = np.mean(data[i:i + lag])

        return lagged_mean

    # Calculate the lagged ensemble mean for the data
    lagged_data_mean = lagged_ensemble_mean(data, lag)

    # Calculate the corresponding model_time for the lagged data
    model_time_lagged = time_array[:-lag][lag-1:]

    return lagged_data_mean, model_time_lagged

# Example usage
# lagged_adjusted_var_mean, model_time_lagged = process_lagged_ensemble_mean(adjusted_var_model_nao_anom_raw, model_time_raw, lag=4)

# Function to adjust the variance of the ensemble
# Used once the no. of ensemble members has been 4x
# through the lagging process
def adjust_variance(model_time_series):
    """
    Adjust the variance of an ensemble mean time series by dividing each value
    by the ensemble mean standard deviation. This function is used after the
    number of ensemble members has been increased by a factor of 4 through the
    lagging process.
    
    Parameters
    ----------
    model_time_series : numpy.ndarray
        The ensemble mean time series data to adjust the variance for.
    
    Returns
    -------
    model_time_series_var_adjust : numpy.ndarray
        The adjusted time series data with variance scaled by the ensemble mean standard deviation.
    """
    # Calculate the standard deviation for the ensemble mean time series
    model_std = np.std(model_time_series)

    # Adjust the variance of the time series by dividing the value by the ensemble mean standard deviation
    model_time_series_var_adjust = model_time_series / model_std

    return model_time_series_var_adjust


# Example usage - included for debugging
# obs_data = np.array([1, 2, 3, 4, 5])
# model_data = np.array([2, 4, 6, 8, 10])

# obs_time_series_var_adjust, model_time_series_var_adjust = adjust_variance(obs_data, model_data)
# print(obs_time_series_var_adjust)
# print(model_time_series_var_adjust)

# function for calculating the RMSE and 5-95% uncertainty intervals for the variance adjusted output
def compute_rmse_confidence_intervals(obs_nao_anoms, adjusted_lagged_model_nao_anoms, lower_bound=5, upper_bound=95):
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
    # Compute the root-mean-square error (RMSE) between the ensemble mean and the observations
    rmse = np.sqrt(np.mean((obs_nao_anoms - adjusted_lagged_model_nao_anoms)**2, axis=0))

    # Calculate the z-scores corresponding to the lower and upper bounds
    z_score_lower = np.percentile(rmse, lower_bound)
    z_score_upper = np.percentile(rmse, upper_bound)

    # Calculate the 5% and 95% confidence intervals using the RMSE
    conf_interval_lower = obs_nao_anoms - z_score_upper * rmse
    conf_interval_upper = obs_nao_anoms + z_score_upper * rmse

    return conf_interval_lower, conf_interval_upper

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

    # Calculate the grand ensemble mean using the new method
    grand_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # Calculate the ACC score using the function pearsonr_score
    acc_score, p_value = pearsonr_score(obs_nao_anom, grand_ensemble_mean, list(model_times_by_model.values())[0], obs_time, "1966-01-01","2010-12-31")

    # Calculate the RPC score using the function calculate_rpc
    rpc = calculate_rpc(acc_score, all_ensemble_members_array)

    # Calculate the 5-95% confidence intervals using the two functions options
    # First calculate_confidence_intervals_sd
    # Then calculate_confidence_intervals
    conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(all_ensemble_members_array)

    # Plot the grand ensemble mean with the ACC score in the legend
    ax.plot(list(model_times_by_model.values())[0], grand_ensemble_mean, color="red", label=f"DCPP-A (ACC: {acc_score:.2f})")


    # Plot the 5-95% confidence intervals
    ax.fill_between(list(model_times_by_model.values())[0], conf_interval_lower, conf_interval_upper, color="red", alpha=0.2, label="5-95% confidence interval")

    # Plot ERA5 data
    ax.plot(obs_time[3:], obs_nao_anom[3:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # Set the title with the ACC and RPC scores
    ax.set_title(f"NAO ensemble mean and individual members (ACC: {acc_score:.2f}, RPC: {rpc:.2f})")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    fig.savefig(os.path.join(plots_dir, "nao_ensemble_mean_and_individual_members.png"), dpi=300)

    # Show the figure
    plt.show()

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

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # Calculate the grand ensemble mean using the new method
    grand_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # Apply lagging and variance adjustment to the grand ensemble mean
    lagged_grand_ensemble_mean, model_time_lagged = process_lagged_ensemble_mean(grand_ensemble_mean, list(model_times_by_model.values())[0], lag)
    lagged_adjusted_grand_ensemble_mean = adjust_variance(lagged_grand_ensemble_mean)

    # Also just apply the variance adjustment to the grand ensemble mean
    adjusted_grand_ensemble_mean = adjust_variance(grand_ensemble_mean)

    # Calculate the ACC score using the function pearsonr_score with the lagged and adjusted grand ensemble mean
    # For the skillful period
    acc_score, p_value = pearsonr_score(obs_nao_anom, lagged_adjusted_grand_ensemble_mean, model_time_lagged, obs_time, "1966-01-01", "2010-12-31") 

    # Calculate the RPC score using the function calculate_rpc
    rpc = calculate_rpc(acc_score, all_ensemble_members_array)

    # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
    conf_interval_lower, conf_interval_upper = compute_rmse_confidence_intervals(obs_nao_anom, lagged_adjusted_grand_ensemble_mean)

    # Plot the grand ensemble mean with the ACC score in the legend
    ax.plot(model_time_lagged, lagged_adjusted_grand_ensemble_mean, color="red", label=f"DCPP-A lagged + var. adjust (ACC: {acc_score:.2f})")

    # Plot the grand ensemble mean variance adjusted only
    ax.plot(model_time_lagged, adjusted_grand_ensemble_mean, color="orange", alpha=0.4, linestyle="--",label="DCPP-A var. adjust")

    # Plot the 5-95% confidence intervals
    ax.fill_between(model_time_lagged, conf_interval_lower, conf_interval_upper, color="red", alpha=0.2, label="5-95% confidence interval")

    # Plot ERA5 data
    ax.plot(obs_time[3:], obs_nao_anom[3:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # Set the title with the ACC and RPC scores
    ax.set_title(f"NAO ensemble mean (lagged and adjusted) and individual members (ACC: {acc_score:.2f}, RPC: {rpc:.2f})")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    fig.savefig(os.path.join(plots_dir, "nao_ensemble_mean_and_individual_members_lagged_and_adjusted.png"), dpi=300)

    # Show the figure
    plt.show()
