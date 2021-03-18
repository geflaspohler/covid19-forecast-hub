# Utility functions supporting experiments
import os
import warnings
import numpy as np
import pandas as pd
import netCDF4
# import xarray as xr
import subprocess
from datetime import datetime, timedelta
import collections
import itertools
import time
import sys
from functools import partial
from src.utils.general_util import printf, tic, toc

def pandas2file(df_to_file_func, out_file):
    """Writes pandas dataframe or series to file, makes file writable by all,
    creates parent directories with 777 permissions if they do not exist,
    and changes file group ownership to sched_mit_hill

    Args: df_to_file_func - function that writes dataframe to file when invoked,
        e.g., df.to_feather
      out_file - file to which df should be written
    """
    # Create parent directories with 777 permissions if they do not exist
    dirname = os.path.dirname(out_file)
    if dirname != '':
        os.umask(0); os.makedirs(dirname, exist_ok=True, mode=0o777)
    printf("Saving to "+out_file)
    tic(); df_to_file_func(out_file); toc()
    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)

def pandas2hdf(df, out_file, key="data", format="fixed"):
    """Write pandas dataframe or series to HDF; see pandas2file for other
    side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      key - key to use when writing to HDF
      format - format argument of to_hdf
    """
    pandas2file(partial(df.to_hdf, key=key, format=format, mode='w'), out_file)

def pandas2feather(df, out_file):
    """Write pandas dataframe or series to feather file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
    """
    pandas2file(df.to_feather, out_file)

def pandas2csv(df, out_file, index=False, header=True):
    """Write pandas dataframe or series to CSV file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      index - write index to file?
      header - write header row to file?
    """
    pandas2file(partial(df.to_csv, index=index, header=header), out_file)

def get_forecast_variable(gt_id):
    """Returns forecast variable name for the given ground truth id

    Args:
       gt_id: ground truth data string 
    """
    if gt_id.endswith("cm_death"):
        return "cm_death"
    if gt_id.endswith("in_death"):
        return "in_death"
    if gt_id.endswith("cm_case"):
        return "cm_case"
    if gt_id.endswith("in_case"):
        return "in_case"
    raise ValueError("Unrecognized gt_id "+gt_id)

def shift_df(df, shift=None, date_col='start_date', groupby_cols=['lat', 'lon'],
             rename_cols=True):
    """Returns dataframe with all columns save for the date_col and groupby_cols
    shifted forward by a specified number of days within each group

    Args:
       df: dataframe to shift
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting;
          if shift is None or shift == 0, original df is returned, unmodified
       date_col: (optional) name of datetime column
       groupby_cols: (optional) if all groupby_cols exist, shifting performed
          separately on each group; otherwise, shifting performed globally on
          the dataframe
       rename_cols: (optional) if True, rename columns to reflect shift
    """
    if shift is not None and shift != 0:
        # Get column names of all variables to be shifted
        # If any of groupby_cols+[date_col] do not exist, ignore error
        cols_to_shift = df.columns.drop(groupby_cols+[date_col], errors='ignore')
        # Function to shift data frame by shift and extend index
        shift_grp_df = lambda grp_df: grp_df[cols_to_shift].set_index(grp_df[date_col]).shift(int(shift),freq="D")
        if set(groupby_cols).issubset(df.columns):
            # Shift ground truth measurements for each group
            df = df.groupby(groupby_cols).apply(shift_grp_df).reset_index()
        else:
            # Shift ground truth measurements
            df = shift_grp_df(df).reset_index()
        if rename_cols:
            # Rename variables to reflect shift
            df.rename(columns=dict(
                list(zip(cols_to_shift, [col+"_shift"+str(shift) for col in cols_to_shift]))),
                      inplace=True)
    return df

def load_measurement(file_name, mask_df=None, shift=None):
    """Loads measurement data from a given file name and returns as a dataframe

    Args:
       file_name: name of HDF5 file from which measurement data will be loaded
       mask_df: (optional) mask dataframe of the form returned by subsetmask();
         if specified, returned dataframe will be restricted to those lat, lon
         pairs indicated by the mask
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting
    """
    # Load ground-truth data
    df = pd.read_hdf(file_name, 'data')

    # Convert to dataframe if necessary
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()
    # Replace multiindex with start_date, lat, lon columns if necessary
    if isinstance(df.index, pd.MultiIndex):
        df.reset_index(inplace=True)
    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        df = subsetmask(df, mask_df)

    # Return dataframe with desired shift
    return shift_df(df, shift=shift, date_col='start_date', groupby_cols=['lat', 'lon'])

# TODO: could adapt
def get_first_year(data_id):
    """Returns first year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or ground truth identifier
         accepted by get_ground_truth
    """
    if data_id.startswith("global"):
        return 2011
    if data_id.endswith("precip"):
        return 1948
    if data_id.startswith("nmme"):
        return 1982
    if data_id.endswith("tmp2m") or data_id.endswith("tmin") or data_id.endswith("tmax"):
        return 1979
    if "sst" in data_id or "icec" in data_id:
        return 1981
    if data_id.endswith("mei"):
        return 1979
    if data_id.endswith("mjo"):
        return 1974
    if data_id.endswith("sce"):
        return 1966
    if "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        return 1948
    if ("slp" in data_id or "pr_wtr" in data_id or "rhum" in data_id or
        "pres" in data_id or "pevpr" in data_id):
        return 1948
    if data_id.startswith("subx_cfsv2"):
        return 1999
    raise ValueError("Unrecognized data_id "+data_id)

def get_last_year(data_id):
    """Returns last year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or
         ground truth identifier accepted by get_ground_truth
    """
    return 2019

# TODO: need to adapt
def get_ground_truth(gt_id, mask_df=None, shift=None):
    """Returns ground truth data as a dataframe

    Args:
       gt_id: string identifying which ground-truth data to return;
         valid choices are "global_precip", "global_tmp2m", "us_precip",
         "contest_precip", "contest_tmp2m", "contest_tmin", "contest_tmax",
         "contest_sst", "contest_icec", "contest_sce",
         "pca_tmp2m", "pca_precip", "pca_sst", "pca_icec", "mei", "mjo",
         "pca_hgt_{}", "pca_uwnd_{}", "pca_vwnd_{}",
         "pca_sst_2010", "pca_icec_2010", "pca_hgt_10_2010",
         "contest_rhum.sig995", "contest_pres.sfc.gauss", "contest_pevpr.sfc.gauss",
         "wide_contest_sst", "wide_hgt_{}", "wide_uwnd_{}", "wide_vwnd_{}",
         "us_tmp2m", "us_tmin", "us_tmax", "us_sst", "us_icec", "us_sce",
         "us_rhum.sig995", "us_pres.sfc.gauss", "us_pevpr.sfc.gauss"
       mask_df: (optional) see load_measurement
       shift: (optional) see load_measurement
    """
    gt_file = os.path.join("data", "dataframes", "gt-"+gt_id+"-14d.h5")
    printf(f"Loading {gt_file}")
    if gt_id.endswith("mei"):
        # MEI does not have an associated number of days
        gt_file = gt_file.replace("-14d", "")
    if gt_id.endswith("mjo"):
        # MJO is not aggregated to a 14-day period
        gt_file = gt_file.replace("14d", "1d")
    return load_measurement(gt_file, mask_df, shift)

# TODO: could adapt 
def get_ground_truth_unaggregated(gt_id, mask_df=None, shifts=None):
    """Returns daily ground-truth data as a dataframe, along with one column
    per shift in shifts
    """
    first_year = get_first_year(gt_id)
    last_year = get_last_year(gt_id)
    gt_file = os.path.join("data", "dataframes",
                           "gt-"+gt_id+"-1d-{}-{}.h5".format(
                               first_year, last_year))
    gt = load_measurement(gt_file, mask_df)
    if shifts is not None:
        measurement_variable = get_measurement_variable(gt_id)
        for shift in shifts:
            # Shift ground truth measurements by shift for each lat lon and extend index
            gt_shift = gt.groupby(['lat', 'lon']).apply(
                lambda df: df[[measurement_variable]].set_index(df.start_date).shift(shift,freq="D")).reset_index()
            # Rename variable to reflect shift
            gt_shift.rename(columns={measurement_variable: measurement_variable +
                                     "_shift"+str(shift)}, inplace=True)
            # Merge into the main dataframe
            gt = pd.merge(gt, gt_shift, on=["lat", "lon", "start_date"], how="outer")
    return gt

def in_month_day_range(test_datetimes, target_datetime, margin_in_days=0):
    """For each test datetime object, returns whether month and day is
    within margin_in_days days of target_datetime month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       test_datetimes: pandas Series of datetime.datetime objects
       target_datetime: target datetime.datetime object (must not be Feb. 29!)
       margin_in_days: number of days allowed between target
         month and day and test date month and day
    """
    # Compute target day of year in a year that is not a leap year
    non_leap_year = 2017
    target_day_of_year = pd.Timestamp(target_datetime.
                                      replace(year=non_leap_year)).dayofyear
    # Compute difference between target and test days of year
    # after adjusting leap year days of year to match non-leap year days of year;
    # This has the effect of treating Feb. 29 as the same date as Feb. 28
    leap_day_of_year = 60
    day_delta = test_datetimes.dt.dayofyear
    day_delta -= (test_datetimes.dt.is_leap_year & (day_delta >= leap_day_of_year))
    day_delta -= target_day_of_year
    # Return true if test day within margin of target day when we account for year
    # wraparound
    return ((np.abs(day_delta) <= margin_in_days) |
            ((365 - margin_in_days) <= day_delta) |
            (day_delta <= (margin_in_days - 365)))

def month_day_subset(data, target_datetime, margin_in_days=0,
                     start_date_col="start_date"):
    """Returns subset of dataframe rows with start date month and day
    within margin_in_days days of the target month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       data: pandas dataframe with start date column containing datetime values
       target_datetime: target datetime.datetime object providing target month
         and day (will treat Feb. 29 like Feb. 28)
       start_date_col: name of start date column
       margin_in_days: number of days allowed between target
         month and day and start date month and day
    """
    if (target_datetime.day == 29) and (target_datetime.month == 2):
        target_datetime = target_datetime.replace(day = 28)
    return data.loc[in_month_day_range(data[start_date_col], target_datetime,
                                       margin_in_days)]
    # return data.loc[(data[start_date_col].dt.month == target_datetime.month) &
    #                (data[start_date_col].dt.day == target_datetime.day)]

# TODO: should adapt 
def load_forecast_from_file(file_name, mask_df=None):
    """Loads forecast data from file and returns as a dataframe

    Args:
       file_name: HDF5 file containing forecast data
       forecast_variable: name of forecasted variable (see get_forecast_variable)
       target_horizon: target forecast horizon
         ("34w" for 3-4 weeks or "56w" for 5-6 weeks)
       mask_df: (optional) see load_measurement
    """
    # Load forecast dataframe
    forecast = pd.read_hdf(file_name)

    # PY37
    if 'start_date' in forecast.columns:
        forecast.start_date = pd.to_datetime(forecast.start_date)
    if 'target_date' in forecast.columns:
        forecast.target_date = pd.to_datetime(forecast.target_date)

    if mask_df is not None:
        # Restrict output to requested lat, lon pairs
        forecast = subsetmask(forecast, mask_df)
    return forecast

# TODO: need to adapt
def get_forecast(forecast_id, mask_df=None, shift=None):
    """Returns forecast data as a dataframe

    Args:
       forecast_id: forecast identifier of the form "{1}-{2}-{3}"
         where {1} is the forecast name in {nmme, nmme0, subx_cfsv2},
         {2} is the forecast variable (see get_forecast_variable),
         and {3} is the target forecast horizon in {34w, 56w}
       mask_df: (optional) see load_measurement
       shift: (optional) number of days by which ground truth measurements
         should be shifted forward; date index will be extended upon shifting
    """
    forecast_file = os.path.join("data", "dataframes",
                               forecast_id_to_fname_mapping[forecast_id]+".h5")
    printf(f"Loading {forecast_file}")
    forecast = load_forecast_from_file(forecast_file, mask_df)

    if forecast_id.startswith("nmme0"):
        models = ['cancm3_0', 'cancm4_0', 'ccsm4_0', 'gfdl_0', 'gfdl-flor-a_0', 'gfdl-flor-b_0', 'cfsv2_0']
        forecast['nmme0_wo_ccsm3_nasa'] = forecast[models].mean(axis=1)
        forecast.drop(models, axis=1, inplace=True)
    elif forecast_id.startswith("nmme"):
        models = ['cancm3', 'cancm4', 'ccsm4', 'gfdl', 'gfdl-flor-a', 'gfdl-flor-b', 'cfsv2']
        forecast['nmme_wo_ccsm3_nasa'] = forecast[models].mean(axis=1)
        forecast.drop(models, axis=1, inplace=True)

    return shift_df(forecast, shift=shift,
                    groupby_cols=['lat', 'lon'])


def get_contest_id(gt_id, horizon):
    """Returns contest task identifier string for the given ground truth
    identifier and horizon identifier

    Args:
       gt_id: ground truth data string 
          belonging to {"in_death", "cm_death", "in_case", "cm_case"}
       horizon: string in {"1w","2w","3w","4w"} indicating target
          horizon for prediction
    """
    # Map gt_id to standard contest form
    if gt_id.endswith("cm_death") or gt_id == "cm_death":
        gt_id = "cm_death"
    if gt_id.endswith("in_death") or gt_id == "in_death":
        gt_id = "in_death"
    if gt_id.endswith("cm_case") or gt_id == "cm_case":
        gt_id = "cm_case"
    if gt_id.endswith("in_case") or gt_id == "in_case":
        gt_id = "in_case"
    else:
        raise ValueError("Unrecognized gt_id "+gt_id)

    # Map horizon to standard contest form
    if horizon == "1w" or horizon == "week1":
        horizon = "1w"
    elif horizon == "2w" or horizon == "week2":
        horizon = "2w"
    elif horizon == "3w" or horizon == "week3":
        horizon = "3w"
    elif horizon == "4w" or horizon == "week4":
        horizon = "4w"
    else:
        raise ValueError("Unrecognized horizon "+horizon)
    # Return contest task identifier
    return gt_id+"_"+horizon

# TODO: may want to adapt
def get_contest_template_file(gt_id, horizon):
    """Returns name of contest template netcdf file for a given ground truth
    descriptor and horizon for prediction

    Args:
       gt_id: see get_contest_id
       horizon: see get_contest_id
    """
    return os.path.join("data", "fcstrodeo_nctemplates",
                        get_contest_id(gt_id, horizon)+"_template.nc")

def get_deadline_delta(target_horizon):
    """Returns number of days between official contest submission deadline date
    and start date of target period
    (14 for week 3-4 target, as it's 14 days away,
    28 for week 5-6 target, as it's 28 days away)

    Args:
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
    """
    if target_horizon == "1w":
        deadline_delta = 7
    elif target_horizon == "2w":
        deadline_delta = 14
    elif target_horizon == "3w":
        deadline_delta = 21
    elif target_horizon == "4w":
        deadline_delta = 28
    else:
        raise ValueError("Unrecognized target_horizon " + target_horizon)
    return deadline_delta

def get_forecast_delta(target_horizon, days_early=1):
    """Returns number of days between forecast date and start date of target period
    (deadline_delta + days_early, as we submit early)

    Args:
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
       days_early: how many days early is forecast submitted?
    """
    return get_deadline_delta(target_horizon) + days_early

# TODO: should adapt 
def get_measurement_lag(data_id):
    """Returns the number of days of lag (e.g., the number of days over
    which a measurement is aggregated plus the number of days
    late that a measurement is released) for a given ground truth data
    measurement

    Args:
       data_id: forecast identifier beginning with "subx-cfsv2" or
         ground truth identifier accepted by get_ground_truth
    """
    # Every measurement is associated with its start date, and measurements
    # are aggregated over one or more days, so, on a given date, only the measurements
    # from at least aggregation_days ago are fully observed.
    # Most of our measurements require 14 days of aggregation
    aggregation_days = 14
    # Some of our measurements are also released a certain number of days late
    days_late = 0
    if data_id.endswith("mjo"):
        # MJO uses only a single day of aggregation and is released one day late
        aggregation_days = 1
        days_late = 1
    elif "sst" in data_id:
        # SST measurements are released one day late
        days_late = 1
    elif data_id.endswith("mei"):
        # MEI measurements are released at most 30 days late
        # (since they are released monthly) but are not aggregated
        aggregation_days = 0
        days_late = 30
    elif "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        # Wind / hgt measurements are released one day late
        days_late = 1
    elif "icec" in data_id:
        days_late = 1
    elif ("slp" in data_id or "pr_wtr.eatm" in data_id or "rhum.sig995" in data_id or
          "pres.sfc.gauss" in data_id or "pevpr.sfc.gauss" in data_id):
        # NCEP/NCAR measurements are released one day late
        days_late = 1
    elif data_id.startswith("subx_cfsv2"):
        # No aggregation required for subx cfsv2 forecasts
        aggregation_days = 0
    return aggregation_days + days_late

def get_start_delta(target_horizon, data_id):
    """Returns number of days between start date of target period and start date
    of observation period used for prediction. One can subtract this number
    from a target date to find the last viable training date.

    Args:
       target_horizon: see get_forecast_delta()
       data_id: see get_measurement_lag()
    """
    if data_id.startswith("nmme"):
        # Special case: NMME is already shifted to match target period
        return None
    return get_measurement_lag(data_id) + get_forecast_delta(target_horizon)

def get_target_date(deadline_date_str, target_horizon):
    """Returns target date (as a datetime object) for a given deadline date
    and target horizon

    Args:
       deadline_date_str: string in YYYYMMDD format indicating official
          contest submission deadline (note: we often submit a day before
          the deadline, but this variable should be the actual deadline)
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
    """
    # Get deadline date datetime object
    deadline_date_obj = datetime.strptime(deadline_date_str, "%Y%m%d")
    
    # Compute target date object
    return deadline_date_obj + timedelta(days=get_deadline_delta(target_horizon))

def df_merge(left, right, on=["lat", "lon", "start_date"], how="outer"):
    """Returns merger of pandas dataframes left and right on 'on'
    with merge type determined by 'how'. If left == None, simply returns right.
    """
    if left is None:
        return right
    else:
        return pd.merge(left, right, on=on, how=how)

def year_slice(df, first_year = None, date_col = 'start_date'):
    """Returns slice of df containing all rows with df[date_col].dt.year >= first_year;
    returns df if first_year is None
    """
    if first_year is None:
        return df
    ###years = pd.to_datetime(df[date_col]).dt.year
    #years = df[date_col].dt.year
    if first_year <= df[date_col].min().year:
        # No need to slice
        return df
    return df[df[date_col] >= f"{first_year}-01-01"]

# TODO: could remove
def get_combined_data_dir(model="default"):
    """Returns path to combined data directory for a given model

    Args:
      model: "default" for default versions of data or name of model for model-
        specific versions of data
    """
    if model == 'default':
        cache_dir = os.path.join('data', 'combined_dataframes')
    else:
        cache_dir = os.path.join('models', model, 'data')
    return cache_dir

# TODO: could remove
def get_combined_data_filename(file_id, gt_id, target_horizon,
                               model="default"):
    """Returns path to directory or file of interest

    Args:
      file_id: string identifier defining data file of interest;
        valid values include
        {"lat_lon_date_data","lat_lon_data","date_data","all_data","all_data_no_NA"}
      gt_id: "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
      target_horizon: "34w" or "56w"
      model: "default" for default versions of data or name of model for model
        specific versions of data
    """
    combo_dir = get_combined_data_dir(model)
    suffix = "feather"
    return os.path.join(combo_dir, "{}-{}_{}.{}".format(file_id,gt_id,
                                                        target_horizon,suffix))

# TODO: could remove or modify
def load_combined_data(file_id, gt_id,
                       target_horizon,
                       model="default",
                       target_date_obj=None,
                       columns=None):
    """Loads and returns a previously saved combined data dataset

    Args:
      file_id: string identifier defining data file of interest;
        valid values include
        {"lat_lon_date_data","lat_lon_data","date_data","all_data","all_data_no_NA"}
      gt_id: "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
      target_horizon: "34w" or "56w"
      model: "default" for default versions of data or name of model for model
        specific versions of data
      target_date_obj: if not None, print any columns in loaded data that are
        missing on this date in datetime format
      columns: list of column names to load or None to load all
    Returns:
       Loaded dataframe
    """
    data_file = get_combined_data_filename(
        file_id, gt_id, target_horizon, model=model)

    # ---------------
    # Read data_file from disk
    # ---------------
    col_arg = "all columns" if columns is None else columns
    printf(f"Reading {col_arg} from file {data_file}")
    tic()
    data = pd.read_feather(data_file, columns=columns)
    toc()
    # Print any data columns missing on target date
    if target_date_obj is not None:
        print_missing_cols_func(data, target_date_obj, True)

    return data

# TODO: could modify
def print_missing_cols_func(df, target_date_obj, print_missing_cols):
    if print_missing_cols is True:
        missing_cols_in_target_date = df.loc[df["start_date"] == target_date_obj].isnull().any()
        if sum(missing_cols_in_target_date) > 0:
            printf("")
            printf("There is missing data for target_date. The following variables are missing: {}"\
                            .format(df.columns[missing_cols_in_target_date].tolist()))
            printf("")

# TODO: could adapt 
def get_id_name(gt_id, None_ok=False):
    if gt_id in ["tmp2m", "temp", "contest_tmp2m", "contest_temp"]:
        return "contest_tmp2m"
    elif gt_id in ["precip", "prate", "contest_prate", "contest_precip"]:
        return "contest_precip"
    elif gt_id in ["us_tmp2m", "us_temp"]:
        return "us_tmp2m"
    elif gt_id in ["us_prate", "us_precip"]:
        return "us_precip"
    elif gt_id is None and None_ok:
        return None
    else:
        raise Exception(f"gt_id not recognized. Value passed: {gt_id}.")

# TODO: could adapt
def get_th_name(target_horizon, None_ok=False):
    if target_horizon in ["34", "34w"]:
        return "34w"
    elif target_horizon in ["56", "56w"]:
        return "56w"
    elif target_horizon is None and None_ok:
        return None
    raise Exception(f"target_horizon not recognized. Value passed: {target_horizon}.")
