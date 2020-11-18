import numpy as np
import pandas as pd
import os
import pdb
from src.utils.general_util import printf, tic, toc
from datetime import datetime 
from file_util import *
import glob


"""
Dictionaries to convert between task id and task labels
"""
gt_id2label= {
    "cumm_death": "cum death",
    "incd_death": "inc death",
    "incd_case": "inc case"
}
horiz2label= {
    "1w": "1 wk ahead",
    "2w": "2 wk ahead",
    "3w": "3 wk ahead",
    "4w": "4 wk ahead"
}

# Map day name to Pandas integer
dow_to_int = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

def get_pred_file(gt_id, horizon, quantile): 
    return os.path.join("data", "dataframes", f"pred-{gt_id}-{horizon}-q{str(quantile).replace('.', '')}.h5")

def get_task_label(gt_id, horizon):
    if (gt_id not in gt_id2label) or (horizon not in horiz2label):
        raise ValueError(f"Task {gt_id}, {horizon} not recognized.")

    return f"{horiz2label[horizon]} {gt_id2label[gt_id]}"

def get_all_models():
    """
    Returns a list containing all avaliable model names as strings
    """
    models = []
    template = os.path.join('data-processed','*')
    # Number of directories above model name
    model_name_index = 1
    for model_dir in glob.glob(template):
        mn = model_dir.split(os.path.sep)[model_name_index]
        models.append(mn)

    return list(set(models)) # Remove any possible duplicate models 

def get_persistant_models(df):
    pred_dates = df.index.get_level_values('target_end_date').unique().sort_values()
    last_date = pred_dates[-1]

    persistant_model_dates = {}
    model_dates = {}
    # Go through dates in reverse order
    # initial_models = df.index.get_level_values('model').unique()
    initial_models = df.loc[df.index.get_level_values('target_end_date') == last_date].index.get_level_values('model').unique()
    models = initial_models.copy()
    initial_models = sorted(list(initial_models))
    for date in pred_dates[::-1]:
        m = df.loc[df.index.get_level_values('target_end_date') == date].index.get_level_values('model').unique()
        model_dates[date] = sorted(list(m))

        models = models.intersection(m)
        persistant_model_dates[date] = sorted(list(models))
    
    return persistant_model_dates, initial_models, model_dates

def get_model_predictions(gt_id, horizon, location, quantile, load_df=True):
    pred_file = get_pred_file(gt_id, horizon, quantile)

    if os.path.exists(pred_file) and load_df:
        df = pd.read_hdf(pred_file)
    else:
        printf(f"File {pred_file} not avaliable. Reading data from source and generating.")
        df = load_predictions_from_file(gt_id, horizon, quantile)
        # Save to file
        pandas2hdf(df, pred_file)

    return load_measurement(df, location)

def load_predictions_from_file(gt_id, horizon, quantile):
    """
    Load model predictions for a specific task
    
    Args:
       gt_id: ground truth task id
       horizon: task horizon
       quantile: task quantile, -1 for point quantile

    """
    # Get task label in covidhub standard 
    task_label = get_task_label(gt_id, horizon)
    
    # Get difference between forecast date and target date for checking legality 
    start_delta = timedelta(days=get_start_delta(horizon, gt_id))

    # Skip over certain models that should not be compared
    models_to_exclude = [""]

    # Initialized prediction dataframe
    cols = ['location', 'model', 'forecast_date', 'target_end_date', 'value']
    pred_df = pd.DataFrame(columns=cols)

    # TODO: Check generality
    template = os.path.join('data-processed','*')
    count = 0

    # Number of directories above model name
    model_name_index = 1
    for model_dir in glob.glob(template):
        mn = model_dir.split(os.path.sep)[model_name_index]
        if mn in models_to_exclude:
            continue
        count += 1
        
        file_template = os.path.join(model_dir, f'*-{mn}.csv')
        for fh in glob.glob(file_template):
            df = pd.read_csv(fh)
            df.loc[df['type'] == "point", 'quantile'] = -1 # Add quantile code to point-queries
            df['location'] = df['location'].astype(str).str.lstrip("0") # Remove trailing zeros before location string
            # task_df = df.query(f"target == '{task_label}' and quantile == '{quantile}' and location == '{location}'").copy()
            task_df = df.query(f"target == '{task_label}' and quantile == '{quantile}'").copy()

            # Check if the forecast is legal
            target_dt = [datetime.strptime(x, '%Y-%m-%d') for x in task_df['target_end_date']]
            forecast_dt = [datetime.strptime(x, '%Y-%m-%d') + start_delta for x in task_df['forecast_date']]
            illegal_dt = [target_dt[i] < forecast_dt[i] for i in range(len(target_dt))]

            if any(illegal_dt):
                printf(f"Illegal forecast for task {task_label}.")
                # continue; for now, allow it within the df, don't fully understand

            task_df['model'] = mn
            pred_df = pd.concat([pred_df, task_df], join="inner", ignore_index=True)


    # Remove duplicate predictions from the same model for the same target_end_date and location; keep most recent forecast
    pred_df.sort_values(by='forecast_date', inplace=True)
    pred_df_nodup = pred_df.loc[~pred_df.duplicated(subset=['location', 'model', 'target_end_date'], keep='last')].copy()
    pred_df_nodup.drop(['forecast_date'], axis='columns', inplace=True)
    return pred_df_nodup.set_index(['model', 'location', 'target_end_date'])

def get_start_delta(target_horizon, data_id):
    """Returns number of days between start date of target period and start date
    of observation period used for prediction. One can subtract this number
    from a target date to find the last viable training date.

    Args:
       target_horizon: see get_forecast_delta()
       data_id: see get_measurement_lag()
    """
    return get_measurement_lag(data_id) + get_forecast_delta(target_horizon)

def get_deadline_delta(target_horizon):
    """Returns number of days between official contest submission deadline date
    and start date of target period

    Args:
       target_horizon:  # TODO
    """
    # Assuming that teams can make a forecast up to Monday for what will happen on Saturday 
    if target_horizon == "1w":
        deadline_delta = 5
    elif target_horizon == "2w":
        deadline_delta = 12
    elif target_horizon == "3w":
        deadline_delta = 19
    elif target_horizon == "4w":
        deadline_delta = 26
    else:
        raise ValueError("Unrecognized target_horizon " + target_horizon)
    return deadline_delta

def get_forecast_delta(target_horizon, days_early=0):
    """Returns number of days between forecast date and start date of target period
    (deadline_delta + days_early, as we submit early)

    Args:
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
       days_early: how many days early is forecast submitted?
    """
    return get_deadline_delta(target_horizon) + days_early

def get_measurement_lag(data_id):
    """Returns the number of days of lag (e.g., the number of days over
    which a measurement is aggregated plus the number of days
    late that a measurement is released) for a given ground truth data
    measurement

    Args:
       data_id: ground truth identifier accepted by get_ground_truth
    """
    # Every measurement is associated with its start date, and measurements
    # are aggregated over one or more days, so, on a given date, only the measurements
    # from at least aggregation_days ago are fully observed.
    # Most of our measurements require 14 days of aggregation
    aggregation_days = 0
    # Some of our measurements are also released a certain number of days late
    days_late = 0
    return aggregation_days + days_late

    # TODO: need to adapt

def get_data_range(gt_id, location):
    df = get_ground_truth(gt_id, location)
    return df.index[0][1], df.index[-1][1] # Get the second value in the multindex

def get_ground_truth(gt_id, location, shift=None, load_df=True):
    """Returns ground truth data as a dataframe

    Args:
       gt_id: string identifying which ground-truth data to return;
         valid choices are "cumm_death", "incd_death", "cumm_case", "incd_case" 
       location: location fips code
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting
    """
    # printf(f"Loading {gt_id}")
    gt_file = get_ground_truth_file(gt_id)
    if os.path.exists(gt_file) and load_df:
        df = pd.read_hdf(gt_file)
    else:
        printf(f"File {gt_file} not avaliable. Reading data from source and generating.")
        df_raw = get_jhu_raw(gt_id)
        df = get_gt_from_jhu(gt_id, df_raw)

         # Save to file
        pandas2hdf(df, gt_file)
    return load_measurement(df, location)

def load_measurement(df, location, shift=None):
    """Loads measurement data from a given file name and returns as a dataframe

    Args:
       df: df containing raw data
       location: location fips code
       shift: (optional) Number of days by which ground truth measurements
          should be shifted forward; date index will be extended upon shifting
    """
    # Convert to dataframe if necessary
    if not isinstance(df, pd.DataFrame):
        df = df.to_frame()

    # Restrict output to requested lat, lon pairs

    if isinstance(df.index, pd.MultiIndex):
        df_ret = df.loc[df.index.get_level_values('location') == location].dropna()
    else:
        df_ret = df.loc[df['location'] == location].dropna()

    # Return dataframe with desired shift
    # return shift_df(df_ret, shift=shift, date_col='start_date', groupby_cols=['lat', 'lon'])
    # TODO: is shifting useful here
    return df_ret

def get_fips_codes():
    fips_file = os.path.join('data-locations', 'locations.csv')
    fips_codes = pd.read_csv(fips_file)
    return fips_codes

def get_gt_from_jhu(gt_id, df, save_to_file=True):
    """
    Prepare ground truth data, incident and cummulative deaths, daily 
    """
    # aggregate by state and nationally
    state_agg = df.groupby(['Province_State']).sum()
    us_nat = df.groupby(['Country_Region']).sum()
    df_state_nat = state_agg.append(us_nat)

    drop_cols = ['UID', 'code3', 'FIPS', 'Lat', 'Long_', 'Population', 'iso2', 'iso3', 'Admin2']
    # drop unnecessary columns
    df_truth = df_state_nat.drop(columns=drop_cols, axis=1, errors='ignore')

    if "cumm" in gt_id: 
        df_truth = df_truth
    elif "incd" in gt_id: 
        # Aggregate incidents over 7 days
        df_truth = df_truth - df_truth.shift(periods=7, axis='columns')
    else: 
        raise ValueError(f'Invalid gt_id {gt_id}')

    fips_codes = get_fips_codes()
    state_fips = fips_codes[fips_codes['abbreviation'].notna()]
    # Strip leading whitespace from fips codes
    # state_fips['location'] = pd.Series(state_fips['location'].str.strip("0"))

    # Add state fips to dataframe
    df_proc = df_truth.merge(state_fips, left_index=True, right_on='location_name', how='left')
    df_proc = df_proc.loc[df_proc.index.dropna()]
    df_proc.index = df_proc.index.astype(np.int64)
    df_proc.sort_index(inplace=True)

    date_cols = [x for x in df_proc.columns if '/' in x] # TODO: make this more robust
    date_cols_obj = [datetime.strptime(x, "%m/%d/%y") for x in date_cols]
    date_to_obj = dict(zip(date_cols, date_cols_obj))
    locations = df_proc.location.to_list()

    # Populate new dataframe with data as multindex
    mi = pd.MultiIndex.from_product([locations, date_cols], names=['location', 'date'])
    df_return = pd.DataFrame(columns=['gt'], index=mi)
    for ind in mi:
        df_return.loc[ind, 'gt'] = float(df_proc.loc[df_proc['location'] == ind[0], ind[1]])

    df_return = df_return.rename(index=date_to_obj)

    return df_return

def get_ground_truth_file(gt_id, agg_days=7):
    return os.path.join("data", "dataframes", f"gt-{gt_id}-{agg_days}d.h5")

def get_jhu_raw(gt_id):
    """Loads the raw Johns Hopkins University ground truth data.

    Args:
       gt_id: string identifying which ground-truth data to return;
         valid choices are "cumm_death", "incd_death", "cumm_case", "incd_case" 
    """
    if gt_id == 'cumm_death' or gt_id == 'incd_death':
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    elif gt_id == 'cumm_case' or gt_id == 'incd_case':
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    else:
        raise ValueError(f'Unrecognized gt_id: {gt_id}')
    return pd.read_csv(url)

def get_target_dates(date_str, first_date, last_date, horizon=None):
    """Return list of target date datetime objects for model evaluation.
        Note: returned object should always be a list, even for single target dates

    Args
    ----------
    date_str : string
        Either a named set of target dates (
            "std_daily"
            "std_weekly"
        or a string with comma-separated dates in YYYYMMDD format
        (e.g., '20170101,20170102,20180309'))
    last_date: datetime object specifying the final data date
    horizon: string, Either "1w", "2w", "3w", "4w". Used for generated contest date periods. 

    Returns
    -------
    list
        List with datetime objects
    """
    number_of_days = (last_date - first_date).days

    if date_str == "std_daily":
        dates = [first_date + timedelta(days=x) for x in range(number_of_days)]
        return dates
    elif date_str == "std_weekly":
        start_date = next_dow(first_date, dow=dow_to_int['Saturday'], after=True)
        dates = [start_date + timedelta(days=x) for x in range(0, number_of_days, 7) if start_date + timedelta(days=x) <= last_date]
        return dates
    elif "-" in date_str:
        # Input is a string of the form '20170101-20180130'
        first_date, last_date = date_str.split("-")
        first_date = string_to_dt(first_date)
        last_date = string_to_dt(last_date)
        dates = [
            first_date + timedelta(days=x)
            for x in range(0, (last_date - first_date).days + 1)
        ]
        return dates
    elif "," in date_str:
        # Input is a string of the form '20170101,20170102,20180309'
        dates = [datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 6:
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime(year=year+1, month=1, day=1)
        else:
            last_date = datetime(year=year, month=month+1, day=1)
        dates = [
            first_date + timedelta(days=x)
            for x in range(0, (last_date-first_date).days)
        ]
        return dates
    elif len(date_str) == 8:
        # Input is a string of the form '20170101', representing a single target date
        dates = [datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    else:
        raise NotImplementedError("Date string provided cannot be transformed "
                                  "into list of target dates.")


def get_pred_by_date(df, target_str, expert_models, degault_value=None):
    """Merges forecasts from experts. Returns a forecast_size by number 
    of experts merged pd DataFrame. Additionally returns a list of experts 
    who are missing a prediction for the target date. Missing predictions are 
    filled with a default_value if provided, or skipped if default_value is None.

    Args:
       expert_filenames: dict from model_name to full paths to the forecast 
           files a given target date and contest objective 
       expert_models: a list of expert model names that provides an ordering
           for the columns of the returned np array
       default_value: when an expert prediction is missing, the corresponding
           column is filled with default_value. 
    """   
    # Read in each experts predictions
    sub_df = df.loc[df.index.get_level_values("target_end_date") == target_str].copy()
    sub_df.index = sub_df.index.droplevel(['location', 'target_end_date'])
    sub_df = sub_df.T
    sub_df.index = pd.Index([target_str], name='target_date')
    
    return sub_df, set(expert_models).difference(set(sub_df.columns))

def generate_expert_df(get_filename_fn, target_date_objs, expert_models, expert_submodels, default_value=None):
    """ Generates a merged expert dataframe for all target dates in 
    target_date_objs. Utility funciton, not possible for real-time funciton

    Args:
       get_filename_fn: partial of get_forecast_filename that takes a model,
           submodel, and target date as input and produces forecast filename
       target_date_objs: a pd Series of target date time objects 
       expert_models: a list of expert model names that provides an ordering
           for the columns of the returned np array
       expert_submodels: a dictionary from model name to selected submodel name          
       default_value: when an expert prediction is missing, the corresponding
           column is filled with default_value. If None, that expert is skipped. 
    """       
    expert_df = None
    for target_date_obj in target_date_objs:
        # Convert target date to string
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')  

        # Get names of submodel forecast files using the selected submodel
        expert_filenames = {m: get_filename_fn(model=m, 
                                               submodel=s,
                                               target_date_str=target_date_str) 
                               for (m, s) in expert_submodels.items()}
        
        # Get expert merged df for target date
        merged_df, missing_experts = get_and_merge_experts(expert_filenames, default_value)
        
        if len(missing_experts) > 0 and default_value is None:
            printf(f"warning: experts {missing_experts} unavailable for target={target_date_obj}; skipping")
            continue

        # Update full expert_df
        if expert_df is None:
            expert_df = merged_df
        else:
            expert_df = pd.merge(expert_df, merged_df, 
                                 left_index=True, right_index=True, how='outer', 
                                 on=expert_models)
    return expert_df.sort_index()


def next_dow(date, dow, after=True):
    """ Returns a datetime object with the first instance of day of week dow
    occuring before or after date
    Args:
      date: target date as a datetime object
      dow: day of week, 1 = Tuesday, 2 = Wednesday 
      after: boolean, whether to find first date before (False) or after (True)
    """
    for d in range(7): # Check a one week period
        if after:
            target = date + timedelta(days=d)
        else:
            target = date - timedelta(days=d)

        if target.weekday() == dow: # test day of week
            return target

def get_task_metrics_dir(model="online_expert", submodel=None, gt_id="cumm_death", horizon="1w"):
    """Returns the directory in which evaluation metrics for a given submodel
    or model are stored

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns metrics
         directory associated with selected submodel or returns None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: 34w or 56w
    """
    if submodel is None:
        submodel = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon)
        if submodel is None:
            return None
    return os.path.join(
        "eval", "metrics", model, "submodel_forecasts", submodel, f"{gt_id}_{horizon}"
)
def get_task_forecast_dir(model="online_expert",
                          submodel=None,
                          gt_id="cumm_death",
                          horizon="1w"):
    """Returns the directory in which forecasts from a given submodel or
    model and a given task are stored

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns forecast
         directory associated with selected submodel or None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: 34w or 56w
    """
    if submodel is None:
        submodel = get_selected_submodel_name(model=model,gt_id=gt_id,
                                              horizon=horizon)
        if submodel is None:
            return None
    return os.path.join("models", model, "submodel_forecasts", submodel,
                        f"{gt_id}_{horizon}")