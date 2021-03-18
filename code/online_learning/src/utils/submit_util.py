# Utility functions supporting submission
import os
import sys
import pandas as pd
import json
from datetime import datetime, timedelta
from src.utils.general_util import printf, set_file_permissions
from src.utils.experiments_util import get_target_date

def get_submission_col(gt_id, target_horizon):
    """Returns the official submission column name for a given prediction task
    
    Args:
      gt_id: "contest_tmp2m" or "contest_precip"
      target_horizon: "34w" or "56w"
    """
    if gt_id.endswith("tmp2m") and target_horizon == '34w':
        return 'temp34'
    elif gt_id.endswith("tmp2m") and target_horizon == '56w':
        return 'temp56'
    elif gt_id.endswith("precip") and target_horizon == '34w':
        return 'prec34'
    elif gt_id.endswith("precip") and target_horizon == '56w':
        return 'prec56'

def get_submission_filename(gt_id, 
                            target_horizon,
                            deadline_date_str):
    """Returns the path to the official submission file for a given 
    submission date and task.

    Args:
       gt_id: contest_tmp2m or contest_precip
       target_horizon: 34w or 56w
       deadline_date_str: official contest submission date in YYYYMMDD format
    """
    task = gt_id+"_"+target_horizon
    # Get version of deadline string in YYYY-MM-DD format
    dashed_date_str = datetime.strptime(
        deadline_date_str, '%Y%m%d').strftime('%Y-%m-%d')
    return os.path.join("submit", deadline_date_str, task, 
                        dashed_date_str+".csv")

def save_submitted_submodel_params(model="keras_linear", gt_id="contest_tmp2m", 
                                   target_horizon="34w", deadline_date="20191112"):
    """Saves json file with model name and paramaters for the official submission file for a given 
    submission date and task.

    Args:
        model: string model name
        gt_id: contest_tmp2m or contest_precip
        target_horizon: 34w or 56w
        deadline_date: string official contest submission date in YYYYMMDD format
    """
    
    selected_submodel_params_file=os.path.join("src","models",model,"selected_submodel.json")
    submitted_submodel_params_file = os.path.join("submit", deadline_date, f'{gt_id}_{target_horizon}', "submitted_submodel.json")

    with open(selected_submodel_params_file) as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']

    submitted_submodel_params = {'model_name': model, 'parameters':json_args}
    
    json_data = {f'{gt_id}_{target_horizon}': submitted_submodel_params}
    
    #set_file_permissions(submitted_submodel_params_file)     
    with open(submitted_submodel_params_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    set_file_permissions(submitted_submodel_params_file)


