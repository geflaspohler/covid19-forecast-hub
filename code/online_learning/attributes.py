# Model attributes
import json
import os
# from src.utils.general_util import set_file_permissions, hash_strings
import string

MODEL_NAME = "online_expert"
SELECTED_SUBMODEL_PARAMS_FILE=os.path.join("src","models",MODEL_NAME,"selected_submodel.json")
SUBMODEL_NAME_TO_PARAMS_FILE = os.path.join(
    "src", "models", MODEL_NAME, "submodel_name_to_params.json")

def get_selected_submodel_name(gt_id, target_horizon, return_params=False):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task    
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']

    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args, return_params=return_params)

def get_submodel_name(expert_models="doy,cfsv2,multillr,llr", 
                      reg='entropic', 
                      alg='rmplus', 
                      hint="None", 
                      delay=0,
                      training_dates="std_weekly",
                      return_params=False,
                      exp_name="None"):
  """Returns submodel name for a given setting of model parameters
     
     Args:
      expert_models: comma sepearted string of expert models
      reg: regularization string
      alg: online learning algorithm string
      hint: hint type string
      training_dates: dates over which online learning was run
      return_params: (boolean) True or False. If True, returns a dictionary
        item containing the algorithm parameters for submodel
      exp_name: an optional string, included to identify a specific experiment
    """
  models = expert_models.split(',')
  models.sort()
  model_strings = (",").join(models)

  # Generate unique hash for expert_models, reg, training_dates 
  #col_hash = hash_strings([model_strings], sort_first=False)
  model_str= get_model_shortcode(models)
  date_str = get_date_shortcode(training_dates)
  reg_str = get_reg_shortcode(reg)
  alg_str = get_alg_shortcode(alg)

  # Include an optional identifying string for an experiment if exp_name == "None":
  if exp_name == "None":
    exp_string = ""
  else:
    exp_string = f"{exp_name}_"
 
  submodel_name = (f"{MODEL_NAME}-{exp_string}{alg_str}_reg{reg_str}_D{delay}_{hint}_{date_str}_{model_str}")

  submodel_params = {
      'alg': alg,
      'expert_models': model_strings,
      'reg': reg,
      'hint': hint,
      'delay': delay,
      'training_dates': training_dates
  }
  if return_params:
    return submodel_name, submodel_params
  return submodel_name

def get_model_shortcode(model_list):
    """
    Get shortcode for the models, passed in as a list of strings
    """
    model_str = ""
    if len(model_list) > 10:
        return "allmodels"
    for m in model_list:
        model_str += m[0].upper()
    return model_str

def get_alg_shortcode(alg_str):
    """
    Get shortcode for the models, passed in as a list of strings
    """
    if alg_str == "adahedgefo":
        return "AF"
    else:
        return alg_str
    

def get_date_shortcode(date_str):
    """
    Get shortcode for the standard date strings, to use in submodel names
    """
    if date_str == "std_contest":
        return "SC"
    elif date_str == "std_contest_daily":
        return "SCD"
    elif date_str == "std_future":
        return "SF"
    elif date_str == "std_test":
        return "ST"
    elif date_str == "std_val":
        return "SV"
    elif date_str == "std_contest_eval":
        return "SCE"
    elif date_str == "std_contest_eval_daily":
        return "SCED"
    else:
        return date_str

def get_reg_shortcode(reg_str):
    """
    Get shortcode for the regularization strings
    """
    if reg_str == "entropic":
        return "E"
    elif reg_str == "quadratic":
        return "Q"
    elif reg_str == "orig":
        return "OR"
    elif reg_str == "delay_hint":
        return "DH"
    elif reg_str == "nodelay_hint":
        return "NH"
    elif reg_str == "delay_nohint":
        return "DN"
    elif reg_str == "plusplus":
        return "PP"
    elif reg_str == "forward_drift":
        return "FD"
    else:
        return reg_str
