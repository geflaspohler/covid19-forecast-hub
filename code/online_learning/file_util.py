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