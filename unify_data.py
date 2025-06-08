import pandas as pd
import numpy as np
from pathlib import Path
import re

def get_optimal_interval(dataframes: list[pd.DataFrame], time_col: str) -> float:
    """
    Find optimal time interval among datasets to avoid loosing data
    """

    min_intervals = []

    for df in dataframes:

        # calculate time between samples
        time_diffs = df[time_col].diff().dropna()

        # extract the most common interval size on the df
        min_intervals.append(time_diffs.mode()[0])

    # return optimal interval based on datasets
    return min(min_intervals)


def join_sdata(path: str, file_exception=None) -> pd.DataFrame:
    """
    Join sensor datasets into a cohessive dataset and save as CSV file.
    Adapt the new dataframe to work with optimal time interval on the 
    basis of the given datasets.
    """
    
    # generate path where datasets are located
    data_folder = Path(path)
    dataframes = []

    # iterate over files
    for file in data_folder.iterdir():
        if file.is_file() and file != file_exception:
            df = pd.read_csv(file)
            dataframes.append(df)

    # get time var from 
    time_var = next((col for col in dataframes[0].columns if "time" in col.lower()), None)

    # raise error if no time variabel found
    if time_var is None:
        raise KeyError("No time variable found. Data must be time frequency data.")

    # extrac max and min times from datasets
    min_time = min(df[time_var].min() for df in dataframes)
    max_time = max(df[time_var].max() for df in dataframes)

    # create common time grid
    period_of_highest_freq = get_optimal_interval(dataframes, time_var) # 100 Hz by visual inspection; function is more precise
    common_time = np.arange(min_time, max_time, period_of_highest_freq) 

    # time adjusted dataframes
    adjusted_dfs = []

    # adjust each df to common time 
    for idx, df in enumerate(dataframes):
        
        # indexing df based on time
        time_indexed_df = df.set_index(df.columns[0])

        # fill data by reindexing to common time
        time_resampled = time_indexed_df.reindex(common_time) # method="nearest" only if missing values need to be imputed directly

        if time_var in time_resampled.columns:
            time_resampled.drop(time_var, axis=1) # drop sensor time 

        adjusted_dfs.append(time_resampled)

    # combine all files together 
    final_df = pd.concat(adjusted_dfs, axis=1)

    # add common time variable and index name
    final_df["Common time (s)"] = common_time
    final_df.index.name = "timestamp"

    return final_df


def add_labels(df: pd.DataFrame, labels: list[str], ctime: list[float], colname: str) -> pd.DataFrame:
    """
    Add activity labels to temporal dataframe
    """
    # create new colum for 
    df[colname] = None

    # get nearest time indices to the commulative measured 
    close_indices = df.index.get_indexer(ctime, method="nearest")

    # fill in label for the cumulative time 
    for idx, (label, time_index) in enumerate(zip(labels, close_indices)):
        # fill labels according to cases 
        if idx == 0:
            # begining -> standpoint
            df.loc[df.index[0]:time_index, colname] = label
        elif idx == len(labels) -1:
            # prev. standpoint -> end
            df.loc[close_indices[idx -1]:df.index[-1], colname] = label
        else:
            # prev. standpoint -> current standpint
            df.loc[close_indices[idx - 1]:time_index, colname] = label

    return df

def safe_interpol(df: pd.DataFrame, lb: float, ub: float) -> pd.DataFrame:
    """
    Interpolate values in dataframe within a percentage range of missing values
    compared to the total amount of entries
    """

    for col in df.columns:
        
        # aply interpolation only if the variable is numeric 
        if pd.api.types.is_numeric_dtype(df[col]):
            
            # count nans per column
            nan_frac = df[col].isna().sum() / len(df)

            # interpolate columnns which have missing values within the range
            if lb < nan_frac < ub:
                df[col].interpolate(method="linear", inpace=True)

    return df

def strtime_to_sec(str_time: str) -> float:
    """
    Convert 'min:sec,msec' to seconds (float)
    """

    min_, sec, msec = map(int, re.split(r"[:,]", str_time))

    total_time = (min_ * 60)  + sec + (msec / 1000)

    return total_time
 

def main():

    # activities in sequence of occurance 
    activ_seq = ["rest", "walk", "phone", "stairs", "rest", 
                 "phone", "socialize", "walk", "study", "walk",
                 "stairs", "walk", "phone", "study", "socialize"]
    # relative activity times
    rtimes = ["01:15,13", "00:50,92", "01:01,08", "1:17,82", "01:15,43",
                      "00:50,37", "01:15,53", "00:42,41", "01:16,86", "00:55,36",
                      "01,29,60", "01:02,60", "00:40,60", "01:45,82", "01:26,80"]
    rtimes_sec = [strtime_to_sec(t) for t in rtimes]

    # actual time (cummulative time)
    linear_time = [rtimes_sec[0]]
    for idx, val in enumerate(rtimes_sec[1:]):
        linear_time.append(val + linear_time[idx])

    # join data frames, add labels and save
    joined_df = join_sdata("sens_data", file_exception="Proximity.csv")
    labeld_df = add_labels(joined_df, activ_seq, linear_time, "Activity")
    labeld_df.to_csv("joined_data.csv", index=False)

    # interpolate missing values
    label_df_interpol = safe_interpol(labeld_df, 0.25, 0.95)
    label_df_interpol.to_csv("joined_interpol_data.csv", index=False)


if __name__ == "__main__":
    main()

