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
        mode = time_diffs.mode()
        # test if mode is notn empty else use median
        if not mode.empty:
            min_intervals.append(mode[0])
        else:
            min_intervals.append(time_diffs.median())

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

    print("Files to be merged: \n")
    # iterate over files
    for file in data_folder.iterdir():
        if file.is_file() and file != file_exception:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"- {file}: {len(df)} entries")
            

    # get time var from 
    time_var = next((col for col in dataframes[0].columns if "time" in col.lower()), None)

    # raise error if no time variabel found
    if time_var is None:
        raise KeyError("No time variable found. Data must be time frequency data.")

    # extrac max and min times from datasets
    min_time = min(df[time_var].min() for df in dataframes)
    max_time = max(df[time_var].max() for df in dataframes)
    print(f"\nMin time across datasets: {min_time}\nMax time across datasets: {max_time}")

    # create common time grid
    period_of_highest_freq = get_optimal_interval(dataframes, time_var) # 100 Hz by visual inspection; function is more precise
    common_time = pd.Index(np.arange(min_time, max_time, period_of_highest_freq))

    print(f"Highes fre. period: {period_of_highest_freq}\nLength of common_time: {len(common_time)}")

    # time adjusted dataframes
    adjusted_dfs = []

    # adjust each df to common time 
    for idx, df in enumerate(dataframes):
        
        # indexing df based on time
        time_indexed_df = df.set_index(time_var)

        # map old indices to common time indices
        mapped_indices = common_time.get_indexer(time_indexed_df.index, method="nearest")

        # re-index and impute values according to mapped indices
        time_resampled = time_indexed_df.reindex(common_time)
        time_resampled.iloc[mapped_indices] = time_indexed_df.values 

        # remove senor time variable 
        if time_var in time_resampled.columns:
            time_resampled.drop(time_var, axis=1)

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
    close_indices = df.index[df.index.get_indexer(ctime, method="nearest")]

    # fill in label for the cumulative time 
    for idx, (label, end_time) in enumerate(zip(labels, close_indices)):
        # define starting time for activeity 
        if idx == 0:
            start = df.index.min()
        else:
            start = ctime[idx - 1]
        
        # define ending time for activity
        if idx == len(ctime) - 1:
            end = df.index.max()
        else:
            end = end_time
        
        # fill in labels
        df.loc[start:end, colname] = label

    return df

def strtime_to_sec(str_time: str) -> float:
    """
    Convert 'min:sec,msec' to seconds (float)
    """

    min_, sec, msec = map(int, re.split(r"[:,]", str_time))

    total_time = (min_ * 60)  + sec + (msec / 1000)

    return total_time

def safe_interpol(df: pd.DataFrame, lb: float=0.25, ub: float=0.95, all: bool= False, edge_case: str="drop") -> pd.DataFrame:
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
            if (lb <= nan_frac <= ub) or all:
                df[col] = df[col].interpolate(method="linear")

                # if missing values -> edge cases (values at start or end: between NaN's)
                if df[col].isna().any():

                    # fill in with neares forward or backwards value
                    if edge_case == "fill":
                        df[col] = df[col].bfill().ffill()
                    # fill in missing values with zeros
                    elif edge_case == "zero":
                        df[col] = df[col].fillna(0)
                    # do nothing 
                    elif edge_case == "drop":
                        pass
                
    return df

def get_dataset(path: str, labels: list[str]=None, rtime: list[float]=None,
                label_col: str=None, impute: bool=False) -> pd.DataFrame:
    # get all files together in one dataframe
    data = join_sdata(path)

    # add labels to dataset if given
    if labels:
        rtime_sec = [strtime_to_sec(t) for t in rtime]
        # get linear time (stopwatch timer)
        ctime = [rtime_sec[0]]
        for idx, t_sec in enumerate(rtime_sec[1:]):
            # cummulative time at each stampoint = time[idx - 1] + current time 
            ctime.append(ctime[idx] + t_sec)
        
        data_labeld = add_labels(data, labels, ctime, label_col)

        # return data with or without imputation
        if impute:
            return safe_interpol(data_labeld)
        else:
            return data_labeld
    else:
        # add imputations and return or just return
        if impute:
            return safe_interpol(data_labeld)
        else:
            return data_labeld
    




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

