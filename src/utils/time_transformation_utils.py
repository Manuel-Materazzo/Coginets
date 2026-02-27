import numpy as np
import pandas as pd
from pandas import DataFrame


def generate_yearly_features(df: DataFrame, date_col: str, date_format: str = None) -> DataFrame:
    """
    Generates seasonality cycle features for a dataframe.
    :param date_format:
    :param df:
    :param date_col:
    :return:
    """

    # when format is provided, parse the date
    if date_format is not None:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    # extract year day, week day, and monh
    df['dayofyear'] = df[date_col] % 365
    df['dayofweek'] = df[date_col] % 7
    df['month'] = df[date_col] % 365 // 31
    # create sin and cos waves for features, to suggest that time is cyclical
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def generate_daily_features(df: DataFrame, time_col: str, seconds: bool = False, time_format: str = None) -> DataFrame:
    """
    Generates day cycle features for a dataframe.
    :param time_format:
    :param seconds:
    :param time_col:
    :param df:
    :return:
    """

    # when format is provided, parse the time
    if time_format is not None:
        df[time_col] = pd.to_datetime(df[time_col], format=time_format)

    # extract an hour and minute series
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    if seconds:
        df['seconds'] = df[time_col].dt.second

    # create sin and cos wave for features, to suggest that time is cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    if seconds:
        df['seconds_sin'] = np.sin(2 * np.pi * df['seconds'] / 60)
        df['seconds_cos'] = np.cos(2 * np.pi * df['seconds'] / 60)

    return df
