import pandas as pd
import numpy as np
from numba import jit

def __create_holiday_feature(df, 
                             values, 
                             holiday_name, 
                             date_column, 
                             map_time, 
                             which_week=None, 
                             holiday=None, 
                             month=None):
    """
    Called inside __time_to_holiday function to add the features to the original dataframe
    ---------------------------------------
    Parameters
    
    df: Pandas dataframe with a date columns.
    values: Which time stamps to create e.g ['days','weeks','months','years']. It came from the holidays dictionary in seasonal features
    holiday_name: String as the first item in the tuple passed to holidays arguments in seasonal_features.
    date_column: Name of the column that contains the dates.
    date_column: dictionary mapping timastamps in the right format for pandas e.g {'days':'D','weeks':'W','months':'M','minutes':'m'}.
    which_week: Which week of the month {'first', 'second', 'last', etc} to form the dynamic holiday.
    holiday: String containing the holiday date for static holiday e.g 'December-25'
    month: String. Month that the dynamic holiday happen
    """
    if (which_week == None) and (holiday_name != None):
        date = df[date_column].apply(lambda x: pd.to_datetime(f'{x.year}-{holiday}', format='%Y-%B-%d'))
    else:
        date = df[date_column].apply(lambda x: which_week.apply(pd.to_datetime(f'{x.year}-{month}', format='%Y-%B')))

    for value in values:
        df[f'{value}_to_{holiday_name}'] = (df[date_column] - date)/np.timedelta64(1, map_time[value])

def __time_to_holidays(df, 
                       date_column,
                       static_holidays=None,
                       dynamic_holidays=None):

    """
    This function is used inside the 'seasonal_features' function and it creates features related to holidays.
    ---------------------------------------
    Parameters

    df: Pandas dataframe with a date columns.
    date_column: Name of the column that contains the dates.
    static_holidays: Holidays that always happens at the same days e.g {('christmas', 'December-25'), ('4-of-July', 'July-04')}.
    dynamic_holidays: Holidays that happens at different days each year e.g {('black-friday', 'November', 'Friday', 'fourth'), ('fathers-day', 'June', 'Sunday', 'third')}.
    """

    map_time = {'days':'D',
                'weeks':'W',
                'months':'M',
                'years':'Y',
                'hours':'h',
                'minutes':'m',
                'seconds':'s'}

    map_weekday = {'Monday':0,
                   'Tuesday':1,
                   'Wednesday':2,
                   'Thursday':3,
                   'Friday':4,
                   'Saturday':5,
                   'Sunday':6}

    map_week = {'first':0,
                'second':1,
                'third':2,
                'fourth':3,
                'last':'last'}

    if static_holidays:
        for key, values in static_holidays.items():
            holiday_name = key[0]
            holiday_date = key[1]
            __create_holiday_feature(df, values, holiday_name, date_column, map_time, holiday=holiday_date)
        
    if dynamic_holidays:
        for key, values in dynamic_holidays.items():

                week = key[-1]
                holiday_name = key[0]
                month = key[1]
                week_day= key[2]

                if week != 'last':
                    week_of_month = pd.tseries.offsets.WeekOfMonth(week=map_week[week], weekday=map_weekday[week_day])

                    __create_holiday_feature(df, values, holiday_name, date_column, map_time, which_week=week_of_month, month=month)

                else:
                    last_week_of_month = pd.tseries.offsets.LastWeekOfMonth(weekday=map_weekday[week_day])

                    __create_holiday_feature(df, values, holiday_name, date_column, map_time, which_week=last_week_of_month, month=month)


def seasonal_features(df, 
                      date_column, 
                      which_ones='all', 
                      cyclical=False,
                      holidays=False,
                      copy=False,
                      **kwargs):
    '''
    Generates seasonal features from a time series
    -------------------------------
    Parameters

    df: Pandas dataframe.

    date_column: String name of the column with the dates.

    which_ones: List containing which features should be created. 
                            Features available-
                            ['day','quarter','month','weekday','dayofyear','week','hour','minute','second'].
                            Default 'all'. All the features will be created.

    cyclical: Boolean, default 'False'. Set 'True' in order to produce sine and cosine conversion.

    holidays: Boolean, default 'False'. If True, it gives the capability to include holidays or any other specific date.
              In order to do that you need to provide two other arguments {static_holidays, dynamic_holidays} to the function as the example below.

                time_periods = ['days','weeks','months','years'] # it can also include: hours, minutes, seconds if applicable.

                static_holidays = {('christmas', 'December-25'): time_periods,
                                   ('4-of-July', 'July-04'): time_periods}

                dynamic_holidays = {('black-friday', 'November', 'Friday', 'fourth'): time_periods,
                                    ('fathers-day', 'June', 'Sunday', 'third'): time_periods}

    copy: Boolean. Default 'False' for the changes to occur in the same dataframe provided. 
          Set 'True' in order to return the result in a new dataframe.
    '''

    if copy: df = df.copy()

    dates = df[date_column].dt

    features = ['day',
                'quarter',
                'month',
                'weekday',
                'dayofyear',
                'week',
                'hour',
                'minute',
                'second'] if which_ones == 'all' else which_ones

    for feature in features:
        attribute = getattr(dates, feature) if feature is not 'week' else getattr(dates.isocalendar(), feature).apply(lambda x: int(x))
        if cyclical:
            df[f'{date_column}_{feature}_cos'] = np.cos(2 * np.pi * attribute/attribute.max())
            df[f'{date_column}_{feature}_sin'] = np.sin(2 * np.pi * attribute/attribute.max())
        else:
            df[f'{date_column}_{feature}'] = attribute

    if holidays:
        __time_to_holidays(df, date_column, **kwargs)
    
    if copy: return df


def lagging_features(df, 
                     target, 
                     lags=None, 
                     lags_diff=None,
                     lags_pct_change=None, 
                     group_by=None,
                     initial_period=1,
                     copy=False):
    '''
    Generates lagged features from a time series
    -------------------------------
    Parameters

    df: Pandas dataframe.

    target: String referring to the target variable.

    lags: List containing integer of which lags to include as features.
          The minimum value of the lags will be used to shift the other features below.

    lags_diff: List with integers of the lag differences to be included as features.

    lags_pct_change: List with integers for the calculation of the percentage change of previous values. Be carefull with zero values.

    group_by: String with the column name referring the groups to be formed in order to generate separated features for each group.

    initial_period: Integer. Periods to shift the features created. Only needed if the lags were not set.
                    Default: 1, which means the features will be shifted one period ahead of time.
                    If the lags were provided the minimum value of the lags will be used as the initial_period.

    copy: Boolean. Default 'False' for the changes to occur in the same dataframe provided. 
                     Set 'True' in order to return the result in a new dataframe.
    '''
    if copy: df = df.copy()

    if lags:
        for lag in lags:
            df[f'lag_{target}_{lag}'] = df.groupby([group_by])[target].shift(lag) if group_by \
            else df[target].shift(lag)

    initial_period = np.min(lags) if lags else initial_period

    if lags_diff:
        for diff in lags_diff:
            df[f'lag_diff_{target}_{diff}'] = df.groupby([group_by])[target].shift(initial_period).diff(diff) if group_by \
            else df[target].shift(initial_period).diff(diff)

    if lags_pct_change:
        for pct in lags_pct_change:
            df[f'pct_chance_{target}_{pct}'] = df.groupby([group_by])[target].shift(initial_period).pct_change(pct) if group_by \
            else df[target].shift(initial_period).pct_change(pct)
            
    if copy: return df


@jit(nopython=True)
def __weighted_average(x, window):
    """Create weighted moving average features"""
    array = np.array([np.nan]*window)
    for i in range(len(x) - window):
        array = np.append(array, x[i : window+i][::-1].cumsum().sum() * 2 / window / (window + 1))
    return array

def moving_statistics_features(df, 
                               target, 
                               windows, 
                               which_ones='all', 
                               group_by=None, 
                               delta_roll_mean=False,
                               weighted_average=False,
                               copy=False):
    '''
    Generates moving statistics features from a time series
    -------------------------------
    Parameters

    df: Pandas dataframe.

    target: String referring to the target variable.
    windows: List of integers containing which window sizes should be considered for calculating the statistics.

    which_ones: List containing which features should be created. 
                            Features available-
                            ['mean','median','std','min','max','skew','kurt','sum'].
                            Default 'all'. All the features will be created.

    group_by: String with the column name referring the groups to be formed in order to generate separated values for each group.

    delta_roll_mean: Boolean. Wether or not to genetare features referring current value minus the mean.

    copy: Boolean. Default 'False' for the changes to occur in the same dataframe provided. 
                     Set 'True' in order to return the result in a new dataframe.
    '''
    if copy: df = df.copy()

    targets = df.groupby(group_by, sort=False, as_index=False)[target] if group_by else df[target]

    features = ['mean',
                'median',
                'std',
                'min',
                'max',
                'skew',
                'kurt',
                'sum'] if which_ones == 'all' else which_ones

    for feature in features:
        for window in windows:
            df[f'{feature}_{target}_{window}'] = getattr(targets.shift().rolling(window), feature)()

    if delta_roll_mean:
        for window in windows:
            if group_by:
                series = df.groupby([group_by])[target]
                groups = series.groups.keys()
                df[f'delta_roll_mean_{target}_{window}'] = \
                pd.concat([(series.get_group(group) - series.get_group(group).rolling(window).mean()).shift() for group in groups])
            else:
                df[f'delta_roll_mean_{target}_{window}'] = \
                (df[target] - df[target].rolling(window).mean()).shift()

    if weighted_average:
        for window in windows:
            if group_by:
                result = [__weighted_average(values[target].values, window) for group, values in targets]
                df[f'weighted_average_{target}_{window}'] = np.concatenate(result)
            else:
                df[f'weighted_average_{target}_{window}'] = __weighted_average(targets.values, window)

    if copy: return df