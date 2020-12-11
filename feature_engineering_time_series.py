import pandas as pd
import numpy as np

def seasonal_features(df, 
                      date_column, 
                      which_ones='all', 
                      cyclical=False,
                      deliver=False):
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

  cyclical: Boolean default 'False'. Set 'True' in order to produce sine and cosine conversion.

  deliver: Boolean. Default 'False' for changes to occur in the same dataframe provided. 
           Set 'True' in order to return the result in a new dataframe.
  '''

  if deliver: df = df.copy()

  obj = df[date_column].dt

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
    attribute = getattr(obj, feature) if feature is not 'week' else getattr(obj.isocalendar(), feature)
    if cyclical:
      df[f'{feature}_cos'] = np.cos(2 * np.pi * attribute/attribute.max())
      df[f'{feature}_sin'] = np.sin(2 * np.pi * attribute/attribute.max())
    else:
      df[feature] = attribute
  
  if deliver: return df


def lag_attributes(df, 
                   target, 
                   lags=None, 
                   lags_diff=None, 
                   group_by=None,
                   deliver=False):
  '''
  Generates lagged features from a time series
  -------------------------------
  Parameters

  df: Pandas dataframe.

  target: String referring to the target variable.

  lags: List containing integer of which lags to include as features.

  lags_diff: List with integers of the lag differences to be included as features.

  group_by: String with the column name referring the groups to be formed in order to generate separated features for each group.

  deliver: Boolean. Default 'False' for changes to occur in the same dataframe provided. 
           Set 'True' in order to return the result in a new dataframe.
  '''
  if deliver: df = df.copy()

  if lags:
    for lag in lags:
      df[f'lag_{target}_{lag}'] = df.groupby([group_by])[target].shift(lag) if group_by else df[target].shift(lag)

  if lags_diff:
    for diff in lags_diff:
      df[f'lag_diff_{target}_{diff}'] = df.groupby([group_by])[target].shift().diff(diff) if group_by else df[target].shift().diff(diff)
      
  if deliver: return df


def moving_attributes(df, 
                      target, 
                      windows, 
                      which_ones='all', 
                      group_by=None, 
                      delta_roll_mean=False,
                      deliver=False):
  '''
  Generates moving statistics features from a time series
  -------------------------------
  Parameters

  df: Pandas dataframe.

  target: String referring to the target variable.
  windows: List of integers containing which window sizes should be considered for calculating the statistics.

  which_ones: List containing which features should be created. 
              Features available-
              ['mean','median','std','min','max','week','skew','kurt','sum'].
              Default 'all'. All the features will be created.

  group_by: String with the column name referring the groups to be formed in order to generate separated values for each group.

  delta_roll_mean: Boolean. Wether or not to genetare features referring current value minus the mean.

  deliver: Boolean. Default 'False' for changes to occur in the same dataframe provided. 
           Set 'True' in order to return the result in a new dataframe.
  '''
  if deliver: df = df.copy()

  obj = df.groupby([group_by])[target].shift() if group_by else df[target].shift()

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
      df[f'{feature}_{target}_{window}'] = getattr(obj.rolling(window), feature)()

  if delta_roll_mean:
    for window in windows:
      if group_by:
        series = df.groupby([group_by])[target]
        groups = series.groups.keys()
        df[f'delta_roll_mean_{target}_{window}'] = pd.concat([(series.get_group(group) - series.get_group(group).rolling(window).mean()).shift() for group in groups])
      else:
        df[f'delta_roll_mean_{target}_{window}'] = (df[target] - df[target].rolling(window).mean()).shift()

  if deliver: return df