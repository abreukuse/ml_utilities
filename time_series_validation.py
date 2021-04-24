"""Time Series Validation"""
import pandas as pd
from operator import le, ge
from typing import List

def time_series_split(X: pd.DataFrame, 
                      date_column: str, 
                      split_dates: List[str],
                      window: str) -> List[tuple]:
    """
    Split data based on time
    -----------------------

    Parameters
    X: Pandas dataframe to be splitted
    date_column: String. Name of the column containing the dates.
    split_dates: List. Each element of the list represent a date e.g '2015-05-20'
    window: String. Options {'sliding', 'expanding'}
    """
    X = X[date_column]
    dates = split_dates
    storage = []
    indices = []

    def slice_tail(end, condition):
        return X.loc[condition(X, end)].index

    def slice_middle(start, end):
        return X.loc[(X > start) & (X < end)].index

    # Separate each chunk of data in the storage list.
    for i in range(len(dates)):

        if dates[i] == dates[0]:
            end = dates[i]
            storage.append(slice_tail(end, le))
        
        elif dates[i] == dates[-1]:
            start = dates[i-1]
            end = dates[i]
            storage.append(slice_middle(start, end))
            storage.append(slice_tail(end, ge))
        
        else:
            start = dates[i-1]
            end = dates[i]
            storage.append(slice_middle(start, end))
    
    lenght_storage = len(storage)

    # Create the indices for train and test regarding the 
    # type of validation chosen i.e 'sliding' or 'expanding'

    # sliding window
    if window == 'sliding':

        indices = [
                   (storage[i], storage[i+1]) 
                   for i in range(lenght_storage - 1)
                   ]

    # expanding window
    if window == 'expanding':
        expanding = storage[0]
        indices = [(storage[0], storage[1])]

        for i in range(1, lenght_storage - 1):
            expanding = expanding.append(storage[i])
            indices.append((expanding, storage[i+1]))

    return indices
