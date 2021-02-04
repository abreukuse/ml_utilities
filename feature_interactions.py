# for categorical_interactions
from itertools import combinations
from functools import reduce

# for numerical_interactions
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def categorical_interactions(X, variables='all', order=2):
    X = X.copy()
    df = X[variables] if variables != 'all' else X
    all_combinations = []
    order = len(df.columns) if order == 'all' else order

    if order > len(df.columns): 
        raise ValueError(f'The value passed to "order" exceeds the number of variables. Please choose a number between 2 and {len(df.columns)}.')
    elif order < 2:
        raise ValueError('The value of "order" needs to be at least 2.')
    else:
        for i in range(2, order+1):
            combination = list(combinations(df, i))
            all_combinations.append(combination)

        all_combinations = reduce(lambda a,b: a+b, all_combinations)

        for each in all_combinations:
            each = list(each)
            X[(reduce(lambda a,b: f'{a}-{b}', each))] = ['-'.join(item) for item in df[each].values]

        return X


def numerical_interactions(X, variables='all', degree=2, interaction_only=True):
    X = X.copy()
    X_sliced = X[variables] if variables != 'all' else X
    len_variables = len(X_sliced.columns)
    interactions = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    interactions_array = interactions.fit_transform(X_sliced)
    new_columns = interactions.get_feature_names(X_sliced.columns)[len_variables:]
    df = pd.DataFrame(interactions_array[:, len_variables:], index=X.index, columns=new_columns)
    X = X.join(df)

    return X