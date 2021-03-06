#-*- coding: utf-8 -*-

# The majority of this module I took from the autofeat lybrary: https://github.com/cod3licious/autofeat
# which is an automated feature engineer tool.
# The original code is here: https://github.com/cod3licious/autofeat/blob/master/autofeat/feateng.py
# I simply made some minor changes in order to fulfill my needs. Like implement fit and transform capabilities.

from builtins import str
import re
import operator as op
from functools import reduce
from itertools import combinations, product
import numpy as np
import pandas as pd
import sympy
from sympy.utilities.lambdify import lambdify
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class NumericalFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Given a DataFrame with original features, perform the feature engineering routine for max_steps.
    It starts with a transformation of the original features (applying log, ^2, sqrt, etc.),
    then in the next step, the features are combined (x+y, x*y, ...), and in further steps, the resulting
    features are again transformed and combinations of the resulting features are computed.

    Inputs:
        - start_features: list with column names for X with features that should be considered for expansion
                            (default: None --> all columns)
        - max_steps: how many feature engineering steps should be performed. Default is 3, this produces:
            Step 1: transformation of original features
            Step 2: first combination of features
            Step 3: transformation of new features
            (Step 4: combination of old and new features)
            --> with 3 original features, after 4 steps you will already end up with around 200k features!
        - transformations: list of transformations that should be applied; possible elements:
                             "1/", "exp", "log", "abs", "sqrt", "^2", "^3", "1+", "1-", "sin", "cos", "exp-", "2^"
                             (first 7, i.e., up to ^3, are applied by default)
        - verbose: verbosity level (int; default: 0)

    Attributes:
        - input_data: original data passed to fit method
        - df_fit_stage: data transformed after fit is applied
        - variables_to_persist: what variables will be includded in the final dataset

    Methods:
        - fit: fit the data in order to proceed with the transformation considering the correlation between variables in the training set
        - transform: transform the data based on the results of the fit process.
    """

    def __init__(self,
                 start_features=None,
                 max_steps=2,
                 transformations=("1/", "exp", "log", "abs", "sqrt", "^2", "^3"),
                 verbose=0):

        self.start_features = start_features
        self.max_steps = max_steps
        self.transformations = transformations
        self.verbose = verbose
        self.input_data = None
        self.df_fit_stage = None
        self.variables_to_persist = None

    def __engineer_features(self, X, y=None, correlation_threshold=0.9):
        
        def colnames2symbols(c, i=0):
            # take a messy column name and transform it to something sympy can handle
            # worst case: i is the number of the features
            # has to be a string
            c = str(c)
            # should not contain non-alphanumeric characters
            c = re.sub(r"\W+", "", c)
            if not c:
                c = "x%03i" % i
            elif c[0].isdigit():
                c = "x" + c
            return c

        def ncr(n, r):
            # compute number of combinations for n chose r
            r = min(r, n - r)
            numer = reduce(op.mul, range(n, n - r, -1), 1)
            denom = reduce(op.mul, range(1, r + 1), 1)
            return numer // denom

        # initialize the feature pool with columns from the dataframe
        if not self.start_features:
            variables = X.columns
            dont_transform = []
            df_dont_transform = pd.DataFrame()
        else:
            for c in self.start_features:
                if c not in X.columns:
                    raise ValueError("[feateng] start feature %r not in X.columns" % c)
            variables = self.start_features
            dont_transform = [c for c in X.columns if c not in X[variables].columns]
            df_dont_transform = pd.DataFrame(X[dont_transform], columns=dont_transform)
            X = X[variables].copy()

        feature_pool = {c: sympy.symbols(colnames2symbols(c, i), real=True) for i, c in enumerate(variables)}
        if self.max_steps < 1:
            if self.verbose > 0:
                print("[feateng] Warning: no features generated for self.max_steps < 1.")
            return X, feature_pool
        # get a copy of the dataframe - this is where all the features will be added
        df = pd.DataFrame(X.copy(), dtype=np.float32)


        def apply_transformations(features_list):
            # feature transformations
            func_transform = {
                "exp": lambda x: sympy.exp(x),
                "exp-": lambda x: sympy.exp(-x),
                "log": lambda x: sympy.log(x),
                "abs": lambda x: sympy.Abs(x),
                "sqrt": lambda x: sympy.sqrt(x),
                "sin": lambda x: sympy.sin(x),
                "cos": lambda x: sympy.cos(x),
                "2^": lambda x: 2**x,
                "^2": lambda x: x**2,
                "^3": lambda x: x**3,
                "1+": lambda x: 1 + x,
                "1-": lambda x: 1 - x,
                "1/": lambda x: 1 / x
            }
            
            # conditions on the original features that have to be met to apply the transformation
            func_transform_cond = {
                "exp": lambda x: np.all(x < 10),
                "exp-": lambda x: np.all(-x < 10),
                "log": lambda x: np.all(x >= 0),
                "abs": lambda x: np.any(x < 0),
                "sqrt": lambda x: np.all(x >= 0),
                "sin": lambda x: True,
                "cos": lambda x: True,
                "2^": lambda x: np.all(x < 50),
                "^2": lambda x: np.all(np.abs(x) < 1000000),
                "^3": lambda x: np.all(np.abs(x) < 10000),
                "1+": lambda x: True,
                "1-": lambda x: True,
                "1/": lambda x: np.all(x != 0)
            }
            # apply transformations to the features in the given features list
            # modifies global variables df and feature_pool!
            nonlocal df, feature_pool#, units
            # returns a list of new features that were generated
            new_features = []
            uncorr_features = set()
            # store all new features in a preallocated numpy array before adding it to the dataframe
            feat_array = np.zeros((df.shape[0], len(features_list) * len(self.transformations)), dtype=np.float32)
            for i, feat in enumerate(features_list):
                if self.verbose and not i % 100:
                    print("[feateng] %15i/%15i features transformed" % (i, len(features_list)), end="\r")
                for ft in self.transformations:
                    # check if transformation is valid for particular feature (i.e. given actual numerical values)
                    # (don't compute transformations on categorical features)
                    if len(df[feat].unique()) > 2 and func_transform_cond[ft](df[feat]):
                        # get the expression (based on the primary features)
                        expr = func_transform[ft](feature_pool[feat])
                        expr_name = str(expr)
                        # we're simplifying expressions, so we might already have that one
                        if expr_name not in feature_pool:
                            feature_pool[expr_name] = expr
                            # create temporary variable expression and apply it to precomputed feature
                            t = sympy.symbols("t")
                            if expr == "log" and np.any(df[feat] < 1):
                                expr_temp = func_transform[ft](t + 1)
                            else:
                                expr_temp = func_transform[ft](t)
                            f = lambdify(t, expr_temp)
                            new_feat = np.array(f(df[feat].to_numpy()), dtype=np.float32)
                            # near 0 variance test - sometimes all that's left is "e"
                            if np.isfinite(new_feat).all() and np.var(new_feat) > 1e-10:
                                corr = abs(np.corrcoef(new_feat, df[feat])[0, 1])
                                if corr < 1.:
                                    feat_array[:, len(new_features)] = new_feat
                                    new_features.append(expr_name)
                                    # correlation test: don't include features that are basically the same as the original features
                                    # but we only filter them out at the end, since they still might help in other steps!
                                    if corr < correlation_threshold:
                                        uncorr_features.add(expr_name)
            if self.verbose > 0:
                print("[feateng] Generated %i transformed features from %i original features - done." % (len(new_features), len(features_list)))
            df = df.join(pd.DataFrame(feat_array[:, :len(new_features)], columns=new_features, index=df.index, dtype=np.float32))
            return new_features, uncorr_features

        def get_feature_combinations(feature_tuples):
            # new features as combinations of two other features
            func_combinations = {
                "x+y": lambda x, y: x + y,
                "x*y": lambda x, y: x * y,
                "x-y": lambda x, y: x - y,
                "y-x": lambda x, y: y - x
            }
            # get all feature combinations for the given feature tuples
            # modifies global variables df and feature_pool!
            nonlocal df, feature_pool#, units
            # only compute all combinations if there are more transformations applied afterwards
            # additions at the highest level are sorted out later anyways
            if steps == self.max_steps:
                combinations = ["x*y"]
            else:
                combinations = list(func_combinations.keys())
            # returns a list of new features that were generated
            new_features = []
            uncorr_features = set()
            # store all new features in a preallocated numpy array before adding it to the dataframe
            feat_array = np.zeros((df.shape[0], len(feature_tuples) * len(combinations)), dtype=np.float32)
            for i, (feat1, feat2) in enumerate(feature_tuples):
                if self.verbose and not i % 100:
                    print("[feateng] %15i/%15i feature tuples combined" % (i, len(feature_tuples)), end="\r")
                for fc in combinations:
                    expr = func_combinations[fc](feature_pool[feat1], feature_pool[feat2])
                    expr_name = str(expr)
                    if expr_name not in feature_pool:
                        feature_pool[expr_name] = expr
                        # create temporary variable expression to apply it to precomputed features
                        s, t = sympy.symbols("s t")
                        expr_temp = func_combinations[fc](s, t)
                        f = lambdify((s, t), expr_temp)
                        new_feat = np.array(f(df[feat1].to_numpy(), df[feat2].to_numpy()), dtype=np.float32)
                        # near 0 variance test - sometimes all that's left is "e"
                        if np.isfinite(new_feat).all() and np.var(new_feat) > 1e-10:
                            corr = max(abs(np.corrcoef(new_feat, df[feat1])[0, 1]), abs(np.corrcoef(new_feat, df[feat2])[0, 1]))
                            if corr < 1.:
                                feat_array[:, len(new_features)] = new_feat
                                new_features.append(expr_name)
                                # correlation test: don't include features that are basically the same as the original features
                                # but we only filter them out at the end, since they still might help in other steps!
                                if corr < correlation_threshold:
                                    uncorr_features.add(expr_name)
            if self.verbose > 0:
                print("[feateng] Generated %i feature combinations from %i original feature tuples - done." % (len(new_features), len(feature_tuples)))
            df = df.join(pd.DataFrame(feat_array[:, :len(new_features)], columns=new_features, index=df.index, dtype=np.float32))
            return new_features, uncorr_features

        # get transformations of initial features
        steps = 1
        if self.verbose > 0:
            print("[feateng] Step 1: transformation of original features")
        original_features = list(feature_pool.keys())
        uncorr_features = set(feature_pool.keys())
        temp_new, temp_uncorr = apply_transformations(original_features)
        original_features.extend(temp_new)
        uncorr_features.update(temp_uncorr)
        steps += 1
        # get combinations of first feature set
        if steps <= self.max_steps:
            if self.verbose > 0:
                print("[feateng] Step 2: first combination of features")
            new_features, temp_uncorr = get_feature_combinations(list(combinations(original_features, 2)))
            uncorr_features.update(temp_uncorr)
            steps += 1
        while steps <= self.max_steps:
            # apply transformations on these new features
            if self.verbose > 0:
                print("[feateng] Step %i: transformation of new features" % steps)
            temp_new, temp_uncorr = apply_transformations(new_features)
            new_features.extend(temp_new)
            uncorr_features.update(temp_uncorr)
            steps += 1
            # get combinations of old and new features
            if steps <= self.max_steps:
                if self.verbose > 0:
                    print("[feateng] Step %i: combining old and new features" % steps)
                new_new_features, temp_uncorr = get_feature_combinations(list(product(original_features, new_features)))
                uncorr_features.update(temp_uncorr)
                steps += 1
            # and combinations of new features within themselves
            if steps <= self.max_steps:
                if self.verbose > 0:
                    print("[feateng] Step %i: combining new features" % steps)
                temp_new, temp_uncorr = get_feature_combinations(list(combinations(new_features, 2)))
                new_new_features.extend(temp_new)
                uncorr_features.update(temp_uncorr)
                steps += 1
                # update old and new features and repeat
                original_features.extend(new_features)
                new_features = new_new_features

        # sort out all features that are just additions on the highest level or correlated with more basic features
        if self.verbose > 0:
            print("[feateng] Generated altogether %i new features in %i steps" % (len(feature_pool) - len(variables), self.max_steps))
            print("[feateng] Removing correlated features, as well as additions at the highest level")
        feature_pool = {c: feature_pool[c] for c in feature_pool if c in uncorr_features and not feature_pool[c].func == sympy.add.Add}
        cols = [c for c in list(df.columns) if (c in feature_pool) and (c not in X.columns)]  # categorical cols not in feature_pool
        if cols:
            # check for correlated features again; this time with the start features
            corrs = dict(zip(cols, np.max(np.abs(np.dot(StandardScaler().fit_transform(df[cols]).T, StandardScaler().fit_transform(X))/X.shape[0]), axis=1)))
            cols = [c for c in cols if corrs[c] < correlation_threshold]

            # correlation between original variables
            cor_matrix = X.astype('float64').corr().abs() # corelation matrix
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool)) # get upper triangle part
            to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > correlation_threshold)]
            keep = [c for c in upper_tri.columns if c not in to_drop]
        else:
            keep = []
            to_drop = []

        selected = keep + cols if keep else list(variables) + cols
        if self.verbose > 0:
            print("[feateng] Generated a total of %i additional features" % (len(feature_pool) - len(variables)))
            print("[feateng] Drop a total of %i features %s from the original set" % (len(to_drop), to_drop))
        
        final_df = df[selected].join(df_dont_transform)

        return final_df

    def fit(self, X, y=None, correlation_threshold=0.9):
        '''
        Inputs:
        - X: pandas DataFrame with original features in columns
        - y: No need to be passed. It´s here only for compatibility with sklearn pipelines
        - correlation_threshold: threshold to be choosed for eliminating correlated features
        '''
        self.input_data = X
        self.df_fit_stage = self.__engineer_features(X, y=None, correlation_threshold=correlation_threshold)
        self.variables_to_persist = self.df_fit_stage.columns

        return self

    def transform(self, X):
        '''
        Inputs:
        - X: pandas DataFrame with original features in columns
        '''
        # check if data to be tranaformed was used in the fit process, so there´s no reason to reapet it.
        if (list(X.index) == list(self.input_data.index)) and all(X.columns == self.input_data.columns) and not (len(X.compare(self.input_data)) > 1):
            return self.df_fit_stage
        else:
            df_transform_stage = self.__engineer_features(X, y=None, correlation_threshold=1)
            return df_transform_stage[self.variables_to_persist]
