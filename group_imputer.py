import pandas as pd
import numpy as np
from scipy.stats import mode

class GroupImputer():

  def __init__(self, grouping, columns, strategy = 'mean'):
    self.grouping = grouping
    self.columns = columns
    self.strategy = strategy
    
  def fit(self, X, y=None):
    method = np.median if self.strategy == 'median' else mode if self.strategy == 'mode' else np.mean
    imputer = X.groupby([self.grouping])[self.columns].aggregate(method)

    # se a moda for escolhida é necessário um passo a mais de processamento
    if self.strategy == 'mode':
      for column in self.columns:
        imputer[column] = imputer[column].map(lambda x: x[0][0])
    
    # verifica se algum grupo não possui nenhum valor e substituir os valores faltantes pela mediana de cada coluna
    if X[self.columns].isnull().sum().sum() > 0:
      dici_impute = {column : imputer[column].aggregate(np.median) for column in imputer.columns if imputer[column].isnull().any()}
      imputer.fillna(dici_impute, inplace=True)

    self.dict_result = imputer.to_dict(orient='index')
    return self

  def transform(self, X, y=None):
    impute_storage = []
    groups = sorted(X[self.grouping].unique())
    for group in groups:
      impute = X[X[self.grouping] == group].fillna(self.dict_result[group])
      impute_storage.append(impute)

    X = pd.concat(impute_storage)
    X.sort_index(inplace=True)
    
    return X