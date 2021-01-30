import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from collections import Counter
def mode(a):
    data = Counter(a)
    data_list = dict(data)
    sort_values = sorted(data.values(), reverse=True)
    mode_val = [item for item, freq in data.items() if freq == sort_values[0]][0]

    if pd.isna(mode_val) and (len(sort_values) > 1): 
        mode_val = [item for item, freq in data.items() if freq == sort_values[1]][0]

    return mode_val
class GroupImputer(BaseEstimator, TransformerMixin):
    '''
    Fazer imputação de dados faltantes de acordo com um agrupamento
    ---------------------------------

    parâmetros

    grouping: Coluna categórica com os níveis que formarão os grupos

    columns: lista contendo quais colunas numéricas devem ser imputadas

    strategy: Que estratégia de imputação deve ser implementada. Opções: {'mean', 'median', 'mode'}. Padrão 'mean'.
    '''

    def __init__(self, grouping, columns, strategy = 'mean'):

        self.grouping = grouping
        self.columns = columns
        self.strategy = strategy
        self._dict_result = None
            
    def fit(self, X, y=None):

        method = np.nanmedian if self.strategy == 'median' else mode if self.strategy == 'mode' else np.mean
        imputer = X.groupby([self.grouping])[self.columns].aggregate(method)
        
        # é preciso verificar se algum grupo não possui nenhum valor e substituir os valores faltantes pela mediana geral de cada coluna
        if imputer.isnull().sum().sum() > 0:
            dici_impute = {column : method(imputer[column]) for column in imputer.columns if imputer[column].isnull().any()}
            imputer.fillna(dici_impute, inplace=True)

        self._dict_result = imputer.to_dict(orient='index')

        return self

    def transform(self, X):
        impute_storage = []
        groups = X[self.grouping].unique()
        for group in groups:
            impute = X[X[self.grouping] == group].fillna(self._dict_result[group])
            impute_storage.append(impute)

        X = pd.concat(impute_storage)
        
        return X