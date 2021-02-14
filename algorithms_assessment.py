#encoding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style('darkgrid')
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate

class ValidationCurves():

    '''
    Assessment of the performace of machine learning models.

    parameters:

    X - Pandas dataframe.

    y - target.

    estimator - The machine learning algorithm choosed.

    hyperparameter - String with the name of the hyperparameter that will be evaluated.

    metric - Metric choosed for the assessment. It can be a string like: 'accuracy', 'neg_mean_absolute_error' or 
             build with the make_scorer function.

    validation - cross validation, it can be a integer number or a function like: KFold, RepeatedKFold, etc.

    pipeline - Dafault 'False'. it need to be set as 'True' if a pipeline is used.

    metric_name - String representing the name of the choosed metric. 
                  It will be the title of the plot when thre make_scorer function is used.

    pipeline_step - The defalut is -1 for the last step in the process, usualy the ML algorithm. 
                    But it is possible to access the intermediarie steps, for instance, the components of a PCA.

    method:
    validation_curves - Plot the graph with the conplexity lines visualization.
    '''

    def __init__(self, X, y,
                 estimator,
                 hyperparameter,
                 metric,
                 validation,
                 pipeline=False,
                 metric_name='metric',
                 pipeline_step=-1):
        
        self.X = X
        self.y = y
        self.estimator = estimator
        self.hyperparameter = hyperparameter
        self.metric = metric
        self.validation = validation
        self.pipeline = pipeline
        self.metric_name = metric_name
        self.pipeline_step = pipeline_step
        self.table = None

    def __computation(self, param_values):
        guardar = []
        for param in param_values:
            self.estimator[self.pipeline_step].set_params(**{self.hyperparameter: param}) if self.pipeline else self.estimator.set_params(**{self.hyperparameter: param})

            validacao_cruzada = cross_validate(self.estimator, 
                                               self.X, 
                                               self.y, 
                                               scoring=self.metric, 
                                               cv = self.validation, 
                                               return_train_score=True, 
                                               n_jobs=-1)

            treino = np.mean(validacao_cruzada['train_score'])
            teste = np.mean(validacao_cruzada['test_score']) 
            hyper = param
            guardar.append((hyper, treino, teste))

            print(self.hyperparameter, ' = ', param)
            print(f'Train: {np.round(treino, 3)} | Validation: {np.round(teste, 3)}\n')

        return guardar

    def validation_curves(self, 
                          param_values,
                          figsize=(8,5),
                          ylim=None):

        '''
        This method generates the plot

        parameters:
        param_values - list containing the hyperparameters values to be tested.
        figsize - tuple that determines the height and width of the plot. 
        ylim - tuple for the range of values in the y axis.
        '''
        
        self.table = pd.DataFrame(self.__computation(param_values), 
                                  columns=[self.hyperparameter,'Train','Validation'])

        melt = pd.melt(self.table, 
                       id_vars=self.hyperparameter, 
                       value_vars=['Train','Validation'], 
                       value_name='Score', 
                       var_name='Set')

        f, ax = plt.subplots(figsize=figsize)
        sn.pointplot(x=self.hyperparameter, y='Score', hue='Set', data=melt, ax=ax)
        ax.set_title(self.metric_name.title()) if not isinstance(self.metric, str) else ax.set_title(self.metric.title())
        ax.set(ylim=ylim)


class LearningCurves():

    '''
    Evaluates the performace of a machine learning estimator based on the increase of the sample size.

    parameters:

    X - Pandas dataframe.

    y - target.

    estimator - Algorithm or pipeline.

    validation - cross validation, it can be a integer number or a function like: KFold, RepeatedKFold, etc.

    metric - Metric choosed for the assessment. It can be a string like: 'accuracy', 'neg_mean_absolute_error' or 
             build with the make_scorer function.

    step_size -  Sample size that will be add in each training cycle.

    shuffle - Deafault 'False', 'True' inr order to shuffle the data.

    metric_name - String representing the name of the choosed metric. 
                  It will be the title of the plot when thre make_scorer function is used.

    method:
    learning_curves - plot.
    '''

    def __init__(self,X, y, 
                 estimator, 
                 validation, 
                 metric, 
                 step_size, 
                 shuffle=False,
                 metric_name='metric'):

        self.X = X
        self.y = y
        self.estimator = estimator
        self.validation = validation
        self.metric = metric
        self.step_size = step_size
        self.shuffle = shuffle
        self.metric_name = metric_name
        self.table = None

    def __computaion(self):
        if self.shuffle:
            self.X, self.y = shuffle(self.X, self.y)

        guardar = []
        for step in range(self.step_size, len(self.X)+1, self.step_size):    
            validacao_cruzada = cross_validate(self.estimator, 
                                               self.X.iloc[:step,:], 
                                               self.y.iloc[:step],
                                               scoring=self.metric,
                                               cv=self.validation,
                                               return_train_score=True,
                                               n_jobs=-1)
            
            treino = np.mean(validacao_cruzada['train_score'])
            teste = np.mean(validacao_cruzada['test_score'])
            quantidade_exemplos = step

            guardar.append((quantidade_exemplos, treino, teste))

            print(f'Samples: {quantidade_exemplos}')
            print(f'Train: {np.round(treino, 3)} | Validation: {np.round(teste, 3)}\n') 
            
        return guardar

    def learning_curves(self,
                        figsize=(8,5),
                        ylim=None):

        '''
        This method generates the plot

        parameters:
        figsize - tuple that determines the height and width of the plot. 
        ylim - tuple for the range of values in the y axis.
        '''

        self.table = pd.DataFrame(self.__computaion(), columns=['Sample size', 'Train','Validation'])

        melt = pd.melt(self.table,
                       id_vars='Sample size',
                       value_vars=['Train','Validation'],
                       value_name='Score',
                       var_name='Set')

        f, ax = plt.subplots(figsize=figsize)
        sn.pointplot(x='Sample size',
                     y='Score', 
                     hue='Set',
                     palette=('green','red'),
                     data=melt,
                     ax=ax)
        ax.set_title(self.metric_name.title()) if not isinstance(self.metric, str) \
        else ax.set_title(self.metric.title())
        ax.set(ylim=ylim)
 