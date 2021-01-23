#encoding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
sn.set_style('darkgrid')
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate

class ComplexityCurves():

    '''
    Avaliador da performance de um modelo de aprendizado de máquina conforme algum hyper-parâmetro vai sendo alterado.

    argumentos:
    X - Pandas dataframe ou numpy array com os atributos.
    y - Atributo alvo.
    modelos - Lista com o conjunto de algoritmos ou pipelines a serem testados.
    parametro - String com nome do hyper-parâmetro que será avaliado.
    metrica - Métrica escolhida para avaliação. Pode ser uma string como: 'accuracy', 'neg_mean_absolute_error' ou feito com a função make_scorer
    validacao - Validação cruzada, pode ser um número inteiro ou uma função; KFold, RepeatedKFold, etc.
    pipeline - Dafault 'False'. Deve ser colocado como 'True' se for usado uma pipeline.
    metric_name - Uma string representando o nome da métrica escolhida que será o título do gráfico quando for usado a função make_scorer.
    etapa_pipeline - O padrão é -1 para a última etapa do processo, normalmente o algoritmo de ML. Mas é possível também acessar etapas intermediárias, por exemplo, as componentes de uma PCA.

    métodos:
    complexity_curves - Plota o gráfico com a visualização das linhas de complexidade.
    '''

    def __init__(self, X, y,
                 estimator,
                 parametro,
                 metrica,
                 validacao,
                 pipeline=False,
                 metric_name='metric',
                 etapa_pipeline=-1):
        
        self.X = X
        self.y = y
        self.estimator = estimator
        self.parametro = parametro
        self.metrica = metrica
        self.validacao = validacao
        self.pipeline = pipeline
        self.metric_name = metric_name
        self.etapa_pipeline = etapa_pipeline
        self.dataframe = None

    def __computation(self, param_values):
        guardar = []
        for param in param_values:
            print(self.parametro, ' = ', param)
            # setar o hyperparametro do estimador
            self.estimator[self.etapa_pipeline].set_params(**{self.parametro: param}) if self.pipeline else self.estimator.set_params(**{self.parametro: param})

            validacao_cruzada = cross_validate(self.estimator, 
                                               self.X, 
                                               self.y, 
                                               scoring=self.metrica, 
                                               cv = self.validacao, 
                                               return_train_score=True, 
                                               n_jobs=-1)

            treino = np.mean(validacao_cruzada['train_score'])
            teste = np.mean(validacao_cruzada['test_score']) 
            hyper = param
            guardar.append((hyper, treino, teste))

            # printing results
            std_treino = np.std(validacao_cruzada['train_score'])
            std_teste = np.std(validacao_cruzada['test_score'])

            print(f'Treino: {np.round(treino, 3)} | Validação: {np.round(teste, 3)}', end=' ') 
            print(f'| std_treino: {np.round(std_treino)} | std_teste: {np.round(std_teste)}\n')

        return guardar

    def complexity_curves(self, 
                          param_values,
                          figsize=(8,5),
                          ylim=None):

        '''
        Esse método cria a visualização
        das curvas de complexidade de treino e validação

        argumentos:
        param_values - lista com os valores ou faixa de valores dos hyper-parâmetros
        figsize - tupla que determina a altura e largura de cada gráfico 
        ylim - tupla com a faixa de valores para o eixo y
        '''

        self.dataframe = pd.DataFrame(self.__computation(param_values), 
                                      columns=[self.parametro,'Treino','Validação'])

        melt = pd.melt(self.dataframe, 
                       id_vars=self.parametro, 
                       value_vars=['Treino','Validação'], 
                       value_name='Score', 
                       var_name='Conjunto')

        f, ax = plt.subplots(figsize=figsize)
        sn.pointplot(x=self.parametro, y='Score', hue='Conjunto', data=melt, ax=ax)
        ax.set_title(self.metric_name.title()) if not isinstance(self.metrica, str) else ax.set_title(self.metrica.title())
        ax.set(ylim=ylim)


class LearningCurves():

    '''
    Avaliador da performance de um modelo de aprendizado de máquina conforme
    a quantidade de dados de treinamento vai aumentando.

    argumentos:
    X - Pandas dataframe ou numpy array com os atributos.
    y - Atributo alvo.
    modelo - Algoritmo ou pipeline
    validacao - Validação cruzada, pode ser um número inteiro ou uma função; KFold, RepeatedKFold, etc.
    metrica - Métrica escolhida para avaliação. Pode ser uma string como: 'accuracy', 'neg_mean_absolute_error' ou feito com a função make_scorer.
    passo -  Quantidade de exemplos acrescentados em cada ciclo de treinamento.
    embaralhar - Deafault 'False', 'True' para embaralhar os dados antes do treinamento.
    metric_name - Uma string representando o nome da métrica escolhida que será o título do gráfico.

    métodos:
    learning_curves - Plota o gráfico com a visualização das linhas de complexidade.
    '''

    def __init__(self,X, y, 
                 modelo, 
                 validacao, 
                 metrica, 
                 step_size, 
                 embaralhar=False,
                 metric_name='metric'):

        self.X = X
        self.y = y
        self.modelo = modelo
        self.validacao = validacao
        self.metrica = metrica
        self.step_size = step_size
        self.embaralhar = embaralhar
        self.metric_name = metric_name
        self.dataframe = None

    def __computaion(self):
        if self.embaralhar:
            self.X, self.y = shuffle(self.X, self.y)

        guardar = []
        for step in range(self.step_size, len(self.X)+1, self.step_size):    
            validacao_cruzada = cross_validate(self.modelo, 
                                               self.X.iloc[:step,:], 
                                               self.y.iloc[:step],
                                               scoring=self.metrica,
                                               cv=self.validacao,
                                               return_train_score=True,
                                               n_jobs=-1)
            
            treino = np.mean(validacao_cruzada['train_score'])
            teste = np.mean(validacao_cruzada['test_score'])
            quantidade_exemplos = step

            guardar.append((quantidade_exemplos, treino, teste))
            print(f'Amostras: {quantidade_exemplos}')
            print(f'Treino: {np.round(treino, 3)} | Validação: {np.round(teste, 3)}\n') 
            
        return guardar

    def learning_curves(self,
                        figsize=(8,5),
                        ylim=None):

        '''
        Esse método cria a visualização
        das curvas de aprendizagem de treino e validação

        argumentos:
        figsize - tupla que determina a altura e largura de cada gráfico 
        ylim - tupla com a faixa de valores para o eixo y
        '''

        self.dataframe = pd.DataFrame(self.__computaion(), columns=['Quantidade de Exemplos', 'Treino','Validação'])

        melt = pd.melt(self.dataframe,
                       id_vars='Quantidade de Exemplos',
                       value_vars=['Treino','Validação'],
                       value_name='Score',
                       var_name='Conjunto')

        f, ax = plt.subplots(figsize=figsize)
        sn.pointplot(x='Quantidade de Exemplos',
                     y='Score', 
                     hue='Conjunto',
                     palette=('green','red'),
                     data=melt,
                     ax=ax)
        ax.set_title(self.metric_name.title()) if not isinstance(self.metrica, str) \
        else ax.set_title(self.metrica.title())
        ax.set(ylim=ylim)
 