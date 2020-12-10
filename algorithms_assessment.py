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
  metric_name - Uma string representando o nome da métrica escolhida que será o título do gráfico.
  etapa_pipeline - O padrão é -1 para a última etapa do processo, normalmente o algoritmo de ML. Mas é possível também acessar etapas intermediárias, por exemplo, as componentes de uma PCA.

  métodos:
  table - Mostra a tabela com os resultados da validação cruzada
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

  def __computation(self, param_values):
    guardar = []
    for param in param_values:
      # setar o hyperparametro do estimador
      self.estimator[self.etapa_pipeline].set_params(**{self.parametro: param}) if self.pipeline else self.estimator.set_params(**{self.parametro: param})

      validacao_cruzada = cross_validate(self.estimator, self.X, self.y, 
                                        scoring=self.metrica, 
                                        cv = self.validacao, 
                                        return_train_score=True, 
                                        n_jobs=-1)
      
      treino = np.mean(validacao_cruzada['train_score'])
      teste = np.mean(validacao_cruzada['test_score']) 
      hyper = param
      guardar.append((hyper, treino, teste))

    return guardar

  def table(self, param_values):

    '''
    Esse método a tabela com os resultados de treino e validação

    argumentos:
    param_values - lista com os valores ou faixa de valores dos hyper-parâmetros
    '''
    DataFrame = pd.DataFrame(self.__computation(param_values), 
           columns=[self.parametro,'Treino','Validação'])
    return DataFrame

  def complexity_curves(self, param_values,
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

    DataFrame = self.table(param_values)
    melt = pd.melt(DataFrame, 
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
  table - Mostra a tabela com os resultados da validação cruzada
  complexity_curves - Plota o gráfico com a visualização das linhas de complexidade.
  '''

  def __init__(self,X, y, 
              modelo, 
              validacao, 
              metrica, 
              passo, 
              embaralhar=False,
              metric_name='metric'):

      self.X = X
      self.y = y
      self.modelo = modelo
      self.validacao = validacao
      self.metrica = metrica
      self.passo = passo
      self.embaralhar = embaralhar
      self.metric_name = metric_name

  def __computaion(self):
      if self.embaralhar:
        self.X, self.y = shuffle(self.X, self.y)

      guardar = []
      for each in range(self.passo, len(self.X), self.passo):    
        validacao_cruzada = cross_validate(self.modelo, self.X[:each, :], self.y[:each],
                                          scoring=self.metrica,
                                          cv=self.validacao,
                                          return_train_score=True,
                                          n_jobs=-1)
        
        treino = np.mean(validacao_cruzada['train_score'])
        teste = np.mean(validacao_cruzada['test_score'])
        quantidade_exemplos = each

        guardar.append((quantidade_exemplos, treino, teste))

      return guardar

  def table(self):
      dataframe = pd.DataFrame(self.__computaion(), 
        columns=['Quantidade de Exemplos', 
                'Treino',
                'Validação'])

      return dataframe

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

    melt = pd.melt(self.table(),
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
    ax.set_title(self.metric_name.title()) if not isinstance(self.metrica, str) else ax.set_title(self.metrica.title())
    ax.set(ylim=ylim)
 