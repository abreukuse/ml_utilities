class ColumnsSelector:
  '''
  Adiociona ou remove variáveis de um dataframe para 
  aplicação de algoritimos de aprendizado de máquina.
  
  Parâmetros
  ----------
  
  data: pandas dataframe
  
  inicializer: de início deve ser uma lista vazia, após isso, 
               as execuções da função irão adicionar ou remover itens dessa 
               os nomes das colunas escolhidas
  '''
  def __init__(self, data, initializer=[]):
    self.data = data
    self.initializer = initializer

  def features(self, insert=None, remove=None):
    '''
    insert: lista com os nomes das colunas que devem ser adicionadas
    
    remove: lista com os nomes das colunas que devem ser removidas
    ''' 

    if (insert != None) and \
            all(element in self.data.columns for element in insert) and \
                        (not all(element in self.initializer for element in insert)):

      for cada in insert:
        self.initializer.insert(len(self.initializer), cada)
        
    # Verify columns name that do not consist in the data provided
    elif insert !=None and not all(element in self.data.columns for element in insert):
      
      off_items = [element for element in insert if element not in self.data.columns]
      print('This elements: {}. Were not found in the data columns provided.'.format(off_items))
    
    if (remove != None):
      for cada in remove:
        self.initializer[:] = sorted(set(self.initializer), key=self.initializer.index)
        self.initializer.remove(cada)
    
    result = self.data[sorted(set(self.initializer), key=self.initializer.index)].copy()
    return result