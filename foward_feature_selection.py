from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def foward_feature_selection(X_train, 
                             X_test, 
                             y_train, 
                             y_test, 
                             model,
                             metric, 
                             columns=None,
                             steps=20,
                             label=1):
  aceitas = []
  valor_maior_score = 0

  variables = columns if columns else [[item] for item in X_train.columns]
  for step in range(steps):
    var_maior_score = None
    for variable in variables:
      if variable in aceitas:
        continue
     
      model.fit(X_train[aceitas + variable], y_train)
      y_pred = model.predict(X_test[aceitas + variable])

      scores = {'accuracy_score': accuracy_score,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'f1_score': f1_score}

      parameters = {'y_true': y_test, 'y_pred': y_pred} if metric == 'accuracy_score' else {'y_true': y_test, 'y_pred': y_pred, 'pos_label': label}

      score = scores[metric](**parameters)

      if score > valor_maior_score:
        var_maior_score = variable
        valor_maior_score = score

    if var_maior_score is None:
      break

    aceitas = [[item] for item in aceitas]
    aceitas.append(var_maior_score)
    aceitas = [y for x in aceitas for y in x] # flatten

    print(f'Melhor variável: {[var_maior_score[0]]} - Score: {valor_maior_score}' )
    print(aceitas)
    print()
    
  return aceitas