from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def forward_feature_selection(X_train, 
                             X_test, 
                             y_train, 
                             y_test, 
                             model,
                             metric, 
                             analyse_together=None,
                             steps=20,
                             label=1):
    accepted = []
    greater_score = 0

    if analyse_together:
        analyse_together_flatten = [item_alone for items_together in analyse_together for item_alone in items_together]
        analyse_alone = [[item] for item in X_train.columns if item not in analyse_together_flatten]
        variables = analyse_alone + analyse_together
    else:
        variables = [[item] for item in X_train.columns]

    for step in range(steps):
        variable_greater_score = None
        for variable in variables:
            if variable[0] in accepted:
                continue
         
            model.fit(X_train[accepted + variable], y_train)
            y_pred = model.predict(X_test[accepted + variable])

            scores = {'accuracy_score': accuracy_score,
                      'precision_score': precision_score,
                      'recall_score': recall_score,
                      'f1_score': f1_score}

            parameters = {'y_true': y_test, 'y_pred': y_pred} if metric == 'accuracy_score' else {'y_true': y_test, 'y_pred': y_pred, 'pos_label': label}

            score = scores[metric](**parameters)

            if score > greater_score:
                variable_greater_score = variable
                greater_score = score

        if variable_greater_score is None:
            break

        accepted = [[item] for item in accepted]
        accepted.append(variable_greater_score)
        accepted = [y for x in accepted for y in x] # flatten

        print(f'Round: {step+1}')
        print(f'Variable selected: {variable_greater_score}')
        print(f'Score: {greater_score}\n')
        
    return accepted