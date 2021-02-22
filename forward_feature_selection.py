from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error)
import operator

def forward_feature_selection(X_train, 
                              X_test, 
                              y_train, 
                              y_test, 
                              model,
                              task_type,
                              metric,
                              probability=False, 
                              analyse_together=None,
                              steps=20,
                              greater_is_better=True,
                              **kwargs):
    """
    This function performs a variable selection procedure and returns the ones that were selected. 
    _______________________________________________________________________
    Parameters:

    {X_train, X_test, y_train, y_test}: pandas dataframe and pandas series as usual.

    model: Single model or a entire sklearn pipeline.

    task_type: {'classification', 'regression'}

    metric: Classification {'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score', 'log_loss'}.
            Regression: {'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error'}.

    probability: Boolean. Estimate the y_pred as a float or not. This is required for some classification metrics. Default: False.

    analyse_together: List of lists containing what variables should be considered together in the selection procedure. 
                      An use case could be a one-hot encoded variable.

    steps: How many rounds should the selection procedure take. Default is 20.

    greater_is_better: For the metric choosed, is it better a greater or lower score? Default is True.

    **kwargs: Key word arguments that should be included for the metric choosed. For instance, 'pos_label' and 'average'.
    """
    accepted = []
    greater_score = 0 if greater_is_better else 1_000_000_000_000

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
            y_pred = model.predict_proba(X_test[accepted + variable])[:, 1] if probability else model.predict(X_test[accepted + variable])

            scores = {'classification':{'accuracy_score': accuracy_score,
                                        'precision_score': precision_score,
                                        'recall_score': recall_score,
                                        'f1_score': f1_score,
                                        'roc_auc_score': roc_auc_score,
                                        'log_loss': log_loss},
                                        
                      'regression':{'mean_absolute_error': mean_absolute_error,
                                    'mean_squared_error': mean_squared_error,
                                    'mean_squared_log_error': mean_squared_log_error,
                                    'median_absolute_error': median_absolute_error}
                     }

            args = [y_test, y_pred]

            score = scores[task_type][metric](*args, **kwargs)

            sign = operator.gt if greater_is_better else operator.lt
            if sign(score, greater_score):
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