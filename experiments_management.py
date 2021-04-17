import os
import mlflow
from pyngrok import ngrok
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate

def generate_mlflow_ui():
	"""
    Creates a remote mlflow user interface with ngrok.
    """
	get_ipython().system_raw("mlflow ui --port 5000 &")
	ngrok.kill()
	ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
	print("MLflow Tracking UI:", ngrok_tunnel.public_url, end='\n\n')


def __log_metrics(metrics,
                  metric_name, 
                  y_train, 
                  y_test, 
                  y_estimate_train, 
                  y_estimate_test):
    """
    Record a particular metric score in mlflow.
    It will be called from the __logging function.
    ---------------------------------------

    Parameters
    metrics: dictionary containing the metrics names as keys and the metrics fucnctions as values.
    metrics_name: String representing the name of the metric.
    y_train and y_test: A numpy array or pandas series with the true train and test target values.
    y_estimate_train and y_estimate_test: A numpy array or pandas series with the predicted train and test target values.

    Return four variables representing the metrics names and values.
    """
    # metric name
    score_name_train = f'train_{metric_name}'
    score_name_test = f'test_{metric_name}'

    # metric score
    score_train = metrics[metric_name](y_train, y_estimate_train)
    score_test = metrics[metric_name](y_test, y_estimate_test)

    if metric_name == 'rmse':
        score_train = np.sqrt(score_train)
        score_test = np.sqrt(score_test)

    # metric log
    mlflow.log_metric(score_name_train, score_train)
    mlflow.log_metric(score_name_test, score_test)

    return score_name_train, score_train, score_name_test, score_test


def __logging(metrics,
              y_train,
              y_test, 
              y_pred_train, 
              y_pred_test, 
              y_proba_train=None, 
              y_proba_test=None):
    """
    Creates a dictionary with all the metrics from train and test.
    It will be called from the simple_split function.
    --------------------------------------------------------

    Parameters
    metrics: dictionary containing the metrics names as keys and the metrics fucnctions as values.
    y_train and y_test: The true target values from train and test.
    y_pred_train and y_pred_test: Array with the estimate results from the algorithms.
    y_proba_train and y_proba_test: An array with the probability results from the algorithms.
    
    Returns a dictionary with metrics names and metrics results.
    """
    metrics_scores = {}
    log_metrics_results = None

    for metric_name in metrics.keys():
        args = [metrics, metric_name, y_train, y_test]

        if metric_name not in ['auc', 'log_loss']:
            log_metrics_results = __log_metrics(*args, y_pred_train, y_pred_test)
        else:
            log_metrics_results = __log_metrics(*args, y_proba_train, y_proba_test)
            
        # Unpacking
        score_name_train = log_metrics_results[0]
        score_train = log_metrics_results[1]
        score_name_test = log_metrics_results[2]
        score_test = log_metrics_results[3]

        # Store the scores in a dictionary
        metrics_scores.setdefault(score_name_train, score_train)
        metrics_scores.setdefault(score_name_test, score_test)

    return metrics_scores


def data_artifacts(X_train):
    """
    Creates and stores data artifacts like a sample of the data, the features and indices.
    ---------------------------------------------------

    Parameter
    X_train: The pandas data frame right before it enters the algorithm in the last but one step in the pipeline.
    """
    os.makedirs('artifacts_temp', exist_ok=True)
    
    features = list(X_train.columns)
    indices = list(X_train.index)
    
    with open('artifacts_temp/features.txt', 'w') as features_txt:
        features_txt.write(str(features))
        
    with open('artifacts_temp/indices.txt', 'w') as indices_txt:
        indices_txt.write(str(indices))
    
    X_train.head(10).to_csv('artifacts_temp/X_train_sample.csv', index=False)
    
    mlflow.log_artifacts('artifacts_temp')


def simple_split(*, task, 
                 pipeline, 
                 X, 
                 y, 
                 test_size, 
                 metrics, 
                 random_state):
    """
    Split the data in train and test sets.
    -------------------------------------

    Parameters
    task: string indicating if it is a 'classification' or 'regression' task.
    pipeline: The sklearn pipeline to run.
    X: Dataframe with all the variables.
    y: Target.
    test_size: Size of the test data. It can be a float representing a percentage or an interger.
    metrics: dictionary containing the metrics names as keys and the metrics fucnctions as values.
    random_state: Random number generator for the split in data.

    Returns a dictionary with metrics names and metrics results.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Get the last but one state of the training data.
    if len(pipeline) > 1:
        X_train = pipeline[:-1].fit_transform(X_train)
        pipeline[-1].fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    # Collect data artifacts
    data_artifacts(X_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    if task == 'classification':
        y_proba_train, y_proba_test = None, None
        
        allowed_metrics = ['precision','recall','f1_score','accuracy','auc','log_loss']
        if not set(metrics.keys()).issubset(allowed_metrics):
            raise ValueError(f'Only these metrics are valid: {allowed_metrics}.')

        if any(item in ['auc','log_loss'] for item in metrics.keys()):
            y_proba_train = pipeline.predict_proba(X_train)[:,1]
            y_proba_test = pipeline.predict_proba(X_test)[:,1]

        metrics_scores = __logging(metrics, 
                                   y_train, 
                                   y_test, 
                                   y_pred_train, 
                                   y_pred_test, 
                                   y_proba_train, 
                                   y_proba_test)

    elif task == 'regression':
        allowed_metrics = ['rmse','mae','mape','msle','r2']
        if not set(metrics.keys()).issubset(allowed_metrics):
            raise ValueError(f'Only these metrics are valid: {allowed_metrics}.')

        metrics_scores = __logging(metrics, 
                                   y_train, 
                                   y_test, 
                                   y_pred_train, 
                                   y_pred_test)

    return metrics_scores


def cross_validation(*, pipeline, X, y, cv, metrics):
    """
    Performs cross validation.
    -------------------------

    Parameters
    pipeline: The sklearn pipeline to run.
    X: Dataframe with all the variables.
    y: target.
    cv: Number of folds as an integer or cross validatio procedure e.g KFold, StratifiedKFold from sklearn.
    metrics: dictionary containing the metrics names as keys and the metrics fucnctions as values.

    Returns a dictionary with metrics names and metrics results.
    """
    cross_validation = cross_validate(pipeline, X, y, 
                                      scoring=metrics, 
                                      cv = cv, 
                                      return_train_score=True, 
                                      n_jobs=-1)

    # log metrics
    metrics = [metric for metric in cross_validation.keys() 
                if metric not in ['fit_time','score_time']]

    metrics_scores = {}
    mean_score, max_score, min_score = 0,0,0
    for metric in metrics:
        if metric.endswith('rmse'):
            mean_score = np.sqrt(np.mean(cross_validation[metric]))
            max_score = np.sqrt(np.max(cross_validation[metric]))
            min_score = np.sqrt(np.min(cross_validation[metric]))
        else:
            mean_score = np.mean(cross_validation[metric])
            max_score = np.max(cross_validation[metric])
            min_score = np.min(cross_validation[metric])

        mlflow.log_metric(f'max_{metric}', max_score)
        mlflow.log_metric(f'min_{metric}', min_score)
        mlflow.log_metric(f'mean_{metric}', mean_score)

        metrics_scores.setdefault(metric, mean_score)

    return metrics_scores


def experiment_manager(*, task, 
                       pipeline, X, y, 
                       runs, 
                       validation, 
                       hyperparameters, 
                       metrics, 
                       random_state=0, 
                       remote_ui=False,
                       **kwargs):
    """
    This function runs experiments and records the results.
    -----------------------------------------------------

    Parameters
    task: string indicating if it is a 'classification' or 'regression' task.
    pipeline: The sklearn pipeline to run.
    X: Dataframe with all the variables.
    y: target.
    validation: 'simple_split' or 'cross_validation'.
    hyperparameters: A function returning a dictionary with all the hyperparameters names as keys 
                     and range values to be tested in each algorithm as values.
    metrics: dictionary containing the metrics names as keys and the metrics fucnctions as values.
             For 'cross_validation', the metrics need to be wrapped with the make_scorer function from sklearn.
    random_state: Random number generator for the split in data.
    remote_ui: Interact with mlflow inerface remotely or locally. Set 'True' if you are using google colab or other remote platform.
    available kwargs: run_label -> For optional labelling in the run name.
                      test_size -> When 'simple_split' is chose. For the size of the test set.
                      cv -> When 'cross_validation' is chose. For the cross validation strategy.
    """
    
    experiment_name = pipeline[-1].__class__.__name__
    mlflow.set_experiment(experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    print(f"Experiment Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}", end='\n\n')

    for run in range(runs):
        optional_run_label = kwargs.get('run_label') if kwargs.get('run_label') != None else ''
        with mlflow.start_run(run_name=f'Run: {run+1}{optional_run_label}'):

            # log hyperpatameters
            for hyperpatameter_name, hyperpatameter in hyperparameters().items():
                mlflow.log_param(hyperpatameter_name.split('__')[1], hyperpatameter)

            # training
            pipeline.set_params(**hyperparameters())
            
            # simple split
            if validation == 'simple_split':
                mlflow.set_tag('random_state_split', random_state)
                mlflow.set_tag('test_size', kwargs['test_size'])
                metrics_scores = simple_split(task=task, 
                                              pipeline=pipeline, 
                                              X=X, 
                                              y=y, 
                                              test_size=kwargs['test_size'], 
                                              metrics=metrics, 
                                              random_state=random_state)

            # cross validation
            elif validation == 'cross_validation':
                mlflow.set_tag('cross_validation', kwargs['cv'])
                metrics_scores = cross_validation(pipeline=pipeline,
                                                  X=X, 
                                                  y=y, 
                                                  cv=kwargs['cv'], 
                                                  metrics=metrics)

        # Print results
        print(f'Run {run+1}', end='\n\n')
        print('HYPERPARAMETERS')
        for key, value in hyperparameters().items():
            print(f'{key[key.find("__")+2:]}: {value}')
        print()

        print('SCORES')
        for key, value in metrics_scores.items():
            print(f'{key}: {np.round(value, 3)}')
        print()

    # mlflow user interface
    if remote_ui == True:
        return generate_mlflow_ui()
    elif remote_ui == False:
        print('Type "mlflow ui" in your terminal in order to interact with mlflow user interface.', end='\n\n')