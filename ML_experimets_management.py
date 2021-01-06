import datetime
import json
import pytz 

IST = pytz.timezone('Brazil/East') 

def training_tracking(original, storage={}):
    def wrapper_function(*args, **kwargs):
        print('Tracking experiments in {}'.format(original.__name__))
        scores, pipeline, features, test_size, random_state = original(*args, **kwargs)

        day = datetime.date.today().strftime("%d %B %Y")
        hour = datetime.datetime.now(IST).strftime("%H:%M:%S")

        experiment = {'metrics': scores,
                                    'pipeline': pipeline,
                                    'features': features,
                                    'test_size': test_size,
                                    'random_state_split': random_state}

        storage.setdefault(day, {}).setdefault(hour, experiment)

        with open('ml_experiments_management.json', 'a+') as output:
            storage_json = {day:{hour:experiment}}
            output.write('{}\n'.format(json.dumps(str(storage_json))))

        return storage
    return wrapper_function

@training_tracking
def training(model, X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores = {'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='micro')}

    print('Accuracy:', scores['accuracy'],
                '\nPrecision:', scores['precision'])
    
    return scores, model, X.columns, test_size, random_state