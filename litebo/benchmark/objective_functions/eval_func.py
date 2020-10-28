from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from litebo.model.lightgbm import LightGBM

def eval_func(model, params, x, y):
    params = params.get_dictionary()
    if model == 'lightgbm':
        model = LightGBM(**params)
    elif model == 'liblinear_svc':
        model = LightGBM(**params)
    elif model == 'random_forest':
        model = LightGBM(**params)
    elif model == 'adaboost':
        model = LightGBM(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)