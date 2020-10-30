from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from litebo.model.lightgbm import LightGBM
from litebo.model.adaboost import AdaboostClassifier
from litebo.model.liblinear_svc import LibLinear_SVC
from litebo.model.random_forest import RandomForest


def eval_func(params, x, y, model, *args, **kwargs):
    params = params.get_dictionary()
    if model == 'lightgbm':
        model = LightGBM(**params)
    elif model == 'liblinear_svc':
        model = LibLinear_SVC(**params)
    elif model == 'random_forest':
        model = RandomForest(**params)
    elif model == 'adaboost':
        model = AdaboostClassifier(**params)
    else:
        raise ValueError('Invalid algorithm - %s.' % model)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)
