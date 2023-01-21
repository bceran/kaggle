import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def main():
    train_df = pd.read_csv("last_train.csv").set_index("Unnamed: 0")
    y_train = pd.read_csv("train.csv")["target"]
    test_df = pd.read_csv("last_test.csv").set_index("Unnamed: 0")

    class_weights_dict = {0: 0.52119527, 1: 10.29508197}  # in extras.py

    xx_train, xx_test, yy_train, yy_test = train_test_split(train_df, y_train, test_size=0.3, random_state=42,
                                                            stratify=y_train)

    # Random Forest

    # Optimize
    param_grid = {
        'max_depth': [6],
        'min_samples_leaf': [15],
        'min_samples_split': [10],
        'n_estimators': [500],
        'max_features': [8],
        'class_weight': [class_weights_dict]
    }

    rf = RandomForestClassifier()

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=1, scoring="roc_auc")

    grid_search.fit(xx_train, yy_train)

    # Model
    rfc = RandomForestClassifier(bootstrap=True,
                                 max_depth=8,
                                 min_samples_leaf=10,
                                 min_samples_split=10,
                                 max_features=10,
                                 n_estimators=600,
                                 class_weight=class_weights_dict)
    rfc.fit(xx_train, yy_train)

    proba_test = rfc.predict_proba(xx_test)
    proba_predict_list = [list(x) for x in list(proba_test)]
    predict_max_value = [(max(proba_predict_list[x]), proba_predict_list[x].index(max(proba_predict_list[x]))) for x in
                         range(len(proba_predict_list))]

    target_df = pd.DataFrame({"musteri": xx_test.index})
    target_df['target'] = ''

    for i in range(len(predict_max_value)):
        target_df['target'][i] = predict_max_value[i][1]

    print(roc_auc_score(yy_test, np.array(target_df['target'])))

    # Catboost
    cbc = CatBoostClassifier(iterations=10,
                             learning_rate=0.001,
                             depth=6,
                             class_weights=class_weights_dict,
                             custom_metric='AUC:hints=skip_train~false',
                             eval_metric='AUC',
                             early_stopping_rounds=10,
                             loss_function='MultiClass')

    cbc.fit(xx_train, yy_train)

    proba_test = cbc.predict_proba(xx_test)
    proba_predict_list = [list(x) for x in list(proba_test)]
    predict_max_value = [(max(proba_predict_list[x]), proba_predict_list[x].index(max(proba_predict_list[x]))) for x in
                         range(len(proba_predict_list))]

    target_df = pd.DataFrame({"musteri": xx_test.index})
    target_df['target'] = ''

    for i in range(len(predict_max_value)):
        target_df['target'][i] = predict_max_value[i][1]

    print(roc_auc_score(yy_test, np.array(target_df['target'])))


if __name__ == '__main__':
    main()
