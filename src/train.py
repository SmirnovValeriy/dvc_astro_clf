import os
import sys
import pickle
import yaml
import joblib
from time import time
import pandas as pd
import numpy as np
from train_lib import *
from hyperopt import hp
from hyperopt import Trials, tpe
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score


if len(sys.argv) != 6:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython3 train.py"
                     "model-pkl-path robust-pkl-path df_test-pkl-path\n")
    sys.exit(1)
df = pd.read_csv(sys.argv[1])
df_train_path = sys.argv[2]
df_test_path = sys.argv[3]
model_filepath = sys.argv[4]
robust_filepath = sys.argv[5]

params = yaml.safe_load(open('params.yaml'))
mode = params['featurize']['mode']
overview = params['featurize']['overview']
log_path = params['common']['log_path']

lgb_reg_params = {
    'min_child_samples': hp.randint('min_child_samples', 80) + 1,
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
    'num_leaves': hp.randint('num_leaves', 100) + 10,
    'bagging_freq': hp.randint('bagging_freq', 20),
    'n_estimators': 1000
}
lgb_fit_params = {
    'early_stopping_rounds': 50,
    'verbose': False
}
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['score'] = lambda y, pred: -accuracy_score(y, pred)

stdout_path = os.path.join(log_path, 'train_log.txt')
with open(stdout_path, 'w') as stdout:
    old_sys_stdout = sys.stdout
    sys.stdout = stdout
    X, y = df.drop(columns=['class']).values, df['class'].values
    train_d, test_d = data_preparation(X, y, test_size=0.3, c=50000)
    print(f"Data train shape: {train_d.shape}, data validation shape: {test_d.shape}")

    X1, y1 = train_d[:, :-1], train_d[:, -1].astype('int')
    X2, y2 = test_d[:, :-1], test_d[:, -1].astype('int')
    robust = RobustScaler()

    X_train_norm = robust.fit_transform(X1)
    X_test_norm = robust.transform(X2)
    y_train = y1
    y_test = y2

    obj = HPOpt(X_train_norm, y_train, cv=3)
    lgb_opt = obj.process(fn_name='lgb_reg',
                          space=lgb_para,
                          trials=Trials(),
                          algo=tpe.suggest,
                          max_evals=50)
    print(lgb_opt)
    gb = lgb.LGBMClassifier(**{'colsample_bytree': lgb_opt[0]['colsample_bytree'],
                               'min_child_samples': lgb_opt[0]['min_child_samples'] + 1,
                               'num_leaves': lgb_opt[0]['num_leaves'] + 10,
                               'bagging_freq': lgb_opt[0]['bagging_freq'],
                               'n_estimators': 1000})
    t = time()
    gb.fit(X_train_norm, y_train,
           eval_set=[(X_train_norm, y_train), (X_test_norm, y_test)],
           **lgb_fit_params)
    print(time() - t)
    
    joblib.dump(train_d, df_train_path)
    joblib.dump(test_d, df_test_path)
    joblib.dump(gb, model_filepath)
    joblib.dump(robust, robust_filepath)
    sys.stdout = old_sys_stdout
