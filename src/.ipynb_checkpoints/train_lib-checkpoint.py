import numpy as np
import pandas as pd
from hyperopt import hp, fmin, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def data_preparation(X, y, c=10000, test_size=0.8):

    X1_train, X1_test, y1_train, y1_test = \
        train_test_split(X[y == 1], y[y == 1],
                         test_size=test_size,
                         random_state=43)
    X2_train, X2_test, y2_train, y2_test = \
        train_test_split(X[y == 2], y[y == 2], test_size=test_size,
                         random_state=43)
    X3_train, X3_test, y3_train, y3_test = \
        train_test_split(X[y == 3], y[y == 3],
                         test_size=test_size,
                         random_state=43)

    count = c
    count1 = c

    X_train = np.concatenate((X1_train[:count], X2_train[:count],
                              X3_train[:count]))
    X_test = np.concatenate((X1_test[:count1], X2_test[:count1],
                             X3_test[:count1]))
    y_train = np.concatenate((y1_train[:count], y2_train[:count],
                              y3_train[:count]))
    y_test = np.concatenate((y1_test[:count1], y2_test[:count1],
                             y3_test[:count1]))

    data = np.concatenate((X_train, y_train.reshape((len(y_train), 1))),
                          axis=1)
    np.random.shuffle(data)
    datat = np.concatenate((X_test, y_test.reshape((len(y_test), 1))),
                           axis=1)
    np.random.shuffle(datat)

    return data, datat


def data_split(df_all, column='sdssdr16_r_cmodel',
               label='Label', balance=True):
    df = df_all.sort_values(column)

    def split(df_loc, c=9000):
        data = df_loc.drop(['LabelQ', 'LabelG', 'LabelS', 'Label'],
                           axis=1).values
        data1 = data[::2]
        data2 = data[1::2]
        np.random.shuffle(data1)
        np.random.shuffle(data2)

        return data1[:c//2], data2[:c//2]

    def train_test(X1, X2, X3, test_size=0.1):
        X1_train, X1_test, y1_train, y1_test = \
            train_test_split(X1, 1*np.ones([len(X1), 1]),
                             test_size=test_size, random_state=43)
        X2_train, X2_test, y2_train, y2_test = \
            train_test_split(X2, 2*np.ones([len(X2), 1]),
                             test_size=test_size,
                             random_state=43)
        X3_train, X3_test, y3_train, y3_test = \
            train_test_split(X3, 3*np.ones([len(X3), 1]),
                             test_size=test_size, random_state=43)

        X_train = np.concatenate((X1_train, X2_train, X3_train))
        X_test = np.concatenate((X1_test, X2_test, X3_test))
        y_train = np.concatenate((y1_train, y2_train, y3_train))
        y_test = np.concatenate((y1_test, y2_test, y3_test))

        train = np.concatenate((X_train, y_train.rashape((len(X_train), 1))))
        test = np.concatenate((X_test, y_test.rashape((len(X_test), 1))))

        return train, test

    df_s = df[df[label] == 0]
    df_q = df[df[label] == 1]
    df_g = df[df[label] == 2]

    if balance:
        c = len(df_g)

    else:
        c = 17000

    X1_s, X2_s = split(df_s, c)
    X1_q, X2_q = split(df_q, c)
    X1_g, X2_g = split(df_g, c)

    X1 = np.concatenate((np.concatenate((X1_s, 1*np.ones([len(X1_s), 1])),
                                        axis=1),
                         np.concatenate((X1_q, 2*np.ones([len(X1_q), 1])),
                                        axis=1),
                         np.concatenate((X1_g, 3*np.ones([len(X1_g), 1])),
                                        axis=1)),
                        axis=0)
    X2 = np.concatenate((np.concatenate((X2_s, 1*np.ones([len(X2_s), 1])),
                                        axis=1),
                         np.concatenate((X2_q, 2*np.ones([len(X2_q), 1])),
                                        axis=1),
                         np.concatenate((X2_g, 3*np.ones([len(X2_g), 1])),
                                        axis=1)),
                        axis=0)

    np.random.shuffle(X1)
    np.random.shuffle(X2)

    return X1, X2


class HPOpt(object):

    def __init__(self, X, y, cv=3):
        self.X = X
        self.y = y
        self.cv = cv

    def process(self, fn_name, space, trials, algo, max_evals):

        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo,
                          max_evals=max_evals, trials=trials)
        except Exception as e:
            print({'status': STATUS_FAIL,
                   'exception': str(e)})
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def rf_reg(self, para):
        reg = RandomForestClassifier(**para['reg_params'])
        return self.train_reg(reg, para)

    def lgb_reg(self, para):
        reg = lgb.LGBMClassifier(**para['reg_params'])
        if self.cv > 1:
            return self.train_cv_gb(reg, para)
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        if len(para['fit_params']) > 0:
            reg.fit(self.X, self.y,
                    eval_set=[(self.X, self.y), (self.X, self.y)],
                    **para['fit_params'])
        else:
            reg.fit(self.X, self.y)
        pred = reg.predict(self.X)
        loss = para['score'](self.y, pred)
        return {'loss': loss, 'status': STATUS_OK}

    def train_cv_gb(self, reg, para):
        kf = KFold(n_splits=self.cv, shuffle=False)
        loss = 0
        for train, test in kf.split(self.X):
            if len(para['fit_params']) > 0:
                reg = lgb.LGBMClassifier(**para['reg_params'])
                reg.fit(self.X[train], self.y[train],
                        eval_set=[(self.X[train], self.y[train]),
                                  (self.X[test], self.y[test])],
                        **para['fit_params'])
            else:
                reg.fit(self.X[train], self.y[train])
            pred = reg.predict(self.X[test])
            score = para['score'](self.y[test], pred)
            loss += score

        loss = loss / self.cv
        return {'loss': loss, 'status': STATUS_OK}
