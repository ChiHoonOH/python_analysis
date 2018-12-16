import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def ex6_xgboosting():
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 옵션
    # n_jobs : 병렬처리로 사용할 쓰래드 갯수
    # objective : 사용할 머신러닝 알고리즘
    # - 기본 reg: linear / reg:logistic
    # loss : 오차범위 수정을 위한 알고리즘, ls, lad, huber, quantile
    rclf = XGBClassifier(random_state=1)
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html

    # 학습한다.
    eclf = XGBRegressor()
    rclf.fit(X_train, y_train)
    # 예측
    y_pred = rclf.predict(X_test)
    print('정확도:', accuracy_score(y_test, y_pred))
    # 학습
    eclf.fit(X_train, y_train)
    # 예측
    y_pred = eclf.predict(X_test)
    print('정확도:', accuracy_score(y_test, y_pred))

    print('정확도:', 1 - mean_squared_error(y_test, y_pred))