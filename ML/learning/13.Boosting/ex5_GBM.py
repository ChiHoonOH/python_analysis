# ex5_GBM
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def ex5_GBM():
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    eclf = GradientBoostingClassifier()

    # 옵션
    # loss : 오차범위 수정을 위한 알고리즘, ls, lad, huber, quantile
    rclf = GradientBoostingRegressor(n_estimators=500, loss='lad')
    # 분류기의 갯수는 최대 500개

    # 학습한다.
    eclf.fit(X_train, y_train)
    # 예측
    y_pred = eclf.predict(X_test)
    # print('정확도:', accuracy_score(y_test, y_pred))
    print('정확도:', 1 - mean_squared_error(y_test,y_pred))
