# ex4_AdaBoosting.py

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def ex4_AdaBoosting():
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf1 = DecisionTreeClassifier(max_depth=2)
    eclf = AdaBoostClassifier(base_estimator=clf1, n_estimators=500, learning_rate=0.1)
    # 분류기의 갯수는 최대 500개

    # 학습한다.
    eclf.fit(X_train, y_train)
    #예측
    y_pred = eclf.predict(X_test)
    print('정확도:', accuracy_score(y_test, y_pred))

