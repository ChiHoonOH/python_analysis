# 부트스트래핑 -> 중복을 허용하는 리샘플링
# bagging은 같은 모델에서 학습 데이터 셋의 subset을 랜덤하게 구성하여 학습 하는
#방식을 말하는데 이때, 샘플링시에 중복을 허용하는 것을 말한다.
# pasting같은 경우 중복 허용 x

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def ex2_Bagging():
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf1 = DecisionTreeClassifier(random_state=1)
    # oob_score : 주어진 데이터셋을 통해 발생된 오차를 수정하기 위해 다른 데이터 셋을 사용할
    # 건지 여부

    eclf = BaggingClassifier(clf1, oob_score=True)

    # 학습한다.
    eclf.fit(X_train, y_train)

    y_pred = eclf.predict(X_test)
    print('정확도:', accuracy_score(y_test, y_pred))