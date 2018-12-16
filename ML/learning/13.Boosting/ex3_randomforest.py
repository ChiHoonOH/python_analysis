# bagging + decision_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def ex3_RandomForest():
    iris = load_iris()
    X = iris.data[:,[2,3]]
    y = iris.target[:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf1 = DecisionTreeClassifier(random_state=1)
    # oob_score : 주어진 데이터셋을 통해 발생된 오차를 수정하기 위해 다른 데이터 셋을 사용할
    # 건지 여부

    eclf = BaggingClassifier(clf1, random_state=1)

    clf2 = RandomForestClassifier(random_state=1)

    # oob_score = True, 학습하지 않는 샘플을 남겨두고, 이 샘플을 이용해서 모형을 평가한다. 
    # 약 37퍼 validation_set라고 생각하면됨.
    # 학습한다.
    eclf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    y_pred = eclf.predict(X_test)
    y_pred_random = clf2.predict(X_test)

    print('bagging 정확도:', accuracy_score(y_test, y_pred))
    print('Randomforest 정확도:', accuracy_score(y_test, y_pred_random))