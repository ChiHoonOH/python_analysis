# ex4_learning.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from time import time
import numpy as np


def ex4_learning():
    iris = load_iris()
    X = iris.data[:150, : ]
    y = iris.target[:150]

    # 학습 데이터와 테스트 데이터로 나눈다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 표준화 작업 : 데이터들을 표준정규분포로 변환하여 적은 학습 횟수와 높은 학습 정확도를
    # 갖기 위해 하는 작업.
    '''
    stime = time()
    sc = StandardScaler()
    # 데이터를 표준화한다.
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    sc.fit(X_test)
    X_test_std = sc.transform(X_test)
    # 학습한다.
    # ml = Perceptron(eta0=0.01, max_iter=40, random_state=0)
    ml = LogisticRegression(C=1000.0, random_state=0)
    ml.fit(X_train_std, y_train)
    dtime = time()
    y_pred = ml.predict(X_test_std)
    '''
    #차원 축소를 이용한 학습

    pca1 = PCA(n_components=1)
    X_low = pca1.fit_transform(X_train)
    stime = time()
    ml = LogisticRegression(C=1000.0, random_state=0)
    ml.fit(X_low, y_train)
    dtime = time()
    print('학습시간:',dtime - stime)

    pca2 = PCA(n_components=1)
    X_low2 = pca2.fit_transform(X_test)
    y_pred = ml.predict(X_low2)

    # print('학습시간:',dtime - stime)
    # 학습 정확도 측정을 위해 예측값을 가져온다.

    print('학습정확도:', accuracy_score(y_test, y_pred))
    # 학습이 완료된 객체를 저장한다.
    # with open('ml.dat', 'wb') as fp:
    #     pickle.dump(sc, fp)
    #     pickle.dump(ml, fp)
    # print('학습 저장까지 완료.')



