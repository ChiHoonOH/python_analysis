# 학습용 데이터
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 데이터 표준화
from sklearn.preprocessing import StandardScaler
# 로지스틱
from sklearn.linear_model import LogisticRegression
# SVC
from sklearn.svm import SVC
# Decision Tree
from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score
import pickle
import mylib.custom as cus

from mylib.plotdregion import *

X = None
y = None
names = None


def step1_get_data():
    # 아이리스 데이터 추출
    iris = datasets.load_iris()
    print(iris)
    # 꽃 정보 데이터 추출
    X = iris.data[:, [2, 3]]  # 꽃잎 정보
    y = iris.target[:]  # 꽃 종류
    # X = iris.data[:150, [2,3]]  # 꽃잎 정보
    # y = iris.target[:150]   # 꽃 종류
    print(y)
    names = iris.target_names[:3]   # 꽃 이름.
    print(names)
    return X, y


def step2_learning():
    X, y = step1_get_data()
    # 학습 데이터와 테스트 데이터로 나눈다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 표준화 작업 : 데이터들을 표준정규분포로 변환하여 적은 학습 횟수와 높은 학습 정확도를
    # 갖기 위해 하는 작업.
    sc= StandardScaler()
    # 데이터를 표준화한다.
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    sc.fit(X_test)
    X_test_std = sc.transform(X_test)
    # 학습한다.
    # ml = Perceptron(eta0=0.01, max_iter=40, random_state=0) #1.0
    # ml = LogisticRegression(C=1000.0, random_state=0) #0.82222
    # kernel = '알고리즘 종류' linear, poly, rbf, sigmoid => default : rbf
    # C : default : 1.0
    # ml = SVC(kernel='linear', C=1.0, random_state=0)  #0.9333
    ml = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)  # 0.9555
    # ml = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)  # 0.8889
    ml.fit(X_train_std, y_train)
    # 학습 정확도 측정을 위해 예측값을 가져온다.
    # y_pred = ml.predict(X_test) 0.2444
    y_pred = ml.predict(X_test_std)
    print('학습정확도:', accuracy_score(y_test, y_pred))
    # 학습이 완료된 객체를 저장한다.
    with open('ml.dat', 'wb') as fp:
        pickle.dump(sc, fp)
        pickle.dump(ml, fp)
    print('학습 저장까지 완료.')


    # 시각화를 위한 작업
    cus.matplot_hangul()
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train,y_test))
    # plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=ml, test_idx=range(0,100), title='perceptron')
    # plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=ml, test_idx=range(0,100), title='Logistic')
    # plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=ml, test_idx=range(0, 100), title='SVM')
    plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=ml, test_idx=range(0, 100), title='Decision_Tree')


def step3_using():
    # 학습이 완료된 객체를 복원한다.
    with open('ml.dat', 'rb') as fp:
        sc = pickle.load(fp)
        ml = pickle.load(fp)
    a1 = input('꽃 잎의 너비를 입력해주세요.:')
    a2 = input('꽃 잎의 길이를 입력해주세요.:')

    X = np.array([[float(a1), float(a2)]])
    X_std = sc.transform(X)
    print(X)
    # 표준화 작업
    sc = StandardScaler()
    y = ml.predict(X)
    print(y)
    if y[0] == 0:
        print('Iris-setosa')
    else:
        print('Iris-versicolor')


# step1_get_data()
step2_learning()
# step3_using()


