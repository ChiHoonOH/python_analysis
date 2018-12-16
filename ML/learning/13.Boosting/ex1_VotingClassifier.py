# ex1_VotingClassifier

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def ex1_VotingClassifier():
    # 테스트할 데이터를 읽어본다.
    ####타이타닉
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')

    #### iris
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target[:]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

    clf1 = LogisticRegression(random_state=1)
    clf2 = DecisionTreeClassifier(random_state=1)
    clf3 = GaussianNB()
    a1 =[('lr', clf1),('rf', clf2),('gnb', clf3)]
    # voting : hard - 다수결 투표에 의해 결과를 선택
    # voting : soft - 예측된 결과의 평균을 기준으로 선택

    
    eclf = VotingClassifier(estimators=a1, voting='hard')
    # 학습한다.
    print('학습시작')
    eclf.fit(X_train, y_train)
    print('예측시작')
    y_pred = eclf.predict(X_test)
    print('정확도: ', accuracy_score(y_test,y_pred))



