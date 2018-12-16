# ex1_VotingClassifier
import numpy as np
from sklearn.model_selection import train_test_split

def ex1_VotingClassifier() :
    # 테스트할 데이터를 읽어온다
    X = np.load('./tatanic_X_train.npy')
    y = np.load('./tatanic_y_train.npy')
    # print(X)
    # print(y)
    # 학습 데이터와 테스트 데이터로 나눈다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)