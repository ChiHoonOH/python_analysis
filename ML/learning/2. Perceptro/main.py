import numpy as np
from perceptron import Perceptron
from time import time
import pickle  # byte 형태로 저장가능(표준을 맞춰주기 위해서(?))
# 학습과 테스트를 위해 사용할 데이터


def step1_learning():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])

    ppn = Perceptron(eta=0.1)
    stime = time()
    ppn.fit(X, y)
    etime = time()

    print('학습에 걸린 시간:', (etime-stime))
    print('학습 중 오차가 난 개수:', ppn.errors_)
    
    # 학습이 완료된 객체를 파일로 저장
    with open('perceptron.dat', 'wb') as fp:
        pickle.dump(ppn,fp)
    print('머신러닝 학습 완료')
def step2_using():
    # 파일로 부터 객체를 복원한다.
    with open('perceptron.dat','rb') as fp:
        ppn =  pickle.load(fp)
    while True:
        a1 = input('첫 번째 2진수를 입력해주세요.:')
        a2 = input('두 번째 2진수를 입력해주세요.:')

        X = np.array([int(a1), int(a2)])
        # 계산된 결과를 가져온다.
        result = ppn.predict(X)
        if result == 1:
            print('결과 : 1')
        else:
            print('결과 : 0')
#step1_learning()
print(step2_using())
 
