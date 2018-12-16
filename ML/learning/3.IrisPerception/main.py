from perceptron import Perceptron
import pandas as pd
import numpy as np
import pickle


def step1_get_data():
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    # 꽃잎 데이터를 추출한다.
    X = df.iloc[0:100, [2, 3]].values
    # print(X[0][0], X[0][1])
    # 꽃 종류 데이터를 추출한다.
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    # y[y == 'Iris-setosa'] = 1
    # y[y == 'Iris-versicolor'] = -1
    # print(y)
    return X, y


def step2_learning():
    ppn = Perceptron(eta=0.1)
    X, y = step1_get_data()
    ppn.fit(X, y)
    print(ppn.errors_)
    print(ppn.w_)
    with open('perceptron.dat', 'wb') as fp:
        pickle.dump(ppn, fp)
    print('학습 완료')


def step3_using():
    with open('perceptron.dat','rb') as fp:
        ppn = pickle.load(fp)

    while True:
        a1 = input('너비를 입력해주세요.:')
        a2 = input('길이를 입력해주세요.:')

        X = np.array([float(a1), float(a2)])
        pred = ppn.predict(X)

        if pred == 1:
            print('결과 : Iris - setosa')
        else:
            print('결과 : Iris - versicolor')




step1_get_data()
step2_learning()

step3_using()