import numpy as np

# perceptron.py
# Percentron 알고리즘을 구현한 파일


class Perceptron:
    # 생성자 함수
    # thresholds : 임계값, 계산된 예측값을 비교하는 값.
    # eta : 학습률
    # n_iter : 학습 횟수
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
        # ?? 과적합이 되는 이유? learning_rate가 값이 적을때 global minima를 찾을 것 같긴하다
    
    # 가중치 * 입력값의 총합을 계산
    # X : 입력값
    def net_input(self, X):
        # 각 자리의 값과 가중치를 곱한 총합을 구한다.
        a1 = np.dot(X, self.w_[1:]) + self.w_[0]  # 곱 연산이 느리기 때문에 더하기 연산으로
        return a1

    # 예측된 결과를 가지고 판단.
    def predict(self, X):
        #print(self.net_input(X))
        a2 = np.where(self.net_input(X) > self.thresholds, 1, -1)
        return a2

    def fit(self, X, y):  # X 입력 데이터 y 는 결과 데이터.
        # 가중치를 담은 행렬을 생성한다.
        #print('X.shape', X.shape)
        self.w_ = np.zeros(1+X.shape[1])  # 아 처음에 하나를 반드시 추가시켜준다그랫던듯?
        #  => 처음에 가중치를 0으로 초기화
        # 예측값과 실제값을 비교하여 다른 값들(의 예측값)을 담음
        self.errors_= []
        
        #  지정된 학습 횟수 만큼 반복한다.
        
        for _ in range(self.n_iter):  # _는 range 안의 값을 아무 것도 쓰지 않는 용도로 씀
            # 예측값과 실제값이 다른 갯수를 담을 변수
            errors = 0
            # 입력받은 입력값과 결과값을 묶어준다.
            temp1 = zip(X, y)
            # 입력값과 결과값의 묶음을 가지고 반복한다.
            for xi, target in temp1:
                # 입력값을 가지고 예측값을 계산한다.
                a1 = self.predict(xi)
                #입력값과 예측값이 다르면 가중치  계산.
                if target != a1:
                    update = self.eta * (target-a1)
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    # 값이 다른 횟수를 누적한다.
                    errors += int(update != 0.0)
        self.errors_.append(errors)
        print(self.w_)


