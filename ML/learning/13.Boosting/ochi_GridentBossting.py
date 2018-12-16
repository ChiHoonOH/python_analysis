import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

###  입력 파라미터
# 데이터
X = np.load('./tatanic_X_train.npy')
y = np.load('./tatanic_y_train.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# 기본 모델 적합, 참값 - 그 모델 예측값을 통해서 다시 적합.
#

model1 = LinearRegression()
model1.fit(X_train, y_train)

y1 = model1.predict(X_train)
err1 = y_train-y1
model2 = LinearRegression()
model2.fit(X_train, err1)

y2 = model2.predict(X_train)
err2 = err1 - y2
model3 = LinearRegression()
model3.fit(X_train, err2)
y3 = model3.predict(X_train)

print(y_test)
print(sum([y1,y2,y3]))

class ochi_GredientBossting():
    def __init__(self,iter_num=100, model=LinearRegression, learning_rate=0.1):
        self.iter_num = iter_num
        self.model = model()
        self.learning_rate = learning_rate

    def fit(self, x, y):
        err = y
        base_estimator = self.model
        result = base_estimator.fit(x, y)
        if self.iter_num == 1:
            return result
        else:
            # >=2 이것만 반복해야함.
            err = y - base_estimator.predict(x)
            return base_estimator.fit(x, err)








# 반복횟수
# 선택하는 모델에 대한 값들도 받는다.




# 내부


def fit():
    pass
func().fit(X_train,y_train)