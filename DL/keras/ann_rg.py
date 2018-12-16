# 보스턴 집값을 하는 회귀 ANN 구현
# 보스턴 인근 주택의 판매 데이터를 활용하여 예측하는 모델을 생성
# 좋은 모델인지는 loss를 보고 판단.
#########################################################################
# 1. 모듈 가져오기  
from keras import layers, models
# 2. 데이터 전처리 => 비지도 스케일링(최대최소)
from sklearn import preprocessing
# 시각화  
import matplotlib.pyplot as plt
from ann_test import plot_acc, plot_loss
from keras import datasets
# 모델링  
# 분산 방식 모델링 객체 지향적 구현
class ANN(models.Model):
    def __init__(self, Nin,Nh,Nout):
        hidden = layers.Dense(Nh)
        # 출력계층 
        output = layers.Dense(Nout)
        # 활성화 함수
        relu = layers.Activation('relu')        
        #입력
        x = layers.Input(shape=(Nin,))
        #히든
        h = relu(hidden(x))
        # 활성화 함수를 사용하지 않았음.
        # 원하는 값을 연속적으로 바로 예측하는 회귀방식에서는 통상적으로 출력 노드에 활성화 함수를 사용하지 않는다.
        #출력 
        y = output(h)

        super().__init__(x,y)
        # 손실함수 : 평균 제곱근 오차
        # 최적화 함수: SGD(확률적 경사 하강법)
        self.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])



# 데이터로드  
def load_data():
    (x_train,y_train), (x_test,y_test) =datasets.boston_housing.load_data()
    print((x_train.shape,y_train.shape), (x_test.shape,y_test.shape))
    print(x_train[:5])
    print('*'*100)
    print(y_train[:5])
    scler = preprocessing.MinMaxScaler()
    # 스케이링 작업
    x_train = scler.fit_transform(x_train)
    x_test = scler.fit_transform(x_test)
    return (x_train,y_train), (x_test,y_test)
    
# 메인 코드  
def main():
    # 매개 변수 설정    
    Nin = 13
    Nh = 5 #(임의설정)
    Nout = 1 # 출력층 1개(회귀를 통해 얻는 결과값을 직접 예측하므로 출력 계층의 
    # 길이는 1로 설정)
    # 모델링 생성
    model = ANN(Nin, Nh, Nout)

    (x_train,y_train), (x_test,y_test) = load_data()
    # 학습
    history = model.fit(x_train, y_train, epochs=100, batch_size=100, 
                                        validation_split=0.2, verbose=2)
    # history = history.history
    # 평가 
    performance = model.evaluate(x_test, y_test, batch_size=100)
    print('평가 결과:', performance)

    # 시각화
    plot_loss(history)

    # 손실율이 한자리수나, 0.~ 단위로 내려가게끔, 튜닝, 데이터 확보 등의 작업이 필요하다.
    # 손실율 자체가 내려가는 구조라서 나쁜 모델은 아니나, 일반적인 손실값을 대비해보면
    # 보다 데이터를 확보하고, 하이퍼파라미터 튜닝이나 기타 활성화 함수 등의 변경을 통한
    # 비교 모델링이 더 필요해보임.
    
# 코드 수행   
if __name__ == '__main__':
    main()