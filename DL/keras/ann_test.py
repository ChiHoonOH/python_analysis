'''
ANN 구성 및 객체형 / 함수형 and 분산형, 단순형, 연쇄형 구성 조합 사용 
'''

# 1단계 : 모듈 가져오기 
from keras import layers, models
import keras

# layers : 각 계층(입력, 은닉, 출력)을 만드는 모듈
# models : 각 layer을 연결하여 신경망을 만들고 컴파일하고, 학습하는 역할 담당
# models.Model 이 객체의 함수들이(컴파일, ... )
#### Modeling 
# 3단계 
# 3-1, 3-2, 3-3, 3-4 생성
# 3-1 : 분산 방식 모델링을 취하는 함수형 구현 
def ann_models_fun(Nin, Nh, Nout):
    # 신경망의 구조 지
    # 입력 계층 정의 : layers.Input()
    x = layers.Input(shape=(Nin,))
    '''
    class Dense():
        def __call__(self, x):
            print(x)
    -------------
    Dense()(10) # (10) call 함수에의해 print(10)
    > 10
    '''
    # 은닉 계층 정의 : layers.Dense(은닉 계층의 노드수 )(입력노드) => 객체를 함수처럼 사용
    # 은닉 계층 정의 : layers.Dense(Nh)(x)
    # 활성화 함수 : layers.Activation('활성화종류지정')
    h = layers.Activation('relu')(layers.Dense(Nh)(x)) # x 가 입력
    # 출력 계층 구성 
    y = layers.Activation('softmax')(layers.Dense(Nout)(h)) # h가 입력
    # 모델은 입력과 출력을 지정해서 생성, 중간 계층은 알아서 연결관계대로 자동 세팅
    # 모델이 딥러닝과 관계된 여러 함수들을 연계되게 처리 
    model = models.Model(x,y)
    # 컴파일 : 케라스는 compile 수행시 target platform 에 맞는 딥러닝 코드를 구성
    # 엔진이 시에노면 GPU, CPU에 최적화되게 알아서 구성 
    # GPU 엔진이 엔비디아이면 쿠다(CUDA) 컴파일러 실행코드 생성
    # 텐서플로는 초기화 진행
    # loss : 손실함수, 분류 크로스 엔트로피 
    # optimizer : 최적화 - 아담
    # metrics : 학습, 성능 검증, 손실 및 정확도 측정, accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

# 3-2 : 연쇄 방식 모델링을 취하는 함수형 구현
def ann_models_seq_fun(Nin, Nh, Nout):
    # Sequential() 함수를 호출해서 초기화를 가장 먼저 진행
    model = keras.models.Sequential()
    # 모델 구조 설정
    # add() 함수를 통해 설정
    # 입력 계층은 별도로 구성하지 않고 첫번째 은닉계층 작성시 포함. => 동시생성
    model.add(layers.Dense(Nh,activation='relu', input_shape=(Nin,)))
    # 출력계층 Nout, Softmax
    model.add(layers.Dense(Nout, activation='softmax'))
    # => 연쇄 방식은 일일이 in out을 지정하면서 하나하나 연결할 필요 없이
    # add 를 이용해서 추가만 시켜주면 된다.
    # 복잡한 형태는 연쇄형에서는 다소 무리 => 복합형 
    # 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3-3 : 분산방식 모델링을 포함하는 객체 지향형 구현
# ANN의 코드로 재사용성을 높이고, 향후 상속을 통해 좀 더 발전적으로 개선
class ANN_Distri(models.Model): # ?? models.Model 이거 못들음, 그냥 import 해준걸로 하면 안돼나
    def __init__(self,Nin, Nh, Nout):
        # 은닉계층
        # 1개가 아닌 여러개라면 for문 같은 반복문을 이용해서 여러개를 생성 리스트로 담는다. 
        hidden = layers.Dense(Nh)
        # 출력계층 
        output = layers.Dense(Nout)
        # 활성화 함수
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')
        # 입력 계층 생성        
        x = layers.Input(shape=(Nin,))
        # 은닉 계층 생성 
        h = relu(hidden(x))
        # 출력 계층 생성
        y = softmax(output(h))
        super().__init__(x, y)  # x 입력계층 y 출력계층(ann_models_fun을 참조하면 이해가능.)
        # 컴파일 
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    

# 3-3 : 연쇄방식 모델링을 포함하는 객체 지향형 구현
class ANN_Seq(models.Sequential):
    def __init__(self,Nin, Nh, Nout):        
        super().__init__()
        input_layer = layers.Dense(Nh, activation='relu', input_shape=(Nin,))
        output_layer = layers.Dense(Nout, activation='softmax')
        # input층은 크기만 입력해주면 된다.
        self.add(input_layer)
        self.add(output_layer)
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#### Data 
# 4단계 데이터 가져오고, 전처리(순서에 따라서 통상 프로젝트에서는 1~2단계 수준(요구사항 파악이 먼저))
import numpy as np
from keras import datasets
from keras.utils import np_utils  # 정답 레이블의 클래스 개수대로 이진화 처리

# 데이터를 불러오기 -> 데이터 로드 -> 훈련용, 테스트용 분류 -> 이진화처리 -> 정규화 처리

def dataLoad():
    # 데이터로드, 훈련용, 테스트용 분류
    (x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()       
    # 이진화
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print(y_train.shape, y_test.shape)
    # 2차 배열로 구조조정(현재 3차배열) 
    L, W, H = x_train.shape
    x_train = x_train.reshape(-1, W*H)
    x_test = x_test.reshape(-1, W*H)
    # 정규화
    # x_train, x_test => 0~255
    x_train = x_train/255
    x_test = x_test/255
    # 리턴
    return (x_train,y_train), (x_test,y_test)

#### Plotting
import matplotlib.pyplot as plt
def plot_loss(story):
    # 버전에 다라 story 값이 하위에 존재할 수도 있으므로 조건 체크 후 처리
    if not isinstance(story, dict):
        story = story.history
        plt.plot(story['loss'])
        plt.plot(story['val_loss'])
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train data performance','valid data performance'])
        plt.show()

def plot_acc(story):
    if not isinstance(story, dict):
        story = story.history
        plt.plot(story['acc'])
        plt.plot(story['val_acc'])
        plt.title('acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(['train data performance','valid data performance'])
        plt.show()

#### Main Code
def main():
    # 2단계: 매개변수 설정
    # 입력 계층의 노드 수 28*28 = 784
    Nin = 784
    # 은닉 계층의 노드 수 -> 가정(100)
    Nh = 100
    # 출력값이 가지는 클래스 수 (정답의 수 ,분류 케이스 수) -> 0~9 : 10
    number_of_class = 10
    # 출력 계층의 노드 수 
    Nout = number_of_class
    # 3단계 : 모델 획득
    model = ANN_Distri(Nin, Nh, Nout)
    # 4단계: 데이터 로드
    (x_train,y_train), (x_test,y_test) = dataLoad()    
#### Training 
    # validation_split : 검증용 데이터로 20% 훈련용 데이터에서 사용하겠다.
    story = model.fit(x_train,y_train, epochs=15, batch_size=100, validation_split=0.2, verbose=1)
    # 정확도 평가
    perform = model.evaluate(x_test, y_test, batch_size=100)
    print(perform)
    # 시각화를 통한 => 성능추이 살펴보고, 하이퍼파라미터 튜닝, 조기 훈련 종료, 과적합처리(dropout)
    print(story)
    plot_loss(story)
    plot_acc(story)
    # 성능 향상 
# 구동
if __name__ == '__main__':
    main()
