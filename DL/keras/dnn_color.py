# 칼라 이미지를 사용하여 분류하는 모델링 DNN
# www.cs.toronto.edu/-kciz/cifar.html

from keras import layers, models


# 매개 변수 준비 
# 데이터가 MNIST

# 연쇄 방식 모델링을 포함한 모델링 
class DNN(models.Sequential):
    def __init__(self, Nin, Nh, Ndo, Nout):
        super().__init__()
        # 은닉층이 하나 이상
        # 첫번째 은닉층
        self.add(layers.Dense(Nh[0], activation='relu', input_shape=(Nin,), name='HiddenLayer-1'))
        self.add(layers.Dropout(Ndo[0]))
        # 두번째 은닉층
        self.add(layers.Dense(Nh[0], activation='relu', name='HiddenLayer-2'))
        # 과적합 방지를 위한 dropout을 적용        
        self.add(layers.Dropout(Ndo[1])) #20%의 확률로 출력 노드로 신호를 보내지 않는다.
        # ?? 이게 왜 과적합 방지가 되는가?
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'] )


def main():    
    # 데이터 불러오기 
    from ann_test import plot_acc, plot_loss
    from DataSupport import getData
    (x_train,y_train) , (x_test,y_test) = getData('colorimg')

    # 입력층 -> 은닉층 -> 드롭아웃 -> 은닉층 -> 드롭아웃 ->출력층
    Nin = x_train.shape[1]
    Nh  = [100, 50]# 2개의 은닉층을 사용 할 것이다. /100, 50임의의 수로 줄이는 방향 가정
    Ndo = [0.1, 0.1] # dropout 2개를 사용하겠다.
    number_of_class = y_train.shape[1]
    Nout = number_of_class




    # 모델 생성
    model = DNN(Nin,Nh,Ndo,Nout)
    # 훈련 5세대, 세트 100, 검증 데이터 20%
    story = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2)    
    # 평가  세트 100개로 수행
    perform = model.evaluate(x_test,y_test,batch_size=100)        
    # 시각화
    plot_loss(story)
    plot_acc(story)
    
if __name__ == '__main__':
    main()