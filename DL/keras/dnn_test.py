from keras import layers,models


# 매개 변수 준비 
# 데이터가 MNIST
Nin = 784
Nh  = [100,50]# 2개의 은닉층을 사용 할 것이다. /100, 50임의의 수로 줄이는 방향 가정
number_of_class = 10
Nout = number_of_class
# 연쇄 방식 모델링을 포함한 모델링 
class DNN(models.Sequential):
    def __init__(self, Nin, Nh, number_of_class, Nout):
        super().__init__()
        # 은닉층이 하나 이상
        # 첫번째 은닉층
        self.add(layers.Dense(Nh[0], activation = 'relu', input_shape=(Nin,), name='HiddenLayer-1'))
        # 두번째 은닉층
        self.add(layers.Dense(Nh[0], activation = 'relu', name='HiddenLayer-2'))
        # 과적합 방지를 위한 dropout을 적용        
        self.add(layers.Dropout(0.2)) #20%의 확률로 출력 노드로 신호를 보내지 않는다.
        # ?? 이게 왜 과적합 방지가 되는가?
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'] )
# 칼라이미지 로드
def dataLoad():
    #www.cs.toronto.edu/~kriz/cifar.html
    from DataSupport import getData
    return getData('color_img')
    

def main():    
    # 데이터 불러오기 
    from ann_test import plot_acc, plot_loss
    (x_train,y_train) , (x_test,y_test) = dataLoad()
    
    # 모델 생성
    model = DNN(Nin,Nh,number_of_class,Nout)
    # 훈련 5세대, 세트 100, 검증 데이터 20%
    story = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2)    
    # 평가  세트 100개로 수행
    perform = model.evaluate(x_test,y_test,batch_size=100)        
    # 시각화
    plot_loss(story)
    plot_acc(story)
    
if __name__ == '__main__':
    main()