# 패키지 불러오기 
from keras import models, layers
import keras 
from keras.datasets import mnist
from keras.utils import np_utils 
import matplotlib.pyplot as plt
# 모델 클래스(분산방식,연쇄방식)

# 분산방식은 각각을 붙이는 작업이 필요했음.
# 1 입력층
# 1히든 노드 
# 1 출력층
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

class distri_method(models.Model):
    def __init__(self,Nin,Nh, Nout):  # 입력층, 히든층, 출력층 갯수
        input_layer = layers.Input(shape=(Nin,))
        hidden_layer = layers.Dense(Nh)
        output_layer = layers.Dense(Nout)

        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = input_layer
        h = relu(hidden_layer(x))
        y = softmax(output_layer(h)) 

        super().__init__(x,y)
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
# 연쇄방식은 각각을 붙이는 작업이 필요 없음. add라는 메서드를 통해서 추가함.
class chain_method(models.Sequential):
    def __init__(self):
        pass


def load_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train_dim = x_train.shape
    x_test_dim = x_test.shape

    x_train = x_train.reshape(-1,x_train_dim[1]*x_train_dim[2])
    x_test = x_test.reshape(-1,x_test_dim[1]*x_test_dim[2])

    x_train = x_train/255
    x_test = x_test/255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print('x_train.shape, x_test.shape',x_train.shape, x_test.shape)
    print('y_train.shape, y_test.shape',y_train.shape, y_test.shape)
    return (x_train,y_train),(x_test,y_test)
# main 함수
def main():
    # 데이터 수집 
    Nin = 784 
    Nh  = 100
    Nout = 10
    (x_train,y_train),(x_test,y_test) = load_data()

    
    # 모델
    model = distri_method(Nin,Nh,Nout)
    # 데이터 학습
    history = model.fit(x_train,y_train,batch_size=100,epochs=35,validation_split=0.2,verbose=1)

    # 평가
    model.evaluate(x_test,y_test,batch_size=100,verbose=1)
    # 시각화
    plot_acc(history)
    plot_loss(history)

if __name__ == '__main__':
    main()
