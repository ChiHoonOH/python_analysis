from keras import models,layers
# 데이터는 케라스 제공 imdb = 영화 25000건 평(텍스트), 평점 정보(추천1, 비추천0) 제공
# 추천이 많으면 긍정신호, 비추천이 많으면 부정으로 보는
from DataSupport import getData, load_imdb

# 모델링 

# 머신(lstm 수행하여 영화평을 분석 예측)
# 분산형 객체지향형 모델링 방법
class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
        # 입력부터 출력층까지 차례대로 구성
        # maxlen 개로 데이터 크기를 조정함.
        x = layers.Input(shape=(maxlen,)) # ?? 잘 이해 안감 이거 80개로 끊었다는것 
        # 임베디드 계층 : 모델의 첫번째 레이어로만 사용 가능 
        # lstm으로 연결하기 위해 삽입한 층, 출력 백터의 크기가 조정된다.
        h = layers.Embedding(max_features, 128)(x)# ?? 고작 128로 임베딩이 되나?
        # 80 * 128 사이즈로 조정이 된다.
        # 과적합처리 20% 적용, 순환 드롭 20% 적용
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        # 출력계층
        y = layers.Dense(1, activation='sigmoid')(h) # 추천 값은 0 or 1 그래서 하나만
        #있으면 됨 , sigmoid인 이유는 이진분류에 적합하기 때문에 
        # 모델 생성
        super().__init__(x,y)
        # 컴파일
        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class Machine:
    def __init__(self,max_features=20000, maxlen=80):
        # 데이터 로드
        self.data = load_imdb(max_features,maxlen)
        # print((x_train.shape,y_train.shape), (x_test.shape,y_test.shape))
        # 모델생성 
        self.model = RNN_LSTM(max_features,maxlen)
    def play(self, batch_size=32, epochs=3):
        # 훈련 -> 평가 
        (x_train,y_train), (x_test,y_test) = self.data
        self.model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs, validation_split=0.2)
        #validation_data => 검증용 데이터를 직접적으로 지정 validation_data=(x_test,y_test)

# 검증용 데이터가 훈련에 영향을 미치지 않기 때문에(뭔소리??)
        res = self.model.evaluate(x_test,y_test,batch_size=batch_size)
        print("예측결과:",res)

def main():
    m = Machine()
    m.play()    
    

if __name__ == '__main__':
    main()
