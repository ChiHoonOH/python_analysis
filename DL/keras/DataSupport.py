####################인공신경망에 필요한 데이터 공급 클래스 혹은 모듈###########

from keras import datasets  # mnist 제공
from keras.utils import np_utils  # 정답 레이블의 클래스 개수대로 이진화처리
from keras.preprocessing import sequence
def getData(style):
    if style == 'colorimg':
        return color_img()
    elif style =='mnist':
        return mnist()
    elif style =='boston_house':
        return boston_house()
    elif style =='imdb':
        return load_imdb()
    else:
        return None


# 데이터는 케라스 제공 imdb = 영화 25000건 평(텍스트), 평점 정보(추천1, 비추천0) 제공
# 추천이 많으면 긍정신호, 비추천이 많으면 부정으로 보는
def load_imdb(max_features=20000, maxlen=80):
    (x_train,y_train) , (x_test,y_test) = datasets.imdb.load_data(num_words=max_features)
    # x_train : 영화 평 텍스트를 인덱스 화 해서 벡터화한 데이터
    # y_train : 1 or 0 으로 구성
    print((x_train.shape,y_train.shape), (x_test.shape,y_test.shape))
    print(x_train[:1])
    print('-'*50)
    print(y_train[:10])
    # 데이터셋의 문장들의 길이가 모두 다르기 때문에 LSTM 처리를 위해서 
    # 문장의 길이를 통일하는 작업 진행
    # 문장의 길이를 80 규정하고 이보다 작으면 0으로 채우는 과정 진행(전처리)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train,y_train) , (x_test,y_test)

def color_img():
    (x_train,y_train) , (x_test,y_test) = datasets.cifar10.load_data()
    print((x_train.shape,y_train.shape), (x_test.shape,y_test.shape))
    # 이진화 처리

    # 이미지를 2차원으로 변경
    L,W,H,C =x_train.shape

    # 이진화처리
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    x_train = x_train.reshape(-1,W*H*C)
    x_test = x_test.reshape(-1,W*H*C)
    # 정규화
    x_train = x_train/255
    x_test = x_test/255

    return (x_train,y_train) , (x_test,y_test)    

def mnist():
    pass

def boston_house():
    pass