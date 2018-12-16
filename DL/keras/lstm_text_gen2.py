#####
# LSTM/RNN 알고리즘을 이용하여 문장을 생성한다.
# LSTM은 장기적으로 직전 데이터뿐만 아니라 과거 데이터도 기억하는 기능이 추가되니 개선 버전
# 내일이라는 단어가 입력되면 이미지는 단어를 날씨?, 약속 이런 방식으로 이어질 것이라
# 예측하고 조합할 수 있다. 내일 날씨라고 입력하면 비가 오고 흐립니다. 이런식으로 
# 문장을 만들 수 있다.
# 케라스 원제공 소스를 가공한  버전 
####

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
####################################################
# 데이터
def loadData():
    sample_file = 'data/욕망이타는숲.txt'
    sample_file = 'data/small.txt'
    with open(sample_file, 'r', encoding='utf8') as fp:
        text = fp.read().replace('\n','')
        print('코퍼스의 길이', len(text))
    # 문자를 하나하나 중복제거해서 리스트화해서 소트처리
    # 불필요한 텍스트 \n 제거 => 정규화
    chars =sorted(list(set(text))) # ?? 근데 이짓거리 왜함?
    # print(chars)
    # 문자가 키, 인덱스가 값 
    char_indices = dict([(c,idx)for idx,c in enumerate(chars)])  # enumerate를 통해서 키를 뽑을 수 있음.    
    # 인덱스가 키, 문자가 값
    indices_char = dict([(idx, c)for idx,c in enumerate(chars)])  # enumerate를 통해서 키를 뽑을 수 있음.    
    # print(char_indices)
    # print(indices_char)
    # 학습 해야할 원본 데이터 벡터화 처리
    # 벡터당 크기세트 20
    # 20 단어 세트로 움직이는 칸수, step 3
    maxlen = 20
    step = 3
    sentence = []
    # 자른 벡터 데이터의 바로 다음 이어지는 단어 1개 담는다.
    next_char = []
    for i in range(0,len(text)-maxlen, step):  # 시작 지점을 range로
        # 텍스트 자르기
        sentence.append(text[i:maxlen+i])
        # 자른 텍스트에 이어지는 단어.
        next_char.append(text[i+maxlen])
    # print('sentence,next_char',sentence,next_char)

    # 차원 정의
    # len(sentence) :maxlen으로 세트된 개수
    # maxlen : 해당 세트의 수(20)
    # len(chars) : 그 세트의 하나한 값을 구성하는 종류의 수 
    # ( len(sentence), maxlen, len(chars)) ## 문장수, 문장길이, 문장을 구성하는 전체 문자 묶음 길이
    # 근데 차원을 왜 이렇게 하나?
    x = np.zeros(( len(sentence), maxlen, len(chars)), dtype=np.bool)
    # 백터화 처리
    for i,sen in enumerate(sentence):
        #print(sen)
        for j, char in enumerate(sen):
            # print(i, j, char)
            x[i, j, char_indices[char]] = 1

    print(x)
            
    # y = np.zeros()

# 모델링

# 문장 생성하기

# 머신
class Machine:
    def __init__(self):
        loadData()

    def play(self):
        pass
# 메인
def main():
    m = Machine()
    m.play()

if __name__ == '__main__':
    main()