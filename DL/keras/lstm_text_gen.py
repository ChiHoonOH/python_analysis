##############################
# LSTM/RNN 인공지능을 이용하여 문자을 생성한다.
# LSTM은 장기적으로 직전 데이터 뿐만 아니라 과거 데이터도 기억하는 기능이 추가도니 개선 버전
# 내일이라는 단어가 입려되면 이어지는 단어를 날씨?, 약속 이런식으로 이어질거라고 예측하고
# 조합할수있는다.  내일 날씨라고 입력하면 비가 오고 흐립니다. 이런식으로 문장을 만들수 잇다
# 케라스 원제공 소스를 가공한 버전
##############################
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
###############################

# 데이터
def loadData(maxlen=20, step=3):
  sample_file = 'data/욕망이타는숲.txt' # 나중에 사용
  sample_file = 'data/small.txt'       # 임시사용
  with open(sample_file, 'r', encoding='utf8') as fp:
    text = fp.read().replace('\n','')
    print( '코퍼스의 길이', len(text) )
  # 문자를 하나하나 중복 제거해서 리스트화해서 소트 처리 
  # 불필요한 텍스트 \n 제거 => 정규화
  chars = sorted(list(set( text )))
  print( len(chars) )
  # 문자가 키, 인덱스가 값
  char_indices = dict( (c, i) for i, c  in enumerate(chars) )
  # 인덱스카 키, 문자가 값
  indices_char = dict( (i, c) for i, c  in enumerate(chars) )
  #print( char_indices )
  #print( indices_char )
  # 학습해야 할 원본 데이터의 백터화 처리
  # 백터당 크기 세트 20
  #maxlen = maxlen 
  # 20단어 세트로 움직이는 칸수, step 3
  #step   = step 
  sentences = []
  # 자른 백터 데이터의 바로 다음 이어지는 단어 1개 담는다
  next_char = []
  for i in range( 0, len(text)-maxlen, step ):
    # 벡터당 텍스트 자르기 
    sentences.append( text[i:i+maxlen] )
    # 자른 텍스트에 이어지는 단어
    next_char.append( text[i+maxlen] )
  #print( sentences )
  #print( next_char )

  # sentences 백터화 작업 
  # 차원 정의
  # len(sentences) : maxlen로 세트된 개수
  # maxlen : 해당 세트의 수(20)
  # len(chars) : 그 세트의 하나한 값을 구성하는 종류의 수 ( 가, 나, 다 )이것만 있었다면 3
  # ( len(sentences),  maxlen,  len(chars) )
  x = np.zeros( (len(sentences),  maxlen,  len(chars)), dtype=np.bool  )
  # 문장에 대한 다음문자 백터화 
  # (문장수, 다음문자)
  y = np.zeros( (len(sentences),  len(chars)), dtype=np.bool )
  # 백터화 처리
  for i, sentence in enumerate( sentences ):
    #print( sentence )
    for j, char in enumerate(sentence):
      #print( i, j, char )
      x[i, j, char_indices[char]] = 1
    # 백터화 처리
    y[ i, char_indices[next_char[i]] ] = 1
  print( x )
  return x, y, text, char_indices, indices_char
  
# 모델링
class RNN_ISTM(Sequential):
  def __init__(self, maxlen, chars_num):    
    super().__init__()
    # 입력층 & 1차 은닉층
    self.add( LSTM(128, input_shape=(maxlen,chars_num)  )  )
    # 2차 은닉층
    self.add( Dense(chars_num) )
    # 출력층
    self.add( Activation('softmax') )
    # 최적화 함수
    # 매우 효과적으로 학습및 데이터 맞춤속도를 최적으로 조정하는 최적화 함수
    optimizer = RMSprop(lr=0.01)
    # 컴파일
    self.compile( loss='categorical_crossentropy', optimizer=optimizer )


# 문장 생성하기
# 머신
class Machine:
  def __init__(self):
    # 매개변수
    self.SENTENCE_MAXLEN = 20
    self.STEP_LEN        = 3
    # 데이터 로드
    x, y, text, char_indices, indices_char  = loadData(self.SENTENCE_MAXLEN, self.STEP_LEN)
    # 모델링 생성
    print( type(x.shape[1]), type(x.shape[2]) )
    self.model = RNN_ISTM( x.shape[1], x.shape[2] )
    self.x = x
    self.y = y
    self.text = text
    # 인코더 : 문자를 넣으면 인덱스값이 나온다
    self.char_indices = char_indices
    # 디코더 : 인덱스값을 넣으면 문자가 나온다
    self.indices_char = indices_char

  # 훈련 및 문장 만들기 (통상 훈련후 만들면 되는데-> 과정을 보기 위해 임의 처리)
  def makeSenetence(self):
    for iter in range(100):# 임의로 훈련수를 넣은것으로, 통상은 에포크로 세트 구성
      # 훈련 
      # 문장이 있고 그다음에 나오는 문자가 y라는 답안을 훈련
      # nb_epoch : 학습여부
      self.model.fit(self.x, self.y, batch_size=128, nb_epoch=1)
      # 훈련용, 테스트용, 검증용 분류 자체를 하지 않았다
      # 문장만들기
      # 임의의 한문장의 시작점 획득
      start_index = random.randint(0, len(self.text) - self.SENTENCE_MAXLEN -1 )
      # 임의값 세트로 생성
      for d in [0.2, 0.5, 1.0, 1.2]:
        print('-- 조정값 : ', d)
        generated = ' '
        # 문장 하나 획득( 재료중에 어디든 가서 20개 획득 )
        sentence  = self.text[ start_index : start_index + self.SENTENCE_MAXLEN ]
        generated += sentence
        # 연속 출력
        sys.stdout.write( generated )
        # 이이서 다음 문자를 예측하여 텍스트르 자동 생성
        for i in range(400):# 임의로 400자 한정
          # 다음 문자에 대항되는 백터 대비 지금 준비된 데이터의 백터의 shape는?
          # 텍스트를 => 백터화룰 해야 예측이던 뭔가 작업 가능
          # ( 문장이 1개, 20(문장의문자개수), 들어올수있는문자수(클레스수) )
          px = np.zeros( (1, self.SENTENCE_MAXLEN, self.x.shape[2]) )
          # 시드 문장의 이진화 처리
          for t, char in enumerate(sentence):
            px[ 0, t, self.char_indices[char] ] = 1
          # 예측 => 주어진 문장에 대해 다음에 나올수 있는 문자에 대한 예측값이 백터로 나온다
          preds = self.model.predict( px, verbose=0)
          # 케라스 LSTM의 샘플링으로 제공하는 수식을 활용하여 인덱스 획득
          next_char_index = self.sampling(preds[0], d)
          # 인덱스-> 디코더 -> 문자
          next_char = self.indices_char[ next_char_index ]
          # 문자를 출력하면서 다음 문자로 이어진다 다음 20글자 준비
          generated += next_char
          # 최초 20개의 문자를 들고 있다가 위에 코드 수행되서 21개 문자를 들고 있게 되었다
          # 다음 문자를 예측하기 위해서 1번 인덱스터 자르면 다시 20개가 된다 
          # 20개라는 수치를 맞췄고, 새로 등장한 문자가 붙으면서 새로운 텍스트가(문장)이 되었다
          sentence = sentence[1:] + next_char
          # 콘솔 출력
          sys.stdout.write( next_char )
          sys.stdout.flush()


  # 다음 문자등 이어지는 문자을 가기 위해 찾는 로직(알고리즘)
  # 케라스에서 제공하는 수식 사용
  # 모델에서 나온 원본 확률 분포의 가중치를 조정하여 새로운 글자의 인덱스를 추출
  # d: 반복의 수치 조정
  # d값이 낮으면 반복적으로 예상이 되는 질문에 맞춰서 처리
  # d값이 높으면 새로운 값이 나오지만 구조가 무너지는 문제점
  def sampling(self, preds, d):
    # 예측 데이터의 타입 변경
    preds = np.asarray(preds).astype('float64')
    # 로그처리및 d값의 정규화
    prdes = np.log(preds) / d
    # 지수 함수 처리
    exp_preds = np.exp(preds)
    # 지수합으로 정규화 
    preds = exp_preds / np.sum(exp_preds)
    # 다항분포에서 샘플링
    pro = np.random.multinomial(1, preds, 1)
    # 최종 문자의 인덱스
    return np.argmax(pro)

  def play(self):
    # 훈련 -> 문장생성 -> 반복
    self.makeSenetence()

# 메인
def main():
  m = Machine()
  m.play()
# 실행
if __name__ == '__main__':
  main()
