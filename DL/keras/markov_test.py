'''
마르코프 체인 방식으로 문장 생성법 구성 
개별 알고리즘 이해(마르코프 or LSTM) => 문장만들기 => 대화에 반응하여 만들기 => 챗봇 => 순환구조 작성
마르코프 체인 방식은 사전 재료가 원하는 주제에 부합되게 맞아야 하고
해당 내용을 사용된 정보들이 계속해서 순환 구조로 사전에 유입 되게 설계해야한다. 
단어 선택 부분에 랜덤이 아닌, 나름대로 선택 로직이 필요하다.
'''

import os
from konlpy.tag import Okt
import json
import random # ?? 단어선택을 위해서라고 하는데 단어선택을 왜함?

def setting_word3Form(dic,tmp):
    print(dic,tmp)
    w1, w2, w3 = tmp
    print(w1,w2,w3)
    # 만들기
    if not w1 in dic: #w1으로 시작하는 키가 존재 하지 않으면
        dic[w1]={} # => {@:{}}
        #dic[w1][w2]={}
    if not w2 in dic[w1]:# @로 시작될 데이터들 중에 키가 유럽인 키가 존재하는가?
        # {'@':{'유럽':()}} 이렇게 생성되었다.
        dic[w1][w2]={}
    if not w3 in dic[w1][w2]: # @로 시작하고 그 다음이 유럽이고 그다음 키가 풍 인 키가 있는가?
        dic[w1][w2][w3]=0
    # 증가됨 
    dic[w1][w2][w3] += 1    
    
    
# 사전 만들기 
def make_dic(words):
    # 3개씩 묶어서 : {이전단어:{현재단어:{다음단어:1}}}
    # 최초문장이 새로 시작하면 @ 키워드를 표식으로 사용.
    dic = {}
    tmp = ['@']
    for word in words:
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:] # 다음 단어부터 3개(유럽, 풍, x)
        #?? 이거 이해안감.
        # 재료 3개가 준비 되었다.
        setting_word3Form(dic,tmp)
        # 문장의 끝을 만나면 새로움 문장의 시작임을 표식처리한다.
        if word == '.':
            tmp = ['@']        
        # print(dic,tmp)
        
    return dic


def word_select(sel):
    # 특정 단어 기준 다음에 나올 단어들 후보 집합
    keys = list(sel.keys())
    print(keys)# @인 경우,문장이 시작하는 단어의 집합.
    return random.choice(keys) # choice(list) list내에서 하나를 짚어줌
    # ?? 그냥 차례대로 단어 하나 짚으면 안되나? 
    # 인공 신경망은 이 선택을 어떻게 할 것인 가에 대한 규칙을 부여할거라 생각함.


# 문장 만들기
def make_sentence(dic):
    ret = []
    # 문자의 시작점 키
    start = dic['@']
    print('문장 만들기 시작...')
    # print(start)
    w1 = word_select(start)  # 첫번째 단어 선택
    w2 = word_select(start[w1])  # 두번째 단어 선택
    # print('w1',w1)
    # print('w2',w2)    
    ret.append(w1)
    ret.append(w2)
    while True:
        w3 = word_select(dic[w1][w2])
        ret.append(w3)
        if w3 == '.': break
        w1, w2 = w2, w3

    return ' '.join(ret) #=> 맞춤법 검사기 사용하여 띄어쓰기 처리

# 메인 시작점
def main():
    sample_file = 'data/욕망이타는숲.txt'
    sample_file = 'data/small.txt' #소량
    # 사전화된 데이터를 저장 -> 다음번 구동부터는 사전을 불러서 사용할 수 있게
    
    dict_file = 'data/markov_dic.json'# 저장 파일인가 이거?
    if not os.path.exists(dict_file):
    #if os.path.exists(dict_file):
        # 사전을 만들어라
        with open(sample_file, 'r', encoding='utf8') as fp:
            text = fp.read()
            # print(text)
            # 형태소 분석
            okt     = Okt()
            txtlist = okt.pos(text, norm=True)
            print(len(txtlist), txtlist)
            words = []
            # ('...','Pun') 이런 부분에 대한 보정처리
            for word in txtlist:# ??                
                if not word[1] in ['Punctuation']:
                    words.append( word[0] )
                # if not word[1] in ['Punctuation']:
                #     words.append(word[0])
                if word[0] =='.':
                    words.append(word[0])
            # print(words)
            # 딕셔너리 생성
            dic = make_dic(words)
            # 덤프(사전저장)
            json.dump(dic, open(dict_file,'w', encoding='utf-8'))
    else:
        # 사전이 존재한다.
        dic = json.load(open(dict_file,'r'))
    
    # 데이터 읽어오기
    # 사전 만들기
    # 문장 만들기 
    for i in range(3):
        print(make_sentence(dic))
        print('-'*50)
    pass

if __name__ == '__main__':
    main()