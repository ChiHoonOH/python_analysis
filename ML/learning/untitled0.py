# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:04:11 2018

@author: ChiHoon
"""

# 문제는 랜덤한 순서로 출력한다.
# 1분 단위로 변경, 수동으로 실행할 수 있는 프로그램을 짠다..




interview_list=[
'자신을 사물에 비유한다면 무엇에 비유하시겠습니까? 그리고 그 이유는 무엇입니까?',
'자신을 색깔에 비유해보세요. 그리고 그 이유는 무엇입니까?',
'상사가 부당한 지시를 하였습니다. 이에 어떻게 대처 하시겠습니까?',
'상사가 당신의 공을 가로챘습니다. 어떻게 하시겠습니까?',
'자신을 물건에 비유해 보세요. 그리고 그 이유는 무엇입니까?',
'지원 동기가 무엇입니까? ',
'롤 모델은? 또는 존경하는 사람은 누구인가요?',
'입사 후 포부에 대해서 말씀해보세요.',
'최근에 읽은 책? 최근에 본 영화는 무엇입니까?',
'본인을 우리가 뽑아야 하는 이유는 무엇입니까?',
'현재 분야에서 필요한 자질은 무엇이라고 생각합니까?',
'살면서 실패한 경험?',
'희생정신을 발휘해본 경험이 있는가?'
]
#interview_dict = dict(zip(interview_list,[1 for _ in range(len(interview_list))]))

#문제를 크게 출력한다.
#print(interview_dict)
import mylib.custom as cus
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

import random
for element in interview_list:    
    interview_dict = dict([tuple([element,9999999999999]),("1",1)])
    plt.subplots(1,1,figsize=(18,8))
    wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',background_color='white',min_font_size=10).generate_from_frequencies(frequencies=interview_dict)
    
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    time.sleep(3)
    #plt.cla()
    plt.clf()
    
    plt.figure(clear=True)
    

# 글자를 크게 띄우는 방법?
