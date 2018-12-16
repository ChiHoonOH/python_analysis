# step2_preprocessing.py

import re
import pandas as pd
from time import time

# 전처리 작업을 위해 호출될 함수


def preprocessor(text):
    # 문자열 내의 html 태그를 삭제한다.
    text = re.sub('<[^>*>]', '', text)
    # 문자열에서 이모티콘을 찾아내는 정규식
    emoticon = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^',text)
    # 문장에서 특수문자를 제거하고
    # 문자열을 소문자로 변환하고
    # 추출한 이모티콘을 붙혀준다.
    text = re.sub('[\W]+',' ', text.lower()+' '.join(emoticon).replace('-', ''))
    # print(emoticon)
    return text
    # 전처리 데이터 저장한다.


def step2_preprocessing():
    # csv 데이터를 읽어 온다.
    df = pd.read_csv('./data/movie_review.csv')
    # 전처리 작업
    stime = time()
    print('전처리 시작')
    df['review'] = df['review'].apply(preprocessor)
    print('전처리 완료')
    print('소요시간:%d' % (time() - stime))

    df.to_csv('./data/refined_movie_review.csv', index=False)