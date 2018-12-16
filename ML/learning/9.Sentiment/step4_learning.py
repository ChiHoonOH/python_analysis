# step4_learning.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from time import time
import pandas as pd
from step3_word_tokenizer import tokenizer, tokenizer_porter

pipeline = None


def step4_learning():
    # csv 파일에서 데이터를 읽어온다.
    df = pd.read_csv('./data/refined_movie_review.csv')
    # 테스트, 학습데이터로 나눈다.
    X_train = df.loc[:35000 - 1, 'review'].values
    y_train = df.loc[:35000 - 1, 'sentiment'].values

    X_test = df.loc[35000:50000, 'review'].values
    y_test = df.loc[35000:50000, 'sentiment'].values

    print(len(X_train), len(X_test))

    # 단어장을 만들어 주는 객체 생성
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)  # 단어는
    # tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_porter)  # 단어는
    # 데이터 학습 객체
    logistic = LogisticRegression(C=10.0, penalty='l2', random_state=0)

    # 파이프라인 설정
    pipeline = Pipeline([('vect',tfidf),('clf',logistic)])

    # 학습 한다.
    stime = time()
    print('러닝 시작')
    pipeline.fit(X_train,y_train)
    print('학습 종료')
    print('총 학습시간 : %d' % (time() - stime))

    # 테스트
    y_pred = pipeline.predict(X_test)
    print('정확도 %.3f' % accuracy_score(y_test, y_pred))

    with open('./data/pipe2.dat', 'wb') as fp:
        pickle.dump(pipeline, fp)
    print('저장완료')