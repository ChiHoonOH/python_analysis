from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
from time import time
import pandas as pd
from konlpy.tag import *
from word_tokenizer import tokenizer_morphs as tokenizer


def step3_learning():
    # 데이터를 읽어온다.
    train_df = pd.read_csv('./movie_train_data.csv')
    test_df = pd.read_csv('./movie_test_data.csv')
    X_train = train_df['text'].tolist()
    y_train = train_df['star'].tolist()

    X_test = test_df['text'].tolist()
    y_test = test_df['star'].tolist()

    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)
    logistic = LogisticRegression(C=10.0, penalty='l2', random_state=0)
    pipe = Pipeline([('vect',tfidf),('clf',logistic)])

    # 학습한다.
    stime = time()
    print('학습시작')
    pipe.fit(X_train, y_train)
    print('학습 종료')
    etime = time()
    print('총 학습시간 : %s ' % (etime-stime))

    # 예측
    y_pred = pipe.predict(X_test)

    # 학습 정확도 측정
    print('정확도 : %s' % accuracy_score(y_test, y_pred))

    with open('./pipe.dat', 'wb') as fp:
        pickle.dump(pipe, fp)

    print('저장완료')