# step5_using.py


import pickle
import numpy as np


def step5_using():
    with open('./data/pipe.dat','rb') as fp:
        pipeline = pickle.load(fp)
    while True:
        text = input('영문으로 리뷰를 작성해주세요.:')
        y = pipeline.predict([text])
        rates = pipeline.predict_proba([text]) * 100
        rate = np.max(rates)
        if y ==1:
            print('긍정적인 의견')
        else:
            print('부정적인 의견')
        print('정확도: %d' % (rate))