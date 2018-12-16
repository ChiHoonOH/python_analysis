# step1_get_data.py

import pandas as pd
import os
import numpy as np
import codecs
# 원하는 데이터 저장 형태 

#  0 별점 내용 긍정/부정


def step1_get_data():
    # 데이터 파일들이 들어있는 경로
    path = '../aclImdb_v1/aclImdb/'
    # 긍정 또는 부정을 의미하는 값
    labels = {'pos':1, 'neg': 0}
    # csv에 저장할 값을 관리할 객체
    df = pd.DataFrame()

    # 디렉토리의 개수만큼 반복한다.
    for s in ('test','train'):
        for name in ('pos', 'neg'):
            # 읽어올 파일들이 들어 있는 디렉토리명을 만든다.
            subpath = '%s/%s' %(s, name)
            dirpath = path + subpath
            # print(dirpath)

            # 현재 디렉토리 안에 있는 파일 목록
            file_list = os.listdir(dirpath)
            # print(file_list)
            # 파일 목록을 순회하면서 목록ㅇ르 가져온다.
            for file in file_list:
                # 내 생각엔 file / 내용 으로 나누어서 저장하면 좋을듯.
                fileName = os.path.join(dirpath, file)
                txt = None
                file_split = file.split('_')
                star = file_split[1].split('.')[0]
                with codecs.open(fileName, 'r', encoding='utf-8') as fp:
                    txt = fp.read()
                # print(file_split[0], star, labels[name], txt)

                df = df.append([[txt, labels[name]]], ignore_index=True)
                print(fileName)

    # 컬럼 설정
    print(df)
    df.columns = ['review', 'sentiment']
    # 순서를  섞는다.
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    # 저장한다.
    df.to_csv('./data/movie_review.csv', index=False)
