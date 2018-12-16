import pandas as pd
from konlpy.tag._okt import Okt
okt = Okt()


def tokenizer_morphs(text):
    return okt.morphs(text)


def word_tokenizer():
    train_df = pd.read_csv('./movie_train_data.csv')
    train_df['text_token'] = train_df['text'].apply(tokenizer_morphs)
    print(train_df['text_token'])