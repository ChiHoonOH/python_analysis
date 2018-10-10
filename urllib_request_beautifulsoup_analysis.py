
# coding: utf-8

# In[1]:


from urllib.request import urlopen
from bs4 import BeautifulSoup


# ### 네이버 금융의 환율 정보 웹스크레핑
# - https://finance.naver.com/marketindex/

# In[10]:


url = 'https://finance.naver.com'
# 염두해 둘 부분은 ajax가 사용되는지, iframe tag가 사용된 것인지 
# 데이터 형태가 html ,xml  등 markup language를 쉽게 파싱하고 데이터를 추출하는 모듈 
# => beautifu soup
pageInfo = urlopen( url + '/marketindex/')
# BeautifulSoup 객체를 만든다 =>  DOM tree를 메모리에 로드한다 <= 행위 : 파싱
soup = BeautifulSoup( pageInfo,  'html.parser')


# In[11]:


# 원본 확인 및 보기 좋게 데코레이션 
#print( soup.prettify() )


# In[13]:


# 환율 정보가 있는 iframe를 찾는다 (관찰및 분석을 통해서 확인했음)
# 사이트 원 주소 : https://finance.naver.com/marketindex/
# 환율 정보 주소는 : https://finance.naver.com/marketindex/exchangeList.nhn
exchangeUrl = soup.select_one('#frame_ex1')['src']
# 환율 정보 풀 주소
exchangeUrl = url + exchangeUrl
print( exchangeUrl )
# iframe는 최초 soup에서 접근이 않됨 -> 재접속 필요


# In[53]:


# 환율 정보 제공 업체, 날짜, 회차 정보 수집
div = soup.find('div', class_='exchange_info')
print(div)
# 방법 2
# 텍스트를 둘러싸고 있는 span를 찾는다
exchange_metaInfo = []
for span in div.find_all('span'):
    #print( span.get_text() )
    if span['class'][0] == 'standard':
        print('->',span.get_text() )
    # 고시회차
    if span.find('em'):
        exchange_metaInfo.append( span.find('em').get_text() )
        #print( span.find('em').get_text() )
    else:
        # 날짜, 공급처
        #print( span.get_text() )
        exchange_metaInfo.append( span.get_text() )

exchange_metaInfo
#print( exchange_metaInfo[1].replace(' 기준', '') )
#print( exchange_metaInfo[1][:len(' 기준')*-1] )
# 데이터 보정
exchange_metaInfo[1] = exchange_metaInfo[1][:len(' 기준')*-1]
# 디비에서 환율을 구분할 코드 데이터 추가
exchange_metaInfo.append('%s_%s' % (exchange_metaInfo[0].replace('.','')[:8], 
                                    exchange_metaInfo[2]) )
print( exchange_metaInfo )

#print( '%s_%s' % (exchange_metaInfo[0].replace('.','')[:8], exchange_metaInfo[2]) )
#print( exchange_metaInfo[0].replace('.','').split()[0] )
# ['2018.10.10 10:31', 'KEB하나은행 기준', '095']
# '20181010_095' 를 동적으로 생성하여서 
# exchange_metaInfo에 추가하시오
# ['2018.10.10 10:31', 'KEB하나은행', '095', '20181010_095']


# 방법 1
# 텍스트를 다 뽑아서 이리 저리 전처리해서 데이터를 뽑는 방법
# '\n2018.10.10 10:31\nKEB하나은행 기준\n고시회차 095회\n'
#div.text[1:-1].split('\n')
# ['2018.10.10 10:31', 'KEB하나은행 기준', '고시회차 095회']


# In[45]:


# 환율 정보 수집을 위한 재접속
exchangeUrl


# In[46]:


res = urlopen( exchangeUrl )
if_soup = BeautifulSoup( res, 'html5lib' )
print( if_soup )


# In[69]:


# tbody밑에 있는 tr들 찾기
info = list()
for tr in if_soup.select('tbody tr'):
    # 모든 텍스트는 td안에 있다
    for td in tr.findAll('td'):
        #print( '[%s]' % td.getText().strip() )
        info.append( td.getText().strip() )
print( len(info), len(info)/7 )


# In[68]:


# numpy 구성
import pandas as pd
import numpy as np


# In[72]:


# arr 이라는 배열명으로 44*7 배열(ndarray) 생성
arr = np.array( info ).reshape( int(len(info)/7), 7)
print( arr.shape, arr.ndim, arr.dtype, arr )


# In[73]:


df = pd.DataFrame( arr )


# In[75]:


df.head(2)


# In[76]:


# 컬럼 준비
col = ['통화명','매매기준율','사실 때','파실 때','보내실 때','받으실 때','미화환산율']
df.columns = col
df.head(2)


# In[77]:


# 통화명 -> 인덱스 변경 
df.set_index('통화명', inplace=True)
df.head(2)


# In[80]:


# 컬럼을 추가 -> 회차란 이름으로 추가 , exchange_metaInfo[3]를 값으로 세팅
df['회차'] = exchange_metaInfo[3]
df


# In[83]:


# CSV 저장, DB 저장
# pip install pymysql
import pymysql as sql
from sqlalchemy import create_engine
import pandas.io.sql as pSql


# In[88]:


df_tmp = df.copy()
df_tmp.index.name = 'cur'
df_tmp.columns = ['rate','buy','sel','send','recv','us_rate', 'code']
df_tmp.head(2)


# In[89]:


# 연결
engine = create_engine('mysql+pymysql://root:12341234@localhost:3306/pythondb'
                       , encoding='utf8')
conn   = engine.connect()
# 쓰기
# 디비에 넣는 df의 컬럼명을 테이블의 컬럼명과 동일하게 맞춰야 들어간다
# 사본작업 해서 테스트
df_tmp.to_sql( name='tbl_exchange',
           con=conn,
           if_exists='append' )
# 닫기
conn.close()


# In[96]:


# code
exchange_metaInfo
# df 생성
meta_df = pd.DataFrame( exchange_metaInfo )
# 축변경
meta_df = meta_df.T
# 컬럼 조정
meta_df.columns = ['date','standard','round','code']
meta_df


# In[97]:


# code 입력
# 연결
engine = create_engine('mysql+pymysql://root:12341234@localhost:3306/pythondb'
                       , encoding='utf8')
conn   = engine.connect()
# 입력
meta_df.to_sql( name='tbl_exchCode',
           con=conn,
           if_exists='append',
           # index 부분은 추가하지않는다
           index=False
            )  
# 닫기
conn.close()

