# step1_GetData.py

import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd


def step1_Getdata():
    # 영화 코드
    soup = BeautifulSoup()
    code_list = [167638, 109906]
    chk = False # 첫번째 영화님이신가 아닌가 .
    for code in code_list:
        site1 = '''https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=%s&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false''' % code
        res1 = requests.get(site1)
        print('res1.status_code:',res1.status_code)
        if res1.status_code == requests.codes.ok:
            print(1)
            bs1 = BeautifulSoup(res1.text, 'html.parser')
            print(2)
            #tag = bs1.select('body > div > div > div.score_total > strong > em')

            score_total = bs1.find(class_='score_total')
            ems = score_total.find_all('em')
            score_total = int(ems[1].text.replace(',',''))


            pageCnt = score_total // 10
            # if score_total % 10 > 0:
            #     pass

            print(pageCnt)
            # 현재 페이지 번호
            now_page = 1
            pageCnt = 5
            # print(tag)

            while now_page <= pageCnt:
                sleep(0.5)
                # 요청할 페이지 주소
                site2 = '''https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=%s&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=%d''' % (code, now_page)
                res2 = requests.get(site2)
                df = pd.DataFrame()
                if res2.status_code == requests.codes.ok:
                    bs2 = BeautifulSoup(res2.text, 'html.parser')
                    score_result = bs2.find(class_='score_result')
                    lis = score_result.find_all('li')


                    # 굳이 이렇게 한 이유는 영화가 엄청늘어나면 메모리부족할 수 있어서
                    # 미리미리 저장한다.

                    for obj in lis:
                        # 평점
                        star_score = obj.find(class_='star_score')

                        star_em = star_score.find('em')
                        star_score = int(star_em.text)# 평점

                        score_reple = obj.find(class_='score_reple')
                        reple_p = score_reple.find('p')
                        score_reple = reple_p.text
                        print(star_score, score_reple)




                        # 데이터를 누적한다.
                        df = df.append([[score_reple, star_score]],ignore_index=True)

                    now_page += 1

                #저장
                if chk == False:
                    df.columns = ['text','star']
                    df.to_csv('naver_star_data.csv', index=False, encoding='euc-kr')
                    ### sig로 저장안하면 다깨짐 ㅇㅇ
                else:

                    df.to_csv('naver_star_data.csv', index=False, encoding='euc-kr', mode='a', header=False)
                chk = True







        # 1단계 : 해당 영화의 평점 페이지 수 계산.
        # 2단계 : 평점 글 정보와 정보를 가져온다.



