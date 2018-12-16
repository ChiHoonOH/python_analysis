# user item rate 중복되지않는값
# '숫자' '숫자' 숫자 '숫자'로 해야한다.

# 보통 회원에 index 번호가 있기 때문에 user에 대한 걱정을 할 필욘 없다.
import surprise
import pandas as pd

ratings_expand = {
    '마동석': {
        '택시운전사': 3.5,
        '남한산성': 1.5,
        '킹스맨:골든서클': 3.0,
        '범죄도시': 3.5,
        '아이 캔 스피크': 2.5,
        '꾼': 3.0,
    },
    '이정재': {
        '택시운전사': 5.0,
        '남한산성': 4.5,
        '킹스맨:골든서클': 0.5,
        '범죄도시': 1.5,
        '아이 캔 스피크': 4.5,
        '꾼': 5.0,
    },
    '윤계상': {
        '택시운전사': 3.0,
        '남한산성': 2.5,
        '킹스맨:골든서클': 1.5,
        '범죄도시': 3.0,
        '꾼': 3.0,
        '아이 캔 스피크': 3.5,
    },
    '설경구': {
        '택시운전사': 2.5,
        '남한산성': 3.0,
        '범죄도시': 4.5,
        '꾼': 4.0,
    },
    '최홍만': {
        '남한산성': 4.5,
        '킹스맨:골든서클': 3.0,
        '꾼': 4.5,
        '범죄도시': 3.0,
        '아이 캔 스피크': 2.5,
    },
    '홍수환': {
        '택시운전사': 3.0,
        '남한산성': 4.0,
        '킹스맨:골든서클': 1.0,
        '범죄도시': 3.0,
        '꾼': 3.5,
        '아이 캔 스피크': 2.0,
    },
    '나원탁': {
        '택시운전사': 3.0,
        '남한산성': 4.0,
        '꾼': 3.0,
        '범죄도시': 5.0,
        '아이 캔 스피크': 3.5,
    },
    '소이현': {
        '남한산성': 4.5,
        '아이 캔 스피크': 1.0,
        '범죄도시': 4.0
    }
}


def data_to_dic():
    # 사용자 목록을 담을리스트
    name_list = []
    # 영화 목록을 담는 그릇
    movie_list = set()
    for user_key in ratings_expand:
        # 사용자의 이름을 담는다.
        name_list.append(user_key)
        # 현재 사용자가 본 영화 이름을 담는다.=> 왜 하나씩 담아줌?- 전체영화목록을 담기 위해서
        # ?? 전체 영화 목록은 어따 쓸라고? => 각 목록에서 index로 찾음으로써 결과를 숫자로
        # 바꿔주기 위해서.
        for movie_key in ratings_expand[user_key]:
            movie_list.add(movie_key)
    movie_list2 = list(movie_list)

    # print(name_list)
    # print(movie_list2)
    # 학습을 위해 사용할 데이터를 담을 딕셔너리
    rating_dic={
        'user_id':[], # 여기서는 굳이 이름 안맞춰줘도됨. 나중에 지정해주면됨
        'item_id':[],
        'rating':[]
    }

    for name_key in ratings_expand:
        for movie_key in ratings_expand[name_key]:
            a1 = name_list.index(name_key) # index 함수는 값이 몇번째에 있는지 알려줌.
            a2 = movie_list2.index(movie_key)
            a3 = ratings_expand[name_key][movie_key]

            rating_dic['user_id'].append(a1)
            rating_dic['item_id'].append(a2)
            rating_dic['rating'].append(a3)
    return name_list, movie_list2, rating_dic


def step5_surprise():
    name_list, movie_list, rating_dic = data_to_dic()
    print(rating_dic)
    # 데이터 셋을 만든다.
    df = pd.DataFrame(rating_dic)
    # rating_scale : 데이터에 담긴 평점이 범위
    reader = surprise.Reader(rating_scale=(0.0, 5.0)) # 0~5점의 범위를 가지고 있다.
    # print(rating_dic.keys())
    col_list = [key for key in rating_dic.keys()]
    data = surprise.Dataset.load_from_df(df[col_list],reader)
    print(data)

    trainset = data.build_full_trainset()
    option1 = {'name':'pearson'}
    algo = surprise.KNNBasic(sim_options=option1)
    algo.fit(trainset)
    
    # 소이현에 대해 영화를 추천 받는다.
    index = name_list.index('소이현')
    result = algo.get_neighbors(index, k=3) #iid 자리 -> 대상 인간

    for r1 in result:
        print(movie_list[r1-1]) # r1이 1번부터 시작하나보네
    # 데이터가 너무 없기 때문에 실행할때마다 쉽게 값이 변한다.
    # 성능 측정은 없네
    
    

def step5_surprise2():
    data = surprise.Dataset.load_builtin('ml-100k')

    # user item rate



    # print(data)
    # print(data.raw_ratings)

    # 모든 컬럼에 있는 데이터를 가져와야함.
    # df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rate', 'id'])
    option1 = {'name': 'msd'}
    option2 = {'name': 'cosine'}
    option3 = {'name': 'pearson'}

    # 추천 목록을 만들기 위한 객체 생성
    # 본질을 흐리지 말것. 애매하게 하는게 제일 나쁘다.
    # 학습용 데이터를 생성한다.
    trainset = data.build_full_trainset()
    algo = surprise.KNNBasic(sim_options=option3)
    print('학습시작')
    algo.fit(trainset)

    # 추천 목록을 가져 온다.
    result = algo.get_neighbors(196, k=3)  # 추천 영화 목록
    print('result type:',type(result))
    for r1 in result:
        print(r1)




