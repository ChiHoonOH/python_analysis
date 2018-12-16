import operator  # ?


def step1_Popularity():
    # 데이터
    ratings = {# 사용자 기반
        'Dave': {
            '달콤한인생': 5,
            '범죄도시': 3,
            '샤인': 3,
            '뽀로로': 2,
            '머털도사' : 1
        },
        'David': {'달콤한인생':5, '범죄도시': 1, '샤인': 4, '뽀로로': 1, '머털도사': 4},
        'Alex': {'달콤한인생': 0, '범죄도시': 4, '샤인': 5, '뽀로로': 5, '머털도사': 2},
        'Andy': {'달콤한인생': 2, '범죄도시': 1, '샤인': 5, '뽀로로': 3, '머털도사': 5}
    }

    # 영화 정보를 담을 딕셔너리를 만든다.

    movie_dic = dict()  # 영화기반


    # 사용자수의 수만큼 반복한다.
    for key1 in ratings:  # key1 사용자이름
        for key2 in ratings[key1]:  # key2 영화이름
            if key2 not in movie_dic:  # 이거 뭔 뜻이냐?
                movie_dic[key2] = ratings[key1][key2]
            else:
                movie_dic[key2] += ratings[key1][key2]

    print(movie_dic)
    # 누적된 평점 총합의 평균을 구한다.
    keys = movie_dic.keys()

    for key in keys:
        movie_dic[key] = movie_dic[key] / 4
    print(movie_dic)
    # 정렬한다.
    sorted_x = sorted(movie_dic.items(), key=operator.itemgetter(1), reverse=True)
    #1번째(0,1,...) 값을 기준으로 정렬
    print(sorted_x[:3])

