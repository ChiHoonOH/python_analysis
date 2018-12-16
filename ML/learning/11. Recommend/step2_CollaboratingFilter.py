import math

# 피타고라스 공식을 이용해 거리를 계산한다.


def sim(i, j):
    return math.sqrt(pow(i, 2)+pow(j, 2))

# 사용자에 대한 유사도 계산


def use_sim(user_name, ratings):
    print(user_name, '-'*40)

    for name in ratings:
        if name != user_name:
            a1 = ratings[user_name]['범죄도시']  # 알렉드의 범죄도시와 나머지 사람의 범죄도시와 거리계산
            a2 = ratings[name]['범죄도시']
            num1 = a1 - a2

            a3 = ratings[user_name]['샤인']
            a4 = ratings[name]['샤인']

            num2 = a3 - a4

            print(name, ":", 1 / (1 + sim(num1, num2)))  # 크면 클수록 유사하다는 말(그래서 역수)
            # 1을 더해준 이유는 두점사이의 거리가 0이 되면 inf 문제가 발생할 수 있어서


def step2_CollaboratingFilter():
    ratings = {  # 사용자 기반
        'Dave': {'달콤한인생': 5, '범죄도시': 3, '샤인': 3, '뽀로로': 2,'머털도사': 1},
        'David': {'달콤한인생': 5, '범죄도시': 1, '샤인': 4, '뽀로로': 1, '머털도사': 4},
        'Alex': {'범죄도시': 4, '샤인': 5, '뽀로로': 5, '머털도사': 2},
        'Andy': {'달콤한인생': 2, '범죄도시': 1, '샤인': 5, '뽀로로': 3, '머털도사': 5}
    }

    # 거리계산
    # 점수 차를 계산한다.
    a1 = ratings['Alex']['범죄도시']
    a2 = ratings['Andy']['범죄도시']

    result1 = a1 - a2

    a3 = ratings['Alex']['샤인']
    a4 = ratings['Andy']['샤인']

    result2 = a3 - a4

    # 유사도 계산
    result3 = sim(result1, result2)
    print(result3)

    # 다차원일 경우는 행렬일 것이다.
    # 행렬로 거리 연산을 어떻게하는지 고민해 볼 필요가 있다.
    # 정규화를 통해 유사도 범위를 0과 1사이에 두고 1에 가까울 수록 유사도가 높게 평가
    # print('Alex----------')
    # for name in ratings:
    #     if name != 'Alex':
    #         a1 = ratings['Alex']['범죄도시']#알렉드의 범죄도시와 나머지 사람의 범죄도시와 거리계산
    #         a2 = ratings[name]['범죄도시']
    #         num1 = a1 - a2
    #
    #         a3 = ratings['Alex']['샤인']
    #         a4 = ratings[name]['샤인']
    #
    #         num2 = a3-a4
    #
    #         print(name,":", 1/ (1+sim(num1,num2))) #크면 클수록 유사하다는 말(그래서 역수)
    #         # 1을 더해준 이유는 두점사이의 거리가 0이 되면 inf 문제가 발생할 수 있어서

    for key in ratings:
        use_sim(key,ratings)

    # print(a1)
    # print(a2)
