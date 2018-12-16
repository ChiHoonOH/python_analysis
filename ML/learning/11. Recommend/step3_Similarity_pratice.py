import numpy as np


def sim_msd(data, name1, name2):
    # data는 {사람이름:{영화이름:평점:}}과 같은 형식읻.
    # mean square difference
    # name1과 name2에서 공통적으로 평점을 매긴 영화끼리 거리를 구한다.
    name1_score = []
    name2_score = []
    for movie_name in data[name1]:
        if movie_name in data[name2]:
            name1_score.append(data[name1][movie_name])
            name2_score.append(data[name2][movie_name])
    name1_score_array = np.array(name1_score)
    name2_score_array = np.array(name2_score)

    return 1/(1+np.mean(np.square(name1_score_array-name2_score_array)))


def sim_cosine(data, name1, name2):
    # cosine은 내적/norm(name1)  * norm(name2)
    # 내적은 아마 평점의 벡터라고 추측
    name1_score = []
    name2_score = []
    for movie_name in data[name1]:
        if movie_name in data[name2]:
            name1_score.append(data[name1][movie_name])
            name2_score.append(data[name2][movie_name])
    name1_score_array = np.array(name1_score)
    name2_score_array = np.array(name2_score)

    return np.dot(name1_score_array, name2_score_array) / \
           (np.linalg.norm(name1_score_array) * np.linalg.norm(name2_score_array))


def sim_pearson(data, name1, name2):
    # pearson 상관 계수 cov(x,y)/(std(x)*std(y))
    name1_score = []
    name2_score = []
    for movie_name in data[name1]:
        if movie_name in data[name2]:
            name1_score.append(data[name1][movie_name])
            name2_score.append(data[name2][movie_name])
    name1_score_array = np.array(name1_score)
    name2_score_array = np.array(name2_score)
    # print(name1_score_array)
    # print(name2_score_array)
    # print('np.cov:', np.cov(name1_score_array, name2_score_array)) - 이거 이상하다 생각함.
    # 공분산의 정의
    # sum((x-x_)(y-y_))/ degree of freedom

    cov_ = np.dot((name1_score_array - np.mean(name1_score_array)),(name2_score_array - np.mean(name2_score_array)))/\
           len(name1_score_array)
    return cov_ / (np.std(name1_score_array) * np.std(name2_score_array))


def top_match(data, name1, limit=3,  sim_function=sim_pearson):
    li = []
    for name in data:
        if name1 != name:
            sim_value = sim_function(data,name1, name)
            li.append((name, sim_value))
    li.sort()
    li.reverse()
    return li[:limit]


ratings = {
        'Dave': {'달콤한인생': 5, '범죄도시': 3, '샤인': 3, '뽀로로': 2, '머털도사': 1},
        'David': {'달콤한인생': 5, '범죄도시': 1, '샤인': 4, '뽀로로': 1, '머털도사': 4},
        'Alex': {'범죄도시': 4, '샤인': 5, '뽀로로': 5, '머털도사': 2},
        'Andy': {'달콤한인생': 2, '범죄도시': 1, '샤인': 5, '뽀로로': 3, '머털도사': 5}
}

print(sim_msd(ratings,'Dave','Andy'))
print(sim_cosine(ratings,'Dave','Andy'))
print(sim_pearson(ratings,'Dave','Andy'))
print(top_match(ratings,'Dave'))