import math

# 사용자간 유사도를 비교할 때 만약 1번 사용자는 평점을 매기고
# 2번 사용자는 평점을 매기지 않았다면, 그 상품에 대한 점수는 계산하지 않는다.(0점으로 계산
# 하면 값이 너무 달라짐, 논리도 맞지 않고)

# 평균 제곱 차이 유사도


def sim_msd(data, name1, name2):
    sum = 0
    count = 0
    # 사용자1에 대한 상품의 수만큼 반복한다.
    for movies in data[name1]:
        if movies in data[name2]:
            sum += pow(data[name1][movies] - data[name2][movies], 2)
            count += 1
    return 1 / (1+(sum/count))  # 유사하지 않을 수록 값이 떨어진다. 유사하면 값이 올라간다.
# 코사인 유사도


def sim_cosine(data, name1, name2):
    sum_name1 = 0
    sum_name2 = 0
    sum_name1_name2 = 0

    for movies in data[name1]:
        if movies in data[name2]:
            sum_name1 += pow(data[name1][movies], 2)
            sum_name2 += pow(data[name2][movies], 2)
            sum_name1_name2 += data[name1][movies] * data[name2][movies]

    return sum_name1_name2 / (math.sqrt(sum_name1) * math.sqrt(sum_name2))
# 이상값에 민감한 척도


# 피어슨 유사도
def sim_pearson(data, name1, name2):
    avg_name1 = 0
    avg_name2 = 0
    count = 0
    for movies in data[name1]:
        if movies in data[name2]:
            avg_name1 += data[name1][movies]
            avg_name2 += data[name2][movies]
            count += 1
    avg_name1 = avg_name1 / count
    avg_name2 = avg_name2 / count

    sum_name1 = 0
    sum_name2 = 0
    sum_name1_name2 = 0
    count = 0

    for movies in data[name1]:
        if movies in data[name2]:
            sum_name1 += pow(data[name1][movies] - avg_name1, 2)
            sum_name2 += pow(data[name2][movies] - avg_name2, 2)
            sum_name1_name2 += (data[name1][movies] - avg_name1) * (data[name2][movies] - avg_name2)
    if (sum_name1 * sum_name2) != 0:
        return sum_name1_name2 / math.sqrt(sum_name1 * sum_name2)
    else:
        return 1

# 사용자간 유사도 계산. - 한사람의 이름을 입력받고 다른 사람들과의 유사도 구해줌.


# 본인을 제외한 나머지 사람들과 유사도를 계산하여, 사람 이름과 값을 저장한다.


def top_match(data, name, index=3, sim_function=sim_pearson):
    li = []
    for i in data:
        if name != i: # 자기자신이 아닐때
            a1 = sim_function(data, name, i)
            li.append((a1, i))
    li.sort()
    li.reverse()
    return li[:index]  # 유사도 상위 index 개의 유사도와 그에 따른 사람이름출력


def step3_Similarity():
    ratings = {
        'Dave': {'달콤한인생': 5, '범죄도시': 3, '샤인': 3, '뽀로로': 2, '머털도사': 1},
        'David': {'달콤한인생': 5, '범죄도시': 1, '샤인': 4, '뽀로로': 1, '머털도사': 4},
        'Alex': {'범죄도시': 4, '샤인': 5, '뽀로로': 5, '머털도사': 2},
        'Andy': {'달콤한인생': 2, '범죄도시': 1, '샤인': 5, '뽀로로': 3, '머털도사': 5}
    }
    a1 = sim_msd(ratings, 'Dave', 'Andy')
    print('평균 제곱 차이 유사도:', a1)
    a2 = sim_cosine(ratings, 'Dave', 'Andy')
    print('코사인 유사도:', a2)
    a3 = sim_pearson(ratings, 'Dave', 'Andy')
    print('피어슨 유사도:', a3)

step3_Similarity()
    # print(top_match(ratings, 'Dave', sim_function=sim_msd))
    # print(top_match(ratings, 'Dave', sim_function=sim_cosine))
    # print(top_match(ratings, 'Dave'))
