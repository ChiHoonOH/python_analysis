# sum(유사도*평점) / sum(유사도)
from step3_Similarity_pratice import sim_pearson,sim_cosine,sim_msd,top_match

ratings = {
        'Dave': {'달콤한인생': 5, '범죄도시': 3, '샤인': 3, '뽀로로': 2, '머털도사': 1},
        'David': {'달콤한인생': 5, '범죄도시': 1, '샤인': 4, '뽀로로': 1, '머털도사': 4},
        'Alex': {'범죄도시': 4, '샤인': 5, '뽀로로': 5, '머털도사': 2},
        'Andy': {'달콤한인생': 2, '범죄도시': 1, '샤인': 5, '뽀로로': 3, '머털도사': 5}
}


def getRcommandatioon(data, person, k=3, sim_function=sim_pearson):

    sim_list = top_match(data, person, k, sim_function)
    sum_sim = 0
    sum_weighted_sim = 0
    for name, sim in  sim_list:
        sum_sim += sim
        for movie_name in data[name]:
            sum_weighted_sim += data[name][movie_name] * sim

    return sum_weighted_sim / sum_sim


def step4_knnRecommendation():


    pass

