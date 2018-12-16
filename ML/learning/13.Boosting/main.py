# main.py

### emsemble => 각각 모델링을 돌리고 ,그 모델링을 평균낸다. => 느림
## voting classifier : 예측(?)을 기준으로 투표를 해서, 투표에서 상위권
# 최종 모델들을 뽑아서 평균을 내서 계산한다.

from ex1_VotingClassifier import ex1_VotingClassifier
from ex2_Bagging import ex2_Bagging
from ex3_randomforest import ex3_RandomForest
from ex4_AdaBoosting import ex4_AdaBoosting
from ex5_GBM import ex5_GBM
from ex6_xgboosting import ex6_xgboosting
from ex7_LGBM import ex7_LGBM
# ex1_VotingClassifier()
# ex2_Bagging()
# ex3_RandomForest()
# ex4_AdaBoosting()
# ex5_GBM()
# ex6_xgboosting()
ex7_LGBM()
