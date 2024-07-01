from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def models():
    models = {
              "k-NN": KNeighborsClassifier(),
              "Decision Tree": DecisionTreeClassifier(random_state=42),
              "BernoulliNB": BernoulliNB(),
              "LogisticRegression": LogisticRegression(random_state=42),
              "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
              "XGBClassifier": XGBClassifier(),

          }
    return models