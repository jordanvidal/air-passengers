from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = XGBRegressor(max_depth=18, learning_rate=0.02, n_estimators=500,subsample=0.8, colsample_bytree=1, min_child_weight=5) 
        #RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
