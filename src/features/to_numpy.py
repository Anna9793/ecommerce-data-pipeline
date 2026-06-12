from sklearn.base import BaseEstimator, TransformerMixin

class ToNumpy(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values