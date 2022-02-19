from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
from config import BEST_MODEL_ADD


class Model(ABC):
    """
    Base classes for any ML model
    """

    def __init__(self):
        ...

    @abstractmethod
    def train(self, x, y):
        ...

    @abstractmethod
    def predict(self, x):
        ...

    @classmethod
    def save_model(self, loc):
        ...


class LinearModel(Model):
    """Linear Regression Model from sklearn"""
    def __init__(self):
        self.model = LinearRegression()
        super().__init__()

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        y_hat = self.model.predict(x)
        return y_hat


def load_model():
    """
    The method allows user to load trained model from pickle file
    :return: Model class with the trained model in it
    """
    if BEST_MODEL_ADD == '':
        raise OSError('No trained model exists')
    else:
        pass
