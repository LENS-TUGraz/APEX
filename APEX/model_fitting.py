import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

def fit_model(x, y, model_type):
    if model_type == "LR":
        model = LinearRegression()
    elif model_type == "svr":
        model = SVR()
    elif model_type == "GP":
        kernel = RBF(length_scale=1.0)
        model = GaussianProcessRegressor(kernel=kernel)
    else:
        raise ValueError("Invalid model type. Supported types are 'linear', 'svr', and 'gaussian'.")

    model.fit(x, y)
    return model
