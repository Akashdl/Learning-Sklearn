#Rules for importing

    # 1. Python level packages ( import os, import sys)

    # 2. Module level imports ( import numpy as np)

    # 3. Local level import ( importing from local files )

# If in doubt use sort import functionality from vs code

from cProfile import label
from tkinter.tix import Y_REGION
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, linear_model, metrics, model_selection

def test_regularisation():
    X = np.random.uniform(low=0.0, high=10.0, size=25)

    y = 3.0 * X

    noise = np.random.randn(25) * 2.0
    y_noisy = y + noise

    # sns.scatterplot(x=X, y=y) # return matplotlib axes 
    # sns.scatterplot(x=X, y=y_noisy)
    # plt.grid()
    # plt.show() # use matplotlib to plot

    clean_linear_regressor = linear_model.LinearRegression()
    noisy_linear_regressor = linear_model.LinearRegression()

    clean_linear_regressor.fit(X=X.reshape(-1,1), y=y)
    noisy_linear_regressor.fit(X=X.reshape(-1,1), y=y_noisy)

    X_eval = np.linspace(0.0, 10.0, 100)
    y_eval_clean = clean_linear_regressor.predict(X_eval.reshape(-1,1))
    y_eval_noisy = noisy_linear_regressor.predict(X_eval.reshape(-1,1))

    y_noisy_with_outlier = y_noisy.copy()
    y_noisy_with_outlier[0] += 500

    outlier_linear_regressor = linear_model.LinearRegression()
    outlier_linear_regressor.fit(X=X.reshape(-1,1), y=y_noisy_with_outlier)
    

    # sns.scatterplot(x=X, y=y) # return matplotlib axes 
    # sns.scatterplot(x=X, y=y_noisy)
    plt.scatter(X, y, label="Clean")
    plt.scatter(X, y_noisy, label="Noisy")
    plt.plot(X_eval, y_eval_clean, label="Clean regressor")
    plt.plot(X_eval, y_eval_noisy, label="Noisy regressor")
    plt.legend()
    plt.show() # use matplotlib to plot



if __name__ == "__main__":
    test_regularisation()
