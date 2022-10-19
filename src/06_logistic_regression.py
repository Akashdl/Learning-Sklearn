from tokenize import PlainToken

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


def logistic_regression():
    age_to_pension = pd.read_csv('../data/age_to_getting_pension.csv')
    
    fig = plt.figure()
    plt.scatter(age_to_pension['age'], age_to_pension['get_pension'])
    plt.show()

if __name__ == "__main__":
    logistic_regression()
