import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def test_regularisation():
    np.random.seed(42)
    X = np.random.uniform(low=0.0, high=10.0, size=25)

    y = 3.0 * X

    noise = np.random.randn(25) * 2.0
    y_noisy = y + noise

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
    y_eval_outlier = outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    ridge_outlier_linear_regressor = linear_model.Ridge()
    ridge_outlier_linear_regressor.fit(X.reshape(-1,1), y_noisy_with_outlier)
    y_eval_outlier_ridge = ridge_outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    ridgecv_outlier_linear_regressor = linear_model.RidgeCV(alphas=10.0**np.arange(-2, 5, 1))
    ridgecv_outlier_linear_regressor.fit(X.reshape(-1,1), y_noisy_with_outlier)
    y_eval_outlier_ridgecv = ridgecv_outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    plt.figure()
    plt.scatter(X, y, label="Clean")
    plt.scatter(X, y_noisy, label="Noisy")
    plt.scatter(X, y_noisy_with_outlier, label="Noisy with outlier")
    
    plt.plot(X_eval, y_eval_clean, label="Clean regressor")
    plt.plot(X_eval, y_eval_noisy, label="Noisy regressor")
    plt.plot(X_eval, y_eval_outlier, label="Outlier regressor")
    plt.plot(X_eval, y_eval_outlier_ridge, label="Ridge outlier regressor")
    plt.plot(X_eval, y_eval_outlier_ridgecv, label="RidgeCV outlier regressor")
    plt.title("With fit_intercept=True")
    plt.legend()
    plt.show()

    outlier_linear_regressor = linear_model.LinearRegression(fit_intercept=False)
    outlier_linear_regressor.fit(X=X.reshape(-1,1), y=y_noisy_with_outlier)
    y_eval_outlier = outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    ridge_outlier_linear_regressor = linear_model.Ridge(fit_intercept=False)
    ridge_outlier_linear_regressor.fit(X.reshape(-1,1), y_noisy_with_outlier)
    y_eval_outlier_ridge = ridge_outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    ridgecv_outlier_linear_regressor = linear_model.RidgeCV(alphas=10.0**np.arange(-2, 5, 1), fit_intercept=False)
    ridgecv_outlier_linear_regressor.fit(X.reshape(-1,1), y_noisy_with_outlier)
    y_eval_outlier_ridgecv = ridgecv_outlier_linear_regressor.predict(X_eval.reshape(-1,1))

    plt.figure()
    plt.scatter(X, y, label="Clean")
    plt.scatter(X, y_noisy, label="Noisy")
    plt.scatter(X, y_noisy_with_outlier, label="Noisy with outlier")
    
    plt.plot(X_eval, y_eval_clean, label="Clean regressor")
    plt.plot(X_eval, y_eval_noisy, label="Noisy regressor")
    plt.plot(X_eval, y_eval_outlier, label="Outlier regressor")
    plt.plot(X_eval, y_eval_outlier_ridge, label="Ridge outlier regressor")
    plt.plot(X_eval, y_eval_outlier_ridgecv, label="RidgeCV outlier regressor")
    plt.title("With fit_intercept=False")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    test_regularisation()
