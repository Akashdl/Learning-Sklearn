import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import linear_model

def linear_regression_in_2D():
    # create artificial X and y 
    X = np.random.uniform(low=0.0, high=3.0, size=(100,2))
    #                            bias   noise
    y = 2.0*X[:,0] + 0.5*X[:,1] + 1 + np.random.randn(100)

    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(X,y)

  # x-coord   y-coord
    x_0_mesh, x_1_mesh = np.meshgrid(
        np.linspace(0,3,10),
        np.linspace(0,3,10)
    )

    x_0_mesh_flattened = x_0_mesh.flatten()
    x_1_mesh_flattened = x_1_mesh.flatten()
    print(x_0_mesh.shape)
    print(x_0_mesh_flattened.shape)
    print(X.shape) 

    points_2d = np.vstack((x_0_mesh_flattened, x_1_mesh_flattened))
    print(points_2d.shape)

    points_2d = points_2d.T
    print(points_2d.shape)

    z_predict = linear_regressor.predict(points_2d)
    print(z_predict.shape) 
    z_predict_mesh = z_predict.reshape(10,10)
    print(z_predict_mesh.shape)                        

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0], X[:,1], y)
    surf = ax.plot_surface(np.linspace(0,3,10), np.linspace(0,3,10), z_predict_mesh, cmap=cm.coolwarm)
    ax.set_xlabel('X[:,0]')
    ax.set_ylabel('X[:,1]')
    ax.set_zlabel('y')
    plt.show()

if __name__ == "__main__":
    linear_regression_in_2D()