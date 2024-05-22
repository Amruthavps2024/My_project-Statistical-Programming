import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
class SimpleLinearRegression:
    def fit(self, X, y):
       
        X_b = np.c_[np.ones((len(X), 1)), X]  # Add a bias term
        params = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
      
        return params
    def predict(self, X, params):
        X_b = np.c_[np.ones((len(X), 1)), X]
        return X_b.dot(params)

def linear_regression():
    df = pd.read_csv('data/cars.csv')
    df_new = df.rename(columns={'Weight': 'wgt','CO2': 'CD'})
    df_new = df_new.dropna()

    X = df_new['wgt'].values.reshape(-1, 1)
    y = df_new['CD'].values.reshape(-1, 1)
    
    
    # Splitting into Training/Testing
    spl = 0.2
    N = len(X)
    sample = int(spl*N)
    X_train, X_test,  = X[:-sample], X[-sample:]
    y_train, y_test   = y[:-sample],y[-sample:]
    # Building the Regression Model
    reg = SimpleLinearRegression()
    params = reg.fit(X_train, y_train)
    # Assess the Accuracy
    predictions = reg.predict(X_test, params)
    # Calculate R-squared
    mean_y = np.mean(y_test)
    total_variation = np.sum((y_test - mean_y) ** 2)
    explained_variation = np.sum((predictions - mean_y) ** 2)
    r_squared = 1 - (explained_variation / total_variation)
    print('R-squared (Efficiency):', round(r_squared, 3))

    # Calculate MSE, MAE, RMSE
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)

    print('Mean Squared Error (MSE):', round(mse, 3))
    print('Mean Absolute Error (MAE):', round(mae, 3))
    print('Root Mean Squared Error (RMSE):', round(rmse, 3))

    # Using the model
    test_data = np.array([100,110,500,1600,1000,200,1750]).reshape(-1, 1)
    predictions = reg.predict(test_data, params)
    print(predictions)

    # Visualize the predictions against actual values
    plt.scatter(X, y, color='b', label='Actual')
    plt.plot(test_data, predictions, color='r', marker='x', label='Predicted')
    plt.xlabel('Weight')
    plt.ylabel('CO2 Emission')
    plt.title('Linear Regression_Numpy:  Weight vs CO2 Emission')
    plt.legend()
    plt.show()

execution_time = timeit.timeit(linear_regression, number=1)
print(f"Execution Time: {execution_time} seconds")
