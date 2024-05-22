import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import timeit
# Load the data
df = pd.read_csv('data/cars.csv')
df_binary = df.rename(columns={ 'Weight': 'wgt','CO2': 'CO'})

# Renaming the columns for easier writing of the code
df_binary.head()
df_binary.info()

#//code adapted from Jolly K,2018
print(df_binary.isnull().any())
df_binary = df_binary.dropna()
#//end of adapted code
def scikit_linear_regression():
# Split into Features and Label
    X= df_binary['wgt'].values.reshape(-1, 1)
    y= df_binary['CO'].values.reshape(-1, 1)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Missing values in X:", np.isnan(X).sum())
    print("Missing values in y:", np.isnan(y).sum())
    print("Is X empty?", X.size == 0)
    print("Is y empty?", y.size == 0)
    print(df_binary.head())

# Splitting into Training/Testing
#//code adapted from Douglass, 2020
    test_dataset_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1, test_size=test_dataset_size)

# Building the Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

# Assess the Accuracy
    predictions = model.predict(X_test)
#//end of adapted code
# Obtaining the Mean Absolute Error

    score = metrics.mean_absolute_error(y_test, predictions)
    print('\nMAE:', round(score, 3))


# Calculate R-squared
    r_squared = r2_score(y_test,predictions)
    print('R-squared (Efficiency):', round(r_squared, 3))
    
#//  Code adapted from Greg,2013
#mse
    mse = metrics.mean_squared_error(y_test, predictions)
    print('MSE:', round(mse, 3))
#rmse
    rmse = np.sqrt(mse)
    print('RMSE:', round(rmse, 3))
#//end of adapted code


# Using the model
    Xnew = np.array([1200,110,1005,1600]).reshape(-1, 1)
# Feature Scaling for new data

# make a prediction
    ynew = model.predict(Xnew)

# show the inputs and predicted outputs
    for i in range(len(Xnew)):
        print(f"X={Xnew[i][0]}, Predicted={ynew[i][0]}")

# Visualize the predictions against actual values
    plt.scatter(X[:180], y[:180], color='b', label='Actual')
    plt.plot(Xnew, ynew, color='r', marker='x', label='Predicted') 
    plt.xlabel('Weight')
    plt.ylabel('CO2')
    plt.title('Linear Regression_Sci-kit:  Weight Vs CO2')
    plt.legend()
    plt.show()
execution_time = timeit.timeit(scikit_linear_regression, number=1)
print(f"Execution Time: {execution_time} seconds")