import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Created using this tutorial:
# https://medium.com/data-science/linear-regression-using-gradient-descent-97a6c8700931

def main():
    plt.rcParams['figure.figsize'] = (8.0, 6.0)

    data = pd.read_csv('data.csv')

    X = data["km"]
    Y = data["price"]

    m = 0
    c = 0

    X_mean = np.mean(X)
    X_std = np.std(X)
    X_scaled = (X - X_mean) / X_std

    L = 1e-2  # Learning Rate
    epochs = 1000  # Number of iterations

    n = float(len(X_scaled))

    # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = m * X_scaled + c
        D_m = (-2/n) * sum(X_scaled * (Y - Y_pred))
        D_c = (-2/n) * sum(Y - Y_pred)
        m -= L * D_m
        c -= L * D_c

    print("Slope (m):", m, "Intercept (c):", c)

    # Calculate predictions for plotting
    Y_pred = m * X_scaled + c

    # Sort the values for plotting a line
    sorted_indices = X_scaled.argsort()
    X_sorted = X_scaled.iloc[sorted_indices]
    Y_pred_sorted = Y_pred.iloc[sorted_indices]

    # Plotting the data and regression line
    plt.scatter(X_scaled, Y, color='blue')
    plt.plot(X_sorted, Y_pred_sorted, color='red')
    plt.xlabel('Normalized Mileage (km)')
    plt.ylabel('Price')
    plt.title('Linear Regression Fit')
    plt.show()

if __name__ == "__main__":
    main()
