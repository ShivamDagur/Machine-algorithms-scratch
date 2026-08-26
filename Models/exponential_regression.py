import math
from Statistical_function import statistics as st

# EXPONENTIAL REGRESSION: y = a * e^(b * x)

# 1. log-transform y (ln(y) = ln(a) + b*x  ->  linear form)....
def exp_transform(y):
    return [math.log(yi) for yi in y]   

# 2. Slope (b) aur Intercept (log-space me, phir "a" nikaalte hain)....
def exp_slope(x, y):
    y_log = exp_transform(y)
    return st.covariance(x, y_log) / st.variance(x)

def exp_intercept(x, y, b):
    y_log = exp_transform(y)
    return st.mean(y_log) - b * st.mean(x)

def exp_fit(x, y):
    b = exp_slope(x, y)
    c_log = exp_intercept(x, y, b)
    a = math.exp(c_log) 
    return a, b


# 3. Predict....
def exp_predict(x, a, b):
    if type(x) == list or type(x) == tuple:
        return [a * math.exp(b * xi) for xi in x]
    return a * math.exp(b * x)


# 4. Gradient Descent version....
def exp_gradient_descent(x, y, learning_rate=0.01, epochs=2000):
    """
    y ko log-transform karte hain: ln(y) = b*x + c_log
    """
    n = len(x)
    y_log = exp_transform(y)

    # Step: x ko normalize karo
    mean_x = st.mean(x)
    std_x = st.std(x)
    x_norm = [(xi - mean_x) / std_x for xi in x]

    m, c = 0.0, 0.0
    for _ in range(epochs):
        y_pred = [m * x_norm[i] + c for i in range(n)]

        dm = (-2 / n) * sum(x_norm[i] * (y_log[i] - y_pred[i]) for i in range(n))
        dc = (-2 / n) * sum((y_log[i] - y_pred[i]) for i in range(n))

        m -= learning_rate * dm
        c -= learning_rate * dc

    # Step: wapas original scale me convert
    b_real = m / std_x
    c_real = c - (m * mean_x / std_x)
    a_real = math.exp(c_real) 
    return a_real, b_real


# demo

import pandas as pd

if __name__ == "__main__":
    df = pd.read_excel("Data/exp_dependent_data.xlsx")
    x = df["data1"].tolist()
    y = df["data2"].tolist()

    print("--- Closed-Form (Least Squares, log-transformed) ---")
    a, b = exp_fit(x, y)
    print(f"a = {a:.4f}, b = {b:.4f}")

    y_pred = exp_predict(x, a, b)

    print("\n--- Metrics ---")
    print(f"MSE: {st.mse(y, y_pred):.4f}")
    print(f"RMSE: {st.rmse(y, y_pred):.4f}")
    print(f"MAE: {st.mae(y, y_pred):.4f}")
    print(f"R²: {st.r_squared(y, y_pred):.4f}")

    print("\n--- Gradient Descent Version ---")
    a_gd, b_gd = exp_gradient_descent(x, y, learning_rate=0.01, epochs=2000)
    print(f"a = {a_gd:.4f}, b = {b_gd:.4f}")