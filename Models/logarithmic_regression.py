import math
import statistics as st

# LOGARITHMIC REGRESSION: y = m * ln(x) + c

# 1. log-transform....
def log_transform(x):
    return [math.log(xi) for xi in x]   # xi > 0 hona chahiye

# 2. Slope (m) aur Intercept (c)....
def log_slope(x, y):
    x_log = log_transform(x)
    return st.covariance(x_log, y) / st.variance(x_log)

def log_intercept(x, y, m):
    x_log = log_transform(x)
    return st.mean(y) - m * st.mean(x_log)

def log_fit(x, y):
    m = log_slope(x, y)
    c = log_intercept(x, y, m)
    return m, c

# 3. Predict....
def log_predict(x, m, c):
    if type(x) == list or type(x) == tuple:
        return [m * math.log(xi) + c for xi in x]
    return m * math.log(x) + c

# 4. Gradient Descent version....
def log_gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    """
    Cost function: J(m,c) = (1/n) * Σ(actual - predicted)^2
    predicted = m*ln(xi) + c

    x_log ko normalize karta hai (mean=0, std=1) taaki
    gradient descent fast aur stable converge kare.
    """
    n = len(x)
    x_log = log_transform(x)

    # Step: x_log ko normalize karo
    mean_xl = st.mean(x_log)
    std_xl = st.std(x_log)
    x_log_norm = [(xl - mean_xl) / std_xl for xl in x_log]

    m, c = 0.0, 0.0
    for _ in range(epochs):
        y_pred = [m * x_log_norm[i] + c for i in range(n)]

        dm = (-2 / n) * sum(x_log_norm[i] * (y[i] - y_pred[i]) for i in range(n))
        dc = (-2 / n) * sum((y[i] - y_pred[i]) for i in range(n))

        m -= learning_rate * dm
        c -= learning_rate * dc

    # Step: wapas ln-space me convert karo
    m_real = m / std_xl
    c_real = c - (m * mean_xl / std_xl)
    return m_real, c_real



# demo

import pandas as  pd
if __name__ == "__main__":
    df = pd.read_excel("log_dependent_data.xlsx")
    x = df["data1"].tolist()
    y = df["data2"].tolist()
    
    print("--- Closed-Form (Least Squares, log-transformed) ---")
    m, c = log_fit(x, y)
    print(f"m = {m:.4f}, c = {c:.4f}")

    y_pred = log_predict(x, m, c)
    # print("Predictions:", [round(p, 3) for p in y_pred])

    print("\n--- Metrics ---")
    print(f"MSE: {st.mse(y, y_pred):.4f}")
    print(f"RMSE: {st.rmse(y, y_pred):.4f}")
    print(f"MAE: {st.mae(y, y_pred):.4f}")
    print(f"R²: {st.r_squared(y, y_pred):.4f}")

    print("\n--- Gradient Descent Version ---")
    m_gd, c_gd = log_gradient_descent(x, y, learning_rate=0.01, epochs=2000)
    print(f"m = {m_gd:.4f}, c = {c_gd:.4f}")