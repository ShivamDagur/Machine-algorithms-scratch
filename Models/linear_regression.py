import statistics as st

# 1 - SLOPE (m) AND INTERCEPT (c) — Least Squares Method --

# 1.a : Slope....
def slope(x, y):
    return st.covariance(x, y) / st.variance(x)

# 1.b : Intercept....
def intercept(x, y, m):
    return st.mean(y) - m * st.mean(x)


def fit(x, y):
    return slope(x, y), intercept(x, y, slope(x, y))



# 2. PREDICT

def predict(x, m, c):
    if type(x) == list or type(x) == tuple:
        return [m * xi + c for xi in x]
    return m * x + c

# 3. GRADIENT DESCENT VERSION (iterative approach)

def gradient_descent(x, y, learning_rate=0.01, epochs=2000):
    """
    Gradient Descent se m aur c seekhte hain (iteratively).

    Cost function: J(m,c) = (1/n) * Σ(actual - predicted)^2

    Gradients (partial derivatives):
    dJ/dm = (-2/n) * Σ(xi * (yi - (m*xi + c)))
    dJ/dc = (-2/n) * Σ(yi - (m*xi + c))

    Update rule:
    m = m - learning_rate * dJ/dm
    c = c - learning_rate * dJ/dc
    """
    n = len(x)

    # Step i: x ko normalize karo (mean=0, std=1)
    mean_x = st.mean(x)
    std_x = st.std(x)
    x_norm = [(xi - mean_x) / std_x for xi in x]

    # Step ii: normalized x pe gradient descent chalao
    m, c = 0.0, 0.0
    for _ in range(epochs):
        y_pred = [m * x_norm[i] + c for i in range(n)]

        dm = (-2 / n) * sum(x_norm[i] * (y[i] - y_pred[i]) for i in range(n))
        dc = (-2 / n) * sum((y[i] - y_pred[i]) for i in range(n))

        m -= learning_rate * dm
        c -= learning_rate * dc

    # Step iii: normalized-space ke m,c ko wapas original scale me convert karo
    m_real = m / std_x
    c_real = c - (m * mean_x / std_x)

    return m_real, c_real


# demo

import pandas as pd


if __name__ == "__main__":
    df = pd.read_excel("linear_dependent_data.xlsx")

    x = df["data1"].tolist()
    y = df["data2"].tolist()

    print("--- Closed-Form (Least Squares) ---")
    m, c = fit(x, y)
    print(f"m = {m:.4f}, c = {c:.4f}")

    y_pred = predict(x, m, c)
    # print("Predictions:", [round(p,3) for p in y_pred])

    print("\n--- Metrics ---")
    print(f"MSE: {st.mse(y, y_pred):.4f}")
    print(f"RMSE: {st.rmse(y, y_pred):.4f}")
    print(f"MAE: {st.mae(y, y_pred):.4f}")
    print(f"R²: {st.r_squared(y, y_pred):.4f}")
    print(f"Correlation (r): {st.correlation(x, y):.4f}")
    print("\n--- Gradient Descent Version ---")
    m_gd, c_gd = gradient_descent(x, y, learning_rate=0.01, epochs=2000)
    print(f"m = {m_gd:.4f}, c = {c_gd:.4f}")