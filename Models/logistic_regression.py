import math
from Statistical_function import statistics as st
# LOGISTIC REGRESSION (Univariate): p = sigmoid(m*x + c)

# 1. Sigmoid function....
def sigmoid(z):
    # bahut bade negative/positive z pe overflow na ho isliye clip kar rahe
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1 + ez)


# 2. Predict probability....
def log_reg_predict_prob(x, m, c):
    if type(x) == list or type(x) == tuple:
        return [sigmoid(m * xi + c) for xi in x]
    return sigmoid(m * x + c)


# 3. Predict class (0/1) using threshold....
def log_reg_predict_class(x, m, c, threshold=0.5):
    probs = log_reg_predict_prob(x, m, c)
    if type(probs) == list:
        return [1 if p >= threshold else 0 for p in probs]
    return 1 if probs >= threshold else 0


# 4. Cost function (Binary Cross-Entropy / Log Loss)....
def log_reg_cost(y, y_pred_prob):
    """
    J(m,c) = -(1/n) * Σ [ y*log(p) + (1-y)*log(1-p) ]
    """
    n = len(y)
    eps = 1e-15  # log(0) avoid karne ke liye
    total = 0
    for i in range(n):
        p = min(max(y_pred_prob[i], eps), 1 - eps)  # p ko (0,1) range me clip karo
        total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
    return -total / n


# 5. Gradient Descent version....
def log_reg_gradient_descent(x, y, learning_rate=0.1, epochs=1000):
    """
    Cost function: J(m,c) = -(1/n) * Σ [ y*log(p) + (1-y)*log(1-p) ]
    p = sigmoid(m*xi + c)

    x ko normalize karta hai (mean=0, std=1) taaki
    gradient descent fast aur stable converge kare.
    """
    n = len(x)

    # Step: x ko normalize karo
    mean_x = st.mean(x)
    std_x = st.std(x)
    x_norm = [(xi - mean_x) / std_x for xi in x]

    m, c = 0.0, 0.0
    cost_history = []

    for _ in range(epochs):
        z = [m * x_norm[i] + c for i in range(n)]
        p = [sigmoid(zi) for zi in z]

        # gradient of BCE loss wrt m aur c (derivation same shape jaisa linear
        # regression ka MSE gradient hota hai, kyunki sigmoid + log-loss
        # combo se yeh simple form nikalta hai)
        dm = (1 / n) * sum((p[i] - y[i]) * x_norm[i] for i in range(n))
        dc = (1 / n) * sum((p[i] - y[i]) for i in range(n))

        m -= learning_rate * dm
        c -= learning_rate * dc

        cost_history.append(log_reg_cost(y, p))

    # Step: wapas original x-scale me convert karo
    m_real = m / std_x
    c_real = c - (m * mean_x / std_x)
    return m_real, c_real, cost_history


# 6. Accuracy metric....
def log_reg_accuracy(y_true, y_pred_class):
    n = len(y_true)
    correct = sum(1 for i in range(n) if y_true[i] == y_pred_class[i])
    return correct / n


# demo

import pandas as pd
if __name__ == "__main__":
    df = pd.read_excel("Data/logistic_dependent_data.xlsx")
    x = df["data1"].tolist()
    y = df["data2"].tolist()   # y sirf 0/1 hona chahiye

    print("--- Gradient Descent (Logistic Regression) ---")
    m, c, cost_history = log_reg_gradient_descent(x, y, learning_rate=0.1, epochs=2000)
    print(f"m = {m:.4f}, c = {c:.4f}")
    print(f"Final Cost (Log Loss): {cost_history[-1]:.4f}")

    y_pred_prob = log_reg_predict_prob(x, m, c)
    y_pred_class = log_reg_predict_class(x, m, c)

    print("\n--- Metrics ---")
    print(f"Log Loss: {log_reg_cost(y, y_pred_prob):.4f}")
    print(f"Accuracy: {log_reg_accuracy(y, y_pred_class):.4f}")