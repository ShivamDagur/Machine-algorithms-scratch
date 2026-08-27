# Machine Algorithms from Scratch

## 📌 About the Project

This project implements core Machine Learning algorithms in Python **from scratch**, without relying on any ML library such as scikit-learn or TensorFlow.

### Why I'm building this

Most ML libraries let you train a model with a single `.fit()` call, but the underlying mathematics — statistics, error metrics, and optimization techniques like gradient descent — stays hidden. This project aims to derive and implement those formulas manually, so that the mathematical foundation of each algorithm is genuinely understood, not just its usage.

### How this will help going forward

- **Strong fundamentals** for learning more advanced algorithms later (Neural Networks, SVM, Ensemble methods), since they build on the same core concepts.
- **Better debugging and tuning skills** — understanding the underlying math makes it easier to diagnose issues in real-world models.
- **Useful for interviews and academic work**, including applying these concepts directly to other projects.
- **A reusable personal ML library** that keeps growing and can be reused in future projects.

### Core Concept

Each algorithm is built in two layers:
1. **Statistics layer** — common building blocks like mean, variance, covariance, correlation, and error metrics (MSE, RMSE, MAE, R²).
2. **Model layer** — uses the statistics layer to implement each algorithm's fitting and prediction logic, either via a closed-form solution or iterative Gradient Descent.

This structure makes it clear that different models share the same mathematical foundation, differing mainly in their assumptions and equations.

## 📂 Structure

```
Data/      → sample .xlsx datasets
Models/    → all algorithm .py files
  └── Statistical_function/statistics.py   → custom stats module (mean, variance, mse, r_squared, etc.)
```

## ✅ Progress

- [x] Custom statistics module (mean, variance, covariance, std, correlation, mse, rmse, mae, r_squared)
- [x] Linear Regression (closed-form + gradient descent)
- [x] Logarithmic Regression
- [x] Exponential Regression (log-transform + least squares)
- [x] Logistic Regression

## 📝 To-Do

- [ ] Polynomial Regression (curve fitting via Gaussian elimination)
- [ ] Multivariate Linear Regression (Normal Equation, custom matrix ops)
- [ ] Model-wrapper class (`fit` / `predict` / `evaluate`) with train/test split

## ⚙️ Run

```bash
git clone https://github.com/ShivamDagur/Machine-algorithms-scratch.git
cd Machine-algorithms-scratch/Models
python linear_regression.py
```

**Requirements:** `pandas`, `openpyxl`
