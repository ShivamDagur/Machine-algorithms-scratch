import math

# 1 - STATISTICS FUNCTION

# 1.a : Mean...
def mean(data) :
  return sum(data)/len(data)

# 1.b : Variance...
def variance(data):
  m = mean(data)
  return sum([(x-m)**2 for x in data])/len(data)

# 1.c : Standard Deviation...
def std(data):
  return math.sqrt(variance(data))

# 1.d : Covariance...
def covariance(data1,data2):
  res = 0
  m1 = mean(data1)
  m2 = mean(data2)
  for i in range(len(data1)):
    res += (data1[i]-m1)*(data2[i]-m2)
  return res/(len(data1))

# 1.e : Correlation...
def correlation(data1,data2):
  return covariance(data1,data2)/std(data1)/std(data2)



# 2 - ERROR FUNCTIONS

# 2.a : Sum of Squared Errors....
def sse(y_actual, y_predicted):
    return sum((y_actual[i] - y_predicted[i]) ** 2 for i in range(len(y_actual)))

# 2.b : Mean Squared Error....
def mse(y_actual, y_predicted):
    return sse(y_actual, y_predicted) / len(y_actual)

# 2.c : Root Mean Squared Error....
def rmse(y_actual, y_predicted):
    return math.sqrt(mse(y_actual, y_predicted))

# 2.d : Mean Absolute Error....
def mae(y_actual, y_predicted):
    n = len(y_actual)
    return sum(abs(y_actual[i] - y_predicted[i]) for i in range(n)) / n

# 2.e : Total Sum of Squares....
def sst(y_actual):
    """
    SST = Σ(actual - mean_y)^2
    """
    mean_y = mean(y_actual)
    return sum((yi - mean_y) ** 2 for yi in y_actual)

# 2.f : R^2 Score....
def r_squared(y_actual, y_predicted):
    """
    R² (Coefficient of Determination):
    R² = 1 - (SSE / SST)
    """
    return 1 - (sse(y_actual, y_predicted) / sst(y_actual))
  
  
# demo 

import pandas as pd

if __name__ == "__main__" :
  df = pd.read_excel("linear_dependent_data.xlsx")
  print(df.head())
  x = df["data1"].tolist()
  y = df["data2"].tolist()
  print(f"Mean: {mean(x):.4f}")
  print(f"Variance: {variance(x):.4f}")
  print(f"Std: {std(x):.4f}")
  print(f"Covariance: {covariance(x,y):.4f}")
  print(f"Correlation: {correlation(x,y):.4f}")