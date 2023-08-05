######################################################
# Sales Prediction with Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("datasets/advertising.csv")
df.head()

X = df[["TV"]]
y = df[["sales"]]

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
b = reg_model.intercept_[0]

# tv'nin katsayısı (w1)
w = reg_model.coef_[0][0]

##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
b + 150 * w

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51

y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

reg_model.score(X, y)  # Bağımsız değişkenlerin bağımlı değişkeni açıklayabilme yüzdesidir

######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)

y = df[["sales"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
b = reg_model.intercept_[0]

# tv'nin katsayısı (w1)
w = reg_model.coef_[0]

##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40
# 2.90
# 0.0468431 , 0.17854434, 0.00258619
# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002
# 2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T
reg_model.predict(yeni_veri)

reg_model.predict(pd.DataFrame([[30], [10], [40]]).T)
reg_model.predict(np.array([30, 10, 40]).reshape(1, -1))

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)  # bağımsız değişkenler modele sorulur ve oda bağımlı değişkenini tahmin eder
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

