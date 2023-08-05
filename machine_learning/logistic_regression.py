"""
                                    Tahmin Edilen Sınıf
                                    1               0
        Gerçek Sınıf    1       True Positive(TP)   False Negative(FN)
                        0       False Positive(FP)  True Negative(TN)

FP = Type I Error
FN = Type II Error

Accuracy: Doğru sınıflandırma oranıdır: (TP + TN) / (TP + TN + FP + FN)
Precision: Pozitif sınıf(1) tahminlerinin başarı oranıdır: TP / (TP + FP)
    Sınıflandırılan pozitif değerlerin gerçekten pozitif olma oranıdır

1 = Sahtekar sınıfı olsun
Sahtekar olarak tahmin ettiklerimiz ne kadar doğru
Yani sahtekar olarak tahmin ettiklerimizin gerçekte ne kadarı sahtekar onu test ederiz

Recall: Pozitif sınıfın(1) doğru tahmin edilme oranıdır: TP / (TP + FN)
    Gerçekten pozitif olanların sınıflandırılan pozitif olma oranıdır

Sahtekarlık işlerinin ne kadarını doğru tahmin ettiğimizi test ederiz

F1 Score: Precision ve Recall metriklerinin harmonik ortalamasıdır: 2 * Precision * Recall / (Precision + Recall)
"""

"""
logLoss = 1/m *( i=1,m[ -yi * log(p(y_pred_i) - (1-yi) * log(1-y_pred_i)) ] )

# 0.80 1e yakın bir sayı yani bir çeşitlilik yok. Buradan hesaplanan entropi daha düşük olucaktır
Churn      Predicted_Churn     Probability_of_Class_1       LogLoss
1           1                           0.8                 -1 * log(0.8) = 0.096910013 
1           0                           0.48                -1 * log(0.48) = 0.31875876262
0           0                           0.3                 -(1-0) * log(1-0.3) = 0.15490195998
1           0                           0.45                
0           1                           0.55
1           1                           0.7
0           0                           0.42                -(1-0) * log(1-0.42) = 0.23657200643
0           0                           0.35
1           1                           0.6                 -1 * log(0.6) = 0.22184874961
1           1                           0.7
"""

######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

import matplotlib

matplotlib.use("Qt5Agg")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("datasets/diabetes.csv")

##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

df["Outcome"].value_counts() / len(df) * 100

df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()


def plot_numerical_cols(df, num_cols):
    df[num_cols].hist(bins=20)
    plt.xlabel(num_cols)
    plt.show(block=True)  # birbirini ezmemesi için block=True


for i in df.columns:
    plot_numerical_cols(df, i)

cols = [col for col in df.columns if col not in "Outcome"]

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})


def target_summary_with_num(df, target, num_col):
    print(df.groupby(target).agg({num_col: "mean"}))


for col in cols:
    target_summary_with_num(df, "Outcome", col)

######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()
df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

######################################################
# Model & Prediction
######################################################

X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

log_model = LogisticRegression().fit(X, y)
b = log_model.intercept_
w = log_model.coef_

y_pred = log_model.predict(X)
proba = np.argmax(log_model.predict_proba(X), axis=1)


######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939

######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# plot_roc_curve(log_model, X_test, y_test)
RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)
