import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

sns.boxplot(df["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

###################
# Aykırı Değer Var mı Yok mu?
###################
df[(df["Age"] > up) | (df["Age"] < low)]
df[(df["Age"] > up) | (df["Age"] < low)].index
df[(df["Age"] > up) | (df["Age"] < low)].any()
df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)


###################
# İşlemleri Fonksiyonlaştırmak
###################

def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    q1 = df[col_name].quantile(q1)
    q3 = df[col_name].quantile(q3)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return low, up


outlier_thresholds(df, "Age")
low, up = outlier_thresholds(df, "Age")
df[(df["Age"] > up) | (df["Age"] < low)]


def check_outlier(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up) | (df[col_name] < low)].any(axis=None):
        return True
    return False


for i in df.select_dtypes(exclude=["object", "category"]).columns:
    print(i, check_outlier(df, i))


def grab_col_names(df, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        df: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() < cat_th]
    cat_but_car = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)} \n{cat_cols}')
    print(f'num_cols: {len(num_cols)} \n{num_cols}')
    print(f'cat_but_car: {len(cat_but_car)} \n{cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)} \n{num_but_cat}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))


###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outlier(df, col_name, index=False):
    low, up = outlier_thresholds(df, col)

    if df[(df[col_name] > up) | (df[col_name] < low)].shape[0] > 10:
        print(df[(df[col_name] > up) | (df[col_name] < low)].head())
    else:
        print(df[(df[col_name] > up) | (df[col_name] < low)])

    if index:
        outlier_index = df[(df[col_name] > up) | (df[col_name] < low)].index
        return outlier_index


for col in num_cols:
    print(col, grab_outlier(df, col, True))

outlier_thresholds(df, "Age")  # lower ve upper değerleri verir
check_outlier(df, "Age")  # Bir değişkende outlier var mı yok mu
grab_outlier(df, "Age", True)  # Outlier değerleri görmek için

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

df.shape
low, up = outlier_thresholds(df, "Fare")
df[~((df["Fare"] > up) | (df["Fare"] < low))].shape


def remove_outlier(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    df_without_outliers = df[~((df[col_name] > up) | (df[col_name] < low))]
    return df_without_outliers


for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################
low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] > up) | (df["Fare"] < low)]["Fare"]
df.loc[((df["Fare"] > up) | (df["Fare"] < low)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up
df.loc[(df["Fare"] < low), "Fare"] = low


def replace_with_thresholds(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    df.loc[(df[col_name] > up), col_name] = up
    df.loc[(df[col_name] < low), col_name] = low

################################
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col)) # replace_with_thresholds kullandık ve aykırı değer kalmadı

###################
# Recap
###################
df = load()
outlier_thresholds(df, "Age") # Aykırı değeri saptama işlemini yaptık ve threshold değerlerini aldık
check_outlier(df, "Age") # thresholdlara göre aykırı değer var mı yok mu
grab_outlier(df, "Age", True) # aykırı değerleri görmek için
remove_outlier(df, "Age") # aykırı değerleri silmek için
replace_with_thresholds(df, "Age") # aykırı değerleri baskılamak için threshold değerlerine göre değiştir

check_outlier(df, "Age") # aykırı değer kalmadı


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df, col))


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]
df[df_scores < th]
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

