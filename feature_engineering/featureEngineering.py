import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")
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
    print(col, check_outlier(df, col))  # replace_with_thresholds kullandık ve aykırı değer kalmadı

###################
# Recap
###################
df = load()
outlier_thresholds(df, "Age")  # Aykırı değeri saptama işlemini yaptık ve threshold değerlerini aldık
check_outlier(df, "Age")  # thresholdlara göre aykırı değer var mı yok mu
grab_outlier(df, "Age", True)  # aykırı değerleri görmek için
remove_outlier(df, "Age")  # aykırı değerleri silmek için
replace_with_thresholds(df, "Age")  # aykırı değerleri baskılamak için threshold değerlerine göre değiştir

check_outlier(df, "Age")  # aykırı değer kalmadı

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

#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)

    if na_name:
        return na_columns


missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################


df.Age.fillna(df.Age.mean()).isnull().sum()
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x,
               axis=0).head()  # 1 sütunu al demek bütün satırlara bak demektir o yüzden axis 0

dff.isnull().sum()
dff.Embarked.fillna(dff.Embarked.mode()[0]).isnull().sum()

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
missing_values_table(dff)

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
missing_values_table(dff)

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["age_imputed_knn"] = dff["Age"]
df.loc[df["Age"].isnull()][["Age", "age_imputed_knn"]].head()

#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

na_cols = missing_values_table(df, True)


def missing_vs_target(df, target, na_columns):
    temp_df = df.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}))


missing_vs_target(df, "Survived", na_cols)

#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

le = LabelEncoder()
le.fit_transform(df["Sex"])[:5]
le.inverse_transform([0, 1])


def label_encoder(df, binary_col):
    le = LabelEncoder()
    df[binary_col] = le.fit_transform(df[binary_col])
    return df


binary_col = [col for col in df.columns if
              df[col].dtype not in ["float64", "float32", "float", "int64", "int32", "int"] and df[col].nunique() == 2]
# nunique boş verileri almaz o yüzden len(df[col].unique()) yapmak yerine bu şekilde kullandık


for col in binary_col:
    label_encoder(df, col)

#############################################
# One-Hot Encoding
#############################################

df = load()
pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
pd.get_dummies(df, columns=binary_col + ["Embarked"], drop_first=True).head()


def one_hot_encoder(df, categorical_cols, drop_first=True):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return df


df = load()

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
one_hot_encoder(df, ohe_cols).head()

#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({
        col_name: df[col_name].value_counts(),
        "Ratio": df[col_name].value_counts() / len(df) * 100
    }))
    print("#" * 50)

    if plot:
        sns.countplot(x=col_name, data=df)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(df, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(df[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": df[col].value_counts(),
            "RATIO": df[col].value_counts() / len(df),
            "TARGET_MEAN": df.groupby(col)[target].mean()
        }), end="\n\n")


rare_analyser(df, "TARGET", cat_cols)


#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(df, rare_perc):
    temp_df = df.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtype == "O" and
                    (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    # bu şarta uyan kolon var mı
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])
    return temp_df


new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)

#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_minmax"] = mms.fit_transform(df[["Age"]])

df.describe().T

age_cols = [col for col in df.columns if "Age" in col]


def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        sns.histplot(df[numerical_col], bins=20)
        plt.show(block=True)

        # df[numerical_col].hist(bins=20)
        # plt.show(block=True)


for col in age_cols:
    num_summary(df, col, True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################
df = load()
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 9.4597, p-value = 0.0000
# p-value < 0.05 olduğu için H0 hipotezi reddedilir.
# Yani kabin bilgisi olanlar ve olmayanlar arasında hayatta kalma oranı arasında anlamlı bir fark vardır.

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split()))
df.head()

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([name for name in x.split() if name.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

###################
# Regex ile Değişken Türetmek
###################

df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=True)
df[["NEW_TITLE", "Survived", "Age"]].groupby("NEW_TITLE").agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month

# day name
dff['day_name'] = dff["Timestamp"].dt.day_name()

dff.head()

#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################

df = load()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.groupby("NEW_SEX_CAT").agg({"Survived": "mean"})

#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)
df.drop(["CABIN", "TICKET", "NAME"], inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if
               df[col].dtype not in ["float64", "float32", "float", "int64", "int32", "int"] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)
df = rare_encoder(df, 0.01)

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

sc = StandardScaler()
df[num_cols] = sc.fit_transform(df[num_cols])

#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

# dff = load()
# dff.dropna(inplace=True)
# dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
# y = dff["Survived"]
# X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
# rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# accuracy_score(y_pred, y_test)


# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
