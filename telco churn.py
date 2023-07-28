import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görev 1 : Keşifçi Veri Analizi

# Adım 1: Genel resmi inceleyiniz.
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")


def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(head))
    print("##################### Info #####################")
    print(df.info())
    print("##################### NA #####################")
    print(df.isnull().sum())
    print("##################### Quantiles #####################")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df.drop("customerID", axis=1, inplace=True)


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].dtypes != "O" and df[col].nunique() < cat_th]
    cat_but_car = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() > car_th]

    cat_cols += num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float", "int32", "float32", "int64",
                                                                "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)} \n{cat_cols}')
    print(f'num_cols: {len(num_cols)} \n{num_cols}')
    print(f'cat_but_car: {len(cat_but_car)} \n{cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)} \n{num_but_cat}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

# Categorical variable analysis
def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("##########################################")
    if plot:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=df[col_name], data=df)
        plt.tight_layout()
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)


# Numerical variable analysis
def num_summary(df, numerical_col, plot=False):
    print(df[numerical_col].describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]).T)
    print("##########################################")

    if plot:
        plt.figure(figsize=(10, 5))
        # df[numerical_col].hist(bins=20)
        sns.histplot(df[numerical_col], kde=True)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


# Kategorik değişkene göre hedef değişken ortalaması
def target_summary_with_cat(df, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}))
    print("#" * 50)


for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# churn değişkenine bağlı olarak kategorik değişken görselleştirme
for col in cat_cols:
    graph = pd.crosstab(index=df['Churn'], columns=df[col]).plot.bar(figsize=(10, 5), rot=0)
    plt.title(col)
    plt.show()


# Numerik değişkene göre hedef değişken ortalaması

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}))
    print("#" * 50)


for col in num_cols:
    target_summary_with_num(df, "Churn", col)


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(df, col_name, q1=0.10, q3=0.90):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, ": ", check_outlier(df, col))

# Adım 6: Eksik gözlem analizi yapınız.

df.isnull().sum()


def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

# for col in df.columns:
#     for val in df[col].values:
#         if val == " ":
#             print(col)
# df[df["tenure"] == 0] # Bu durumda tenure değişkeni 0 olan müşterilerin TotalCharges değişkeni NaN oluyor.
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

nan_values = df[df.isnull().any(axis=1)].index

# outputa baktığımızda total charges değişkenlerinin nan olduğu durumda tenure değeri sıfır. Bu demek oluyor ki total charges
# değişkeni nan olduğu durumlarda müşterinin şirkette kaldığı ay sayısı 0. Aşağıdaki kod satırında görüldüğü üzere tenure
# değişkeninin sıfır olduğu durumda TotalCharges NaN durumda.


df["TotalCharges"].fillna(df.iloc[nan_values]["MonthlyCharges"], inplace=True)  # 1 aylık ödemeler ile doldurmak için
# silmek isteseydik:
# df["TotalCharges"].dropna(inplace=True)
# NaN değerlere 0 yazmak için:
# df["TotalCharges"].fillna(0, inplace=True)


# Adım 7: Korelasyon analizi yapınız.

f, ax = plt.subplots(figsize=[10, 5])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Görev 2 : Feature Engineering


# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.


for col in num_cols:
    print(col, ": ", check_outlier(df, col))

df.isnull().sum().any()

# Base Model Kurulumu

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]


def one_hot_encoder(df, categorical_cols, drop_first=True):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return df


dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

CatboostModel = CatBoostClassifier(verbose=False, random_state=45).fit(X_train, y_train)
y_pred = CatboostModel.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 4)}")

"""
Accuracy: 0.7913
Recall: 0.6515
Precision: 0.4983
F1: 0.5647
Auc: 0.7397
"""


## Aykırı değer analizi

def outlier_thresholds(df, col_name, q1=0.10, q3=0.90):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(df, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(df, variable, q1=0.05, q3=0.95)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit


# Aykırı değer analizi ve baskılama işlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


# Adım 2: Yeni değişkenler oluşturunuz.

def categorize_tenure(tenure):
    if tenure <= 12:
        return 'Short-Term'
    elif tenure > 12 and tenure <= 24:
        return 'Mid-Term'
    else:
        return 'Long-Term'


df['TenureGroup'] = df['tenure'].apply(categorize_tenure)

df['HasInternetService'] = df['InternetService'].apply(lambda x: 'Yes' if x != 'No' else 'No')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


def categorize_family_size(partner, dependents):
    if partner == 'No' and dependents == 'Yes':
        return 'Family'
    elif partner == 'Yes' or dependents == 'Yes':
        return 'Couple'
    else:
        return 'Single'


df['FamilySize'] = df.apply(lambda x: categorize_family_size(x['Partner'], x['Dependents']), axis=1)

df['AvgMonthlyChargesPerTenure'] = df['TotalCharges'] / df['tenure']

df["MonthlyByTotal"] = df['MonthlyCharges'] / df['TotalCharges']


def categorize_monthly_charges(charges):
    if charges < 50:
        return 'Low Charges'
    elif charges >= 50 and charges < 100:
        return 'Moderate Charges'
    else:
        return 'High Charges'


df['MonthlyChargesCategory'] = df['MonthlyCharges'].apply(categorize_monthly_charges)


def bundled_streaming_services(tv, movies):
    if tv == 'Yes' and movies == 'Yes':
        return 'Bundled Streaming'
    else:
        return 'No Streaming Bundle'


df['StreamingServicesBundled'] = df.apply(lambda x: bundled_streaming_services(x['StreamingTV'], x['StreamingMovies']),
                                          axis=1)


def multiple_services_used(df):
    count = 0
    if df['PhoneService'] == 'Yes':
        count += 1
    if df['InternetService'] != 'No':
        count += 1
    if df['MultipleLines'] == 'Yes':
        count += 1
    if df['OnlineSecurity'] == 'Yes':
        count += 1
    if df['OnlineBackup'] == 'Yes':
        count += 1
    if df['DeviceProtection'] == 'Yes':
        count += 1
    if df['TechSupport'] == 'Yes':
        count += 1
    if df['StreamingTV'] == 'Yes':
        count += 1
    if df['StreamingMovies'] == 'Yes':
        count += 1

    return count


df['MultipleServicesUsed'] = df.apply(multiple_services_used, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def label_encoder(df, binary_col):
    labelencoder = LabelEncoder()
    df[binary_col] = labelencoder.fit_transform(df[binary_col])
    return df


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.


ohe_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]


def one_hot_encoder(df, categorical_cols, drop_first=True):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return df


df = one_hot_encoder(df, ohe_cols)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

np.isinf(df[num_cols]).sum()  # Sonsuz değerleri kontrol edin

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

rs = RobustScaler()  # Medyanı çıkar iqr'a böl.
df[num_cols] = rs.fit_transform(df[num_cols])

# Adım 5: Model oluşturunuz

X = dff.drop(["Churn"], axis=1)
y = dff["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

"""
Accuracy: 0.79
Recall: 0.65
Precision: 0.51
F1: 0.57
Auc: 0.74
"""
