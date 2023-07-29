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
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Görev 1 : Keşifçi Veri Analizi

# Adım 1: Genel resmi inceleyiniz.

df = pd.read_csv("datasets/diabetes.csv")


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


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

# Kategorik değişkenlerin analizi
def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("##########################################")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)


# Numerik değişkenlerin analizi
def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(10, 5))
        # df[numerical_col].hist(bins=20)
        sns.histplot(df[numerical_col], kde=True)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)


# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin
# ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

# Kategorik değişkenlerin target değişkene göre analizi
def target_summary_with_cat(df, target, categorical_col):
    print(pd.DataFrame({"Target_Mean": df.groupby(cat_cols)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}))
    print("#" * 50)


for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)


# Numerik değişkenlerin target değişkenine göre analizi

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}))
    print("#" * 50)


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(df, col_name, q1=0.20, q3=0.95):
    # q1=0.05, q3=0.95 | q1=0.25, q3=0.75
    q1 = df[col_name].quantile(q1)
    q3 = df[col_name].quantile(q3)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return low, up


for col in num_cols:
    print(col, outlier_thresholds(df, col))

"""
Pregnancies (-12.5, 23.5)
Glucose (-34.0, 310.0)
BloodPressure (15.0, 135.0)
SkinThickness (-66.0, 110.0)
Insulin (-439.5, 732.5)
BMI (-1.8424999999999976, 72.13749999999999)
DiabetesPedigreeFunction (-1.1507749999999994, 2.503024999999999)
Age (-29.5, 110.5)
"""


def check_outlier(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up) | (df[col_name] < low)].any(axis=None):
        return True
    return False


for col in num_cols:
    print(col, check_outlier(df, col))


def grab_outlier(df, col_name, index=False):
    low, up = outlier_thresholds(df, col)

    if df[(df[col_name] > up) | (df[col_name] < low)].shape[0] > 0:
        print(f"{col_name} Aykırı değer sayısı:", len(df[(df[col_name] > up) | (df[col_name] < low)]))
        print(df[(df[col_name] > up) | (df[col_name] < low)].head())
    else:
        print(f"{col} için aykırı değer yok.")

    if index:
        outlier_index = df[(df[col_name] > up) | (df[col_name] < low)].index
        return outlier_index


for col in num_cols:
    print(col, grab_outlier(df, col, True))
    print("*" * 50)


# Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)

    if na_name:
        return na_columns


missing_values_table(df)
# Adım 7: Korelasyon analizi yapınız.

plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
plt.tight_layout()
plt.show()

# Model
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17, shuffle=True, stratify=y)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

"""
Accuracy: 0.76
Recall: 0.66
Precision: 0.65
F1: 0.66
Auc: 0.74
"""


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(12, 7))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)


# Görev 2 :Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin
# değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

def maybe_missing(df, col_name):
    variables = df[df[col_name] == 0].shape[0]
    return variables


for col in num_cols:
    print(col, maybe_missing(df, col))

na_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

for col in na_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])  # 0 değerinin anlamlı olmadığı değişkenler için NA ataması

na_columns = missing_values_table(df, True)


def missing_vs_target(df, target, na_columns):
    temp_df = df.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}))


missing_vs_target(df, "Outcome", na_columns)

# Filling in missing values
for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()


def replace_with_thresholds(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    df.loc[(df[col_name] > up), col_name] = up
    df.loc[(df[col_name] < low), col_name] = low


# Outlier Suppression
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

"""
# Outliers are still in there in the name of Insulin adn SkinThickness variable

dff = pd.get_dummies(df[["Insulin","SkinThickness"]], drop_first=True)
dff.isnull().sum()

# Standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# Implement the KNN method
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["Insulin"] = dff["Insulin"]
df["SkinThickness"]= dff["SkinThickness"]
df.isnull().sum()
"""

# Adım 2: Yeni değişkenler oluşturunuz.

df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_DiabetesPedigreeFunction'] = df['Age'] * df['DiabetesPedigreeFunction']
df['Pregnancies_Per_Age'] = df['Pregnancies'] / df['Age']
df['Glucose_Minus_Insulin'] = df['Glucose'] - df['Insulin']
df['GIR'] = df['Glucose'] / df['Insulin']
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
df["Age_Cat"] = pd.cut(df["Age"], bins=[0, 15, 25, 64, 82], labels=["Child", "Young", "Adult", "Senior"])
df['BMI_DiabetesPedigree'] = df['BMI'] * df['DiabetesPedigreeFunction']
df['Age_Insulin'] = df['Age'] * df['Insulin']
df['Glucose_BMI_Difference'] = df['Glucose'] - df['BMI']
df["glucose_per_bmi"] = df["Glucose"] / df["BMI"]
df["insulin_per_age"] = df["Insulin"] / df["Age"]


# # # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma
# df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
# df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
# df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (
#         (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
# df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
# df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (
#         (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
# df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
# df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
# df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
#
# # Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
# df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
# df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
# df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (
#         (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
# df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
# df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (
#         (df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
# df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
# df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
# df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"


def categorize_blood_pressure(blood_pressure):
    if blood_pressure < 60:
        return 'Low'
    elif blood_pressure >= 60 and blood_pressure <= 80:
        return 'Normal'
    else:
        return 'High'


df['BloodPressureLevel'] = df['BloodPressure'].apply(categorize_blood_pressure)


def insulin_level(dataframe):
    if dataframe["Insulin"] <= 100:
        return "Normal"
    if dataframe["Insulin"] > 100 and dataframe["Insulin"] <= 126:
        return "Prediabetes"
    elif dataframe["Insulin"] > 126:
        return "Diabetes"


df["Insulin_Level"] = df.apply(insulin_level, axis=1)


def glucose_level(glucose):
    if 16 <= glucose <= 140:
        return "Normal"
    else:
        return "Abnormal"


df["Glucose"] = df["Glucose"].apply(glucose_level)


# def glucose_level(dataframe, col_name="Glucose"):
#     if 16 <= dataframe[col_name] <= 140:
#         return "Normal"
#     else:
#         return "Abnormal"
# df["Glucose_Level"] = df.apply(glucose_level, axis=1)
# df["Glucose_Level"] = df.apply(glucose_level, axis=1)

def categorize_skin_thickness(skin_thickness):
    if skin_thickness < 10:
        return 'Thin'
    elif skin_thickness >= 10 and skin_thickness < 25:
        return 'Normal'
    else:
        return 'Thick'


df['SkinThicknessCategory'] = df['SkinThickness'].apply(categorize_skin_thickness)


def categorize_pregnancy_group(pregnancies):
    if pregnancies == 0:
        return 'No Pregnancy'
    elif pregnancies <= 3:
        return 'Low Pregnancy'
    else:
        return 'High Pregnancy'


df['PregnancyGroup'] = df['Pregnancies'].apply(categorize_pregnancy_group)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.


binary_col = [col for col in df.columns if
              df[col].dtypes in ['object', 'category'] and df[col].nunique() == 2 and col not in ["OUTCOME"]]


def label_encoder(df, binary_col):
    le = LabelEncoder()
    df[binary_col] = le.fit_transform(df[binary_col])
    return df


for col in binary_col:
    label_encoder(df, col)


def rare_analyser(df, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(df[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": df[col].value_counts(),
            "RATIO": df[col].value_counts() / len(df),
            "TARGET_MEAN": df.groupby(col)[target].mean()
        }), end="\n\n")


rare_analyser(df, "Outcome", cat_cols)


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
rare_analyser(new_df, "Outcome", cat_cols)


def one_hot_encoder(df, categorical_cols, drop_first=True):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return df


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.


rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# mm = MinMaxScaler()
# df[num_cols] = mm.fit_transform(df[num_cols])
# ss = StandardScaler()
# df[num_cols] = ss.fit_transform(df[num_cols])

# Adım 5: Model oluşturunuz.

"""
# rare_analyser sonrası new_df üzerinden yapılan modelin sonuçları
my_df = new_df.copy()
my_df = one_hot_encoder(my_df, ohe_cols)
my_df[num_cols] = rs.fit_transform(my_df[num_cols])
X = my_df.drop("Outcome", axis=1)
y = my_df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Accuracy: 0.76
# Recall: 0.69
# Precision: 0.59
# F1: 0.64
# Auc: 0.74
"""

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

"""
Accuracy: 0.78
Recall: 0.71
Precision: 0.64
F1: 0.68
Auc: 0.76
"""


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
