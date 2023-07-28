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

df[cat_cols].value_counts()

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin
# ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

for col in num_cols:
    print(df.groupby(col).agg({"Outcome": ["mean"]}))


# Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(df, col_name, q1=0.25, q3=0.75):
    q1 = df[col_name].quantile(q1)
    q3 = df[col_name].quantile(q3)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return low, up


for col in num_cols:
    print(col, outlier_thresholds(df, col))


def check_outlier(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up) | (df[col_name] < low)].any(axis=None):
        return True
    return False


for col in num_cols:
    print(col, check_outlier(df, col))


def grab_outlier(df, col_name, index=False):
    low, up = outlier_thresholds(df, col)

    if df[(df[col_name] > up) | (df[col_name] < low)].shape[0] > 10:
        print(f"{col_name} Aykırı değer sayısı:", len(df[(df[col_name] > up) | (df[col_name] < low)]))
        print(df[(df[col_name] > up) | (df[col_name] < low)].head())

    else:
        print(f"{col_name} Aykırı değer sayısı:", len(df[(df[col_name] > up) | (df[col_name] < low)]))
        print(df[(df[col_name] > up) | (df[col_name] < low)])

    if index:
        outlier_index = df[(df[col_name] > up) | (df[col_name] < low)].index
        return outlier_index


for col in num_cols:
    print(col, grab_outlier(df, col))
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

# Görev 2 :Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin
# değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df.loc[(df["Glucose"] == 0), "Glucose"] = np.nan
df.loc[(df["Insulin"] == 0), "Insulin"] = np.nan

na_cols = missing_values_table(df, True)


def missing_vs_target(df, target, na_columns):
    temp_df = df.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}))


missing_vs_target(df, "Outcome", na_cols)

df["Insulin"].fillna(df.groupby("Glucose")["Insulin"].transform("mean"), inplace=True)
df["Glucose"].fillna(df.groupby("Insulin")["Glucose"].transform("mean"), inplace=True)

df["Glucose"].fillna(df["Glucose"].mean(), inplace=True)
df["Insulin"].fillna(df["Insulin"].mean(), inplace=True)


def replace_with_thresholds(df, col_name):
    low, up = outlier_thresholds(df, col_name)
    df.loc[(df[col_name] > up), col_name] = up
    df.loc[(df[col_name] < low), col_name] = low


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 2: Yeni değişkenler oluşturunuz.

df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_DiabetesPedigreeFunction'] = df['Age'] * df['DiabetesPedigreeFunction']
df['Pregnancies_Per_Age'] = df['Pregnancies'] / df['Age']
df['Glucose_Minus_Insulin'] = df['Glucose'] - df['Insulin']


def categorize_blood_pressure(blood_pressure):
    if blood_pressure < 60:
        return 'Low'
    elif blood_pressure >= 60 and blood_pressure <= 80:
        return 'Normal'
    else:
        return 'High'


df['BloodPressureLevel'] = df['BloodPressure'].apply(categorize_blood_pressure)

df['GIR'] = df['Glucose'] / df['Insulin']


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi >= 18.5 and bmi < 25:
        return 'Normal'
    elif bmi >= 25 and bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


df['BMICategory'] = df['BMI'].apply(categorize_bmi)


def categorize_skin_thickness(skin_thickness):
    if skin_thickness < 10:
        return 'Thin'
    elif skin_thickness >= 10 and skin_thickness < 25:
        return 'Normal'
    else:
        return 'Thick'


df['SkinThicknessCategory'] = df['SkinThickness'].apply(categorize_skin_thickness)


def categorize_age_group(age):
    if age <= 2:
        return 'Babies'
    elif age <= 16:
        return "Children"
    elif age <= 30:
        return 'Young Adults'
    elif age <= 45:
        return 'Middle-aged Adults'
    else:
        return 'Old Adults'


df['AgeGroup'] = df['Age'].apply(categorize_age_group)


def categorize_pregnancy_group(pregnancies):
    if pregnancies == 0:
        return 'No Pregnancy'
    elif pregnancies <= 3:
        return 'Low Pregnancy'
    else:
        return 'High Pregnancy'


df['PregnancyGroup'] = df['Pregnancies'].apply(categorize_pregnancy_group)

df['BMI_DiabetesPedigree'] = df['BMI'] * df['DiabetesPedigreeFunction']
df['Age_Insulin'] = df['Age'] * df['Insulin']
df['Glucose_BMI_Difference'] = df['Glucose'] - df['BMI']

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

binary_col = [col for col in df.columns if
              df[col].dtype not in ["float64", "float32", "float", "int64", "int32", "int"] and df[col].nunique() == 2]


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

ss = StandardScaler()
mm = MinMaxScaler()
rs = RobustScaler()

df[num_cols] = rs.fit_transform(df[num_cols])
# df[num_cols] = mm.fit_transform(df[num_cols])
# df[num_cols] = ss.fit_transform(df[num_cols])

# Adım 5: Model oluşturunuz.


y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(accuracy_score(y_pred, y_test))


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