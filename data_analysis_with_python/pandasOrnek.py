import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

type(df.loc[:3, "age"])
type(df.loc[:3, ["age"]])

df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age": ["mean", "sum"]})
df.groupby("sex").agg({"age": ["mean", "sum"], "survived": ["mean"]})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"], "survived": "mean", "sex": "count"})

df.pivot_table("survived", "sex", "embarked")
df.pivot_table("survived", "sex", "embarked", aggfunc="sum")
df.pivot_table("survived", "sex", ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table("survived", "sex", "new_age")
df.pivot_table("survived", "sex", ["new_age", "class"])


def func(col_name):
    return (col_name / col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("survived")].apply(func).head()
