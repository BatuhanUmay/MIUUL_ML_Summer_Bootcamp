# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

df = sns.load_dataset("tips")
df.head()
   
sns.countplot(x="sex", data=df)
plt.show()

sns.boxplot(x=df["total_bill"])
plt.show()


