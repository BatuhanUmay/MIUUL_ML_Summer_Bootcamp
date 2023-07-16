########################################################################

import seaborn as sns

df = sns.load_dataset('car_crashes')
df_columns = [i for i in df.columns]
num_cols = {i: [df[i].mean(), df[i].min(), df[i].max(), df[i].var()] for i in df.select_dtypes(exclude=['object'])}
########################################################################

num_cols = {col: [df[col].mean(), df[col].min(), df[col].max(), df[col].var()] for col in df.columns if
            df[col].dtype != "O"}
########################################################################

agg_list = ["mean", "min", "max", "var"]
soz = {i: agg_list for i in num_cols}
df[num_cols].agg(soz)
