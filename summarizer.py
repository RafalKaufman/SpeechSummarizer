import re

import pandas as pd

train_data = pd.read_csv("data\\train.csv")
train_data.set_index("id", drop=True, inplace=True)
train_data.drop_duplicates(subset="dialogue", inplace=True)
train_data.dropna(axis=0, inplace=True)
print(train_data.iloc[0, 0])
train_data.iloc[0, 0] = re.sub("#.*#", "", train_data.iloc[0, 0])
print(train_data.iloc[0, 0])
